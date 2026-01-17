# -*- coding: utf-8 -*-
# @Time    : 2026-01-17
# @Author  : Based on CAV-MAE by Yuan Gong (MIT)
# @File    : run_cavjepa_ft.py
# @Description: Fine-tuning script for CAV-JEPA pretrained models

import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_ft import train, validate

# Import CAVJEPAFT model (fallback to CAVMAEFT if not available)
try:
    from models.cav_jepa import CAVJEPAFT
    print("Using CAVJEPAFT model")
except ImportError:
    from models.cav_mae import CAVMAEFT as CAVJEPAFT
    print("Warning: CAVJEPAFT not found, falling back to CAVMAEFT")

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='cav-jepa-ft', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400"])
parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=10, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--label_smooth", type=float, default=0.1, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)

args = parser.parse_args()


def load_jepa_pretrain_weights(model, pretrain_path, strict=False):
    """
    Load CAV-JEPA pretrained weights into CAVJEPAFT model.

    The CAVJEPA pretraining model has additional components (target encoder, predictor)
    that are not present in the fine-tuning model. This function filters out those
    extra weights and loads only the context encoder weights.
    """
    print(f"Loading pretrained weights from: {pretrain_path}")

    mdl_weight = torch.load(pretrain_path, map_location='cpu')

    # Handle DataParallel wrapper if present
    if any(k.startswith('module.') for k in mdl_weight.keys()):
        # Remove module. prefix
        mdl_weight = {k.replace('module.', ''): v for k, v in mdl_weight.items()}

    # Filter out target encoder and predictor weights (JEPA-specific components not needed for FT)
    excluded_prefixes = ('target_', 'predictor_')
    filtered_weights = {}
    excluded_keys = []

    for k, v in mdl_weight.items():
        if k.startswith(excluded_prefixes):
            excluded_keys.append(k)
        else:
            filtered_weights[k] = v

    if excluded_keys:
        print(f"Excluded {len(excluded_keys)} JEPA-specific weights (target encoder/predictor)")
        if len(excluded_keys) <= 20:
            for k in excluded_keys:
                print(f"  - {k}")
        else:
            print(f"  First 10: {excluded_keys[:10]}")
            print(f"  ... and {len(excluded_keys) - 10} more")

    # Add module. prefix back if model is DataParallel
    if isinstance(model, torch.nn.DataParallel):
        filtered_weights = {f'module.{k}': v for k, v in filtered_weights.items()}

    # Load weights
    miss, unexpected = model.load_state_dict(filtered_weights, strict=strict)

    print(f"Loaded {len(filtered_weights)} weights")
    if miss:
        print(f"Missing keys ({len(miss)}): {miss[:10]}..." if len(miss) > 10 else f"Missing keys: {miss}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}..." if len(unexpected) > 10 else f"Unexpected keys: {unexpected}")

    return miss, unexpected


# all exp in this work is based on 224 * 224 image
im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
              'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}

if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Create model
if args.model == 'cav-jepa-ft':
    print('finetune a cav-jepa model with 11 modality-specific layers and 1 modality-sharing layers')
    audio_model = CAVJEPAFT(label_dim=args.n_class, modality_specific_depth=11)
elif args.model == 'cav-mae-ft':
    # Fallback for compatibility
    from models.cav_mae import CAVMAEFT
    print('finetune a cav-mae model (compatibility mode)')
    audio_model = CAVMAEFT(label_dim=args.n_class, modality_specific_depth=11)
else:
    raise ValueError(f'model {args.model} not supported')

if args.pretrain_path == 'None':
    warnings.warn("Note you are finetuning a model without any pretraining.")
else:
    # Wrap in DataParallel first
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    # Load pretrained weights
    load_jepa_pretrain_weights(audio_model, args.pretrain_path, strict=False)

print("\nCreating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs("%s/models" % args.exp_dir)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir, start_epoch, end_epoch):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location='cpu')
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location='cpu')
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models from {:d} to {:d}'.format(model_cnt, start_epoch, end_epoch))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA

# evaluate with multiple frames
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)
if args.wa == True:
    sdA = wa_model(args.exp_dir, start_epoch=args.wa_start, end_epoch=args.wa_end)
    torch.save(sdA, args.exp_dir + "/models/audio_model_wa.pth")
else:
    # if no wa, use the best checkpoint
    sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')
msg = audio_model.load_state_dict(sdA, strict=True)
print(msg)
audio_model.eval()

# skip multi-frame evaluation, for audio-only model
if args.skip_frame_agg == True:
    val_audio_conf['frame_use'] = 5
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
    if args.metrics == 'mAP':
        cur_res = np.mean([stat['AP'] for stat in stats])
        print('mAP is {:.4f}'.format(cur_res))
    elif args.metrics == 'acc':
        cur_res = stats[0]['acc']
        print('acc is {:.4f}'.format(cur_res))
else:
    res = []
    multiframe_pred = []
    total_frames = 10 # change if your total frame is different
    for frame in range(total_frames):
        val_audio_conf['frame_use'] = frame
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
        print(audio_output.shape)
        if args.metrics == 'acc':
            audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
        elif args.metrics == 'mAP':
            audio_output = torch.nn.functional.sigmoid(audio_output.float())

        audio_output, target = audio_output.numpy(), target.numpy()
        multiframe_pred.append(audio_output)
        if args.metrics == 'mAP':
            cur_res = np.mean([stat['AP'] for stat in stats])
            print('mAP of frame {:d} is {:.4f}'.format(frame, cur_res))
        elif args.metrics == 'acc':
            cur_res = stats[0]['acc']
            print('acc of frame {:d} is {:.4f}'.format(frame, cur_res))
        res.append(cur_res)

    # ensemble over frames
    multiframe_pred = np.mean(multiframe_pred, axis=0)
    if args.metrics == 'acc':
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(multiframe_pred, 1))
        print('multi-frame acc is {:f}'.format(acc))
        res.append(acc)
    elif args.metrics == 'mAP':
        AP = []
        for k in range(args.n_class):
            # Average precision
            avg_precision = metrics.average_precision_score(target[:, k], multiframe_pred[:, k], average=None)
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print('multi-frame mAP is {:.4f}'.format(mAP))
        res.append(mAP)
    np.savetxt(args.exp_dir + '/mul_frame_res.csv', res, delimiter=',')
