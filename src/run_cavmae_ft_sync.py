# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

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
import dataloader_sync as dataloader
import models
import numpy as np
import warnings
import json
from sklearn import metrics
from traintest_ft_sync import train, validate
from dataloader_sync import eval_collate_fn, train_collate_fn
import neptune
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# finetune cav-mae model

run = neptune.init_run(
    project="junioroteia/CAV-MAE",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGE4NDA2NS1hYmE2LTQ3YWYtODllMC02ODk4NGNlODY0MDUifQ==",
    tags=["finetuning"],
)  # your credentials

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
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
# not used in the formal experiments, only in preliminary experiments
parser.add_argument("--lr_patience", type=int, default=1, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lr_scheduler", type=str, default="step", help="learning rate scheduler", choices=["step", "cosine"])

parser.add_argument("--lrscheduler_start", default=2, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=1, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)

parser.add_argument("--wa", help='if do weight averaging in finetuning', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=1, help="which epoch to start weight averaging in finetuning")
parser.add_argument("--wa_end", type=int, default=10, help="which epoch to end weight averaging in finetuning")
parser.add_argument("--wa_interval", type=int, default=1, help="interval for weight averaging")


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
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (default: use all samples)")
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--aggregate", type=str, default="None")
parser.add_argument("--n_regster_tokens", type=int, default=4)

args = parser.parse_args()

# all exp in this work is based on 224 * 224 image
im_res = 224

if args.aggregate != "None":
    mode = 'eval'
else:
    mode = 'train'

audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
              'dataset': args.dataset, 'mode':mode, 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res, 'num_samples': args.num_samples}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':mode, 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res, 'num_samples': args.num_samples}

def get_loader(args, audio_conf, val_audio_conf, train_csv, val_csv):
    print('Now process ' + args.dataset)
    train_dataset = dataloader.AudiosetDataset(train_csv, audio_conf, label_csv=args.label_csv)
    val_dataset = dataloader.AudiosetDataset(val_csv, val_audio_conf, label_csv=args.label_csv)
    
    print('Number of training samples: {}'.format(len(train_dataset)))
    print('Number of validation samples: {}'.format(len(val_dataset)))

    if args.aggregate != "None":
        collate_fn = eval_collate_fn
    else:
        collate_fn = train_collate_fn

    if args.bal == 'bal':
        print('Using balanced sampler')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        if args.num_samples is not None:
            samples_weight = samples_weight[:args.num_samples]
            if len(samples_weight) != len(train_dataset):
                raise ValueError(f"Number of weights ({len(samples_weight)}) does not match number of samples in dataset ({len(train_dataset)})")
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, 
            pin_memory=True, drop_last=True, collate_fn=collate_fn)
    else:
        print('Using random sampler')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
        pin_memory=True, drop_last=True, collate_fn=collate_fn)
    
    return train_loader, val_loader

train_loader, val_loader = get_loader(args, audio_conf, val_audio_conf, args.data_train, args.data_val)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.model == 'cav-mae-ft':
    print('finetune a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
    audio_model = models.CAVMAEFTSync(audio_length=args.target_length, label_dim=args.n_class, modality_specific_depth=11, aggregate=args.aggregate, num_register_tokens=args.n_regster_tokens)
else:
    raise ValueError('model not supported')

if args.pretrain_path == 'None':
    warnings.warn("Note you are finetuning a model without any finetuning.")

# finetune based on a CAV-MAE pretrained model, which is the default setting unless for ablation study
if args.pretrain_path != 'None':
    # TODO: change this to a wget link
    mdl_weight = torch.load(args.pretrain_path)
       
    # Remove the mismatched keys
    keys_to_remove = ['module.pos_embed_a', 'module.decoder_pos_embed_a']
    for key in keys_to_remove:
        if key in mdl_weight:
            del mdl_weight[key]
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print(f"Missing keys: {miss}")
    print(f"Unexpected keys: {unexpected}")
    print('now load cav-mae pretrained weights from ', args.pretrain_path)
    print(miss, unexpected)

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
train(audio_model, train_loader, val_loader, args, run)

# average the model weights of checkpoints, note it is not ensemble, and does not increase computational overhead
def wa_model(exp_dir, start_epoch, end_epoch, interval):
    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location='cpu')
    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1, interval):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location='cpu')
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1
    print('wa {:d} models: {}'.format(model_cnt, range(start_epoch+1, end_epoch+1, interval)))
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)
    return sdA


results = []  # List to store results

for wa_start in range(args.wa_start, args.n_epochs, 5):
    for wa_interval in [1, 3, 5]:
        # evaluate with multiple frames
        if not isinstance(audio_model, torch.nn.DataParallel):
            audio_model = torch.nn.DataParallel(audio_model)
        if args.wa:
            sdA = wa_model(args.exp_dir, start_epoch=wa_start, end_epoch=args.wa_end, interval=wa_interval)
            torch.save(sdA, args.exp_dir + "/models/audio_model_wa_{}.pth".format(wa_interval))
        else:
            # if no wa, use the best checkpoint
            sdA = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location='cpu')
        msg = audio_model.load_state_dict(sdA, strict=True)
        print(msg)
        audio_model.eval()

        # skip multi-frame evaluation, for audio-only model
        if args.skip_frame_agg:
            val_audio_conf['frame_use'] = 5
            stats, _, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
            if args.metrics == 'mAP':
                cur_res = np.mean([stat['AP'] for stat in stats])
                print('mAP is {:.4f}'.format(cur_res))
            elif args.metrics == 'acc':
                cur_res = stats[0]['acc']
                print('acc is {:.4f}'.format(cur_res))
        else:
            # Validate the model and get outputs
            stats, _, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
            
            # Apply softmax or sigmoid based on the evaluation metric
            if args.metrics == 'acc':
                audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
            elif args.metrics == 'mAP':
                audio_output = torch.nn.functional.sigmoid(audio_output.float())

            # Convert outputs to numpy arrays
            audio_output, target = audio_output.numpy(), target.numpy()
            
            # Calculate and store the current frame's performance
            if args.metrics == 'mAP':
                cur_res = np.mean([stat['AP'] for stat in stats])
                print('final mAP is {:.4f}'.format(cur_res))
            elif args.metrics == 'acc':
                cur_res = stats[0]['acc']
                print('final acc is {:.4f}'.format(cur_res))

        # Save the results with the corresponding wa configuration
        results.append((f"wa_start: {wa_start}, wa_interval: {wa_interval}", cur_res))

# Print results in a table format
print("\nResults Summary:")
print("{:<30} {:<10}".format("Models Aggregated", "Final Result"))
for model_info, result in results:
    print("{:<30} {:<10.4f}".format(model_info, result))

import pandas as pd
# Log the results summary table to Neptune
table_data = {
    "Models Aggregated": [model_info for model_info, _ in results],
    "Final Result": [result for _, result in results]
}
run["results/summary"].upload(neptune.types.File.as_html(pd.DataFrame(table_data)))
            