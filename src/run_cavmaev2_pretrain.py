# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloaderv2 as dataloader
import models
import numpy as np
from traintest_cavmaev2 import train
import neptune
import wandb
from torchviz import make_dot


def load_partial_state_dict(model, state_dict):
    own_state = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    loaded_keys = 0

    for name, param in state_dict.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded_keys += 1
            else:
                missing_keys.append(name)
                print(f"Shape mismatch for {name}: checkpoint {param.shape}, model {own_state[name].shape}")
        else:
            unexpected_keys.append(name)
            # print(f"Skipping {name} as it's not in the model.")

    print(f"Loaded keys: {loaded_keys}")
    print(f"Missing keys: {len(missing_keys)} (due to shape mismatch or not being in the model)")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    return missing_keys, unexpected_keys

def adapt_key(key):
    # Replace 'encoder' with 'blocks_v' and adapt other necessary parts of the key
    new_key = key.replace('encoder.', 'module.blocks_v.')
    # Additional adjustments might be necessary depending on further structure differences
    return new_key


# pretrain cav-mae model

run = neptune.init_run(
    project="junioroteia/CAV-MAE",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGE4NDA2NS1hYmE2LTQ3YWYtODllMC02ODk4NGNlODY0MDUifQ==",
)  # your credentials

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt"])
parser.add_argument("--dataset_mean", type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments, only for preliminary experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--pretrain_video_path", type=str, default='None', help="pretrained video model path")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=None)
parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
parser.add_argument("--mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])

parser.add_argument('--wandb_name', type=str, default=None, help="wandb name")


args = parser.parse_args()

im_res = 224
audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res}


# Extract command-line arguments into a dictionary
cmd_args = vars(args)

# Flatten the audio_conf and val_audio_conf into the main dictionary
# Prefixing each key with 'audio_conf_' or 'val_audio_conf_' to avoid key conflicts
for key in ['num_mel_bins', 'target_length', 'freqm', 'timem', 'mixup', 'dataset', 'mode', 'mean', 'std', 'noise', 'label_smooth', 'im_res']:
    cmd_args[f'audio_conf_{key}'] = cmd_args[key] if key in cmd_args else None
    cmd_args[f'val_audio_conf_{key}'] = cmd_args[key] if key in cmd_args else None

# The 'im_res' value should be set separately if it's not a command-line argument
if 'im_res' not in cmd_args:
    cmd_args['audio_conf_im_res'] = 224
    cmd_args['val_audio_conf_im_res'] = 224

params = cmd_args

wandb.init(project="cavmaev2", entity="edsonroteia", name=args.wandb_name, config=params) 

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))

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
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.model == 'cav-mae':
    print('pretrain a cav-mae model with 11 modality-specific layers and 1 modality-sharing layers')
    audio_model = models.CAVMAEv2(audio_length=args.target_length, norm_pix_loss=args.norm_pix_loss, modality_specific_depth=11, tr_pos=args.tr_pos)
else:
    raise ValueError('model not supported')

# Load the pretrained weights
if args.pretrain_path != 'None':
    mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
    video_mae_weight = torch.load(args.pretrain_video_path, map_location=torch.device('cpu'))
    
    # Adapt keys in video_mae_weight
    adapted_video_weights = {adapt_key(k): v for k, v in video_mae_weight['model'].items()}

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)

    # Load model weights
    miss, unexpected = load_partial_state_dict(audio_model, mdl_weight)
    miss_video, unexpected_video = load_partial_state_dict(audio_model, adapted_video_weights)
    
    print('now load mae pretrained weights from ', args.pretrain_path)
    print('now load video pretrained weights from ', args.pretrain_video_path)
    # Optionally, print missing and unexpected keys
    print('Missing keys:', miss_video)
    print('Unexpected keys:', unexpected_video)

# if args.cont_model != None:
#     print('now load pretrained weights from : ' + args.cont_model)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sdA = torch.load(args.cont_model, map_location=device)
#     if isinstance(audio_model, torch.nn.DataParallel) == False:
#         audio_model = torch.nn.DataParallel(audio_model)
#     audio_model.load_state_dict(sdA, strict=True)

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
print(args)
train(audio_model, train_loader, val_loader, args)
run.stop()