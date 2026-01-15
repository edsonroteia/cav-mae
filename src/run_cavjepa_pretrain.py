# -*- coding: utf-8 -*-
# @Time    : 1/15/26
# @Author  : Based on run_cavmae_pretrain.py by Yuan Gong (MIT)
# @File    : run_cavjepa_pretrain.py
# @Description: Entry point for CAV-JEPA pretraining

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
import dataloader as dataloader
import models
import numpy as np
from traintest_cavjepa import train

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data arguments
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='cav-jepa', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used",
                    choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt"])
parser.add_argument("--dataset_mean", type=float, help="the dataset audio spec mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, help="the dataset audio spec std, used for input normalization")
parser.add_argument("--target_length", type=int, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)

# Training arguments
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

# Data augmentation
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")

# Model initialization
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path (adapted JEPA weights)")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=None)

# JEPA-specific arguments
parser.add_argument("--jepa_loss_weight", type=float, default=1.0, help="weight for JEPA loss")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
parser.add_argument("--momentum_start", type=float, default=0.996, help="EMA momentum start value")
parser.add_argument("--momentum_end", type=float, default=0.999, help="EMA momentum end value")
parser.add_argument("--predictor_depth", type=int, default=4, help="predictor network depth")
parser.add_argument("--predictor_dim", type=int, default=384, help="predictor network dimension")

args = parser.parse_args()

im_res = 224
audio_conf = {
    'num_mel_bins': 128,
    'target_length': args.target_length,
    'freqm': 0,
    'timem': 0,
    'mixup': args.mixup,
    'dataset': args.dataset,
    'mode': 'train',
    'mean': args.dataset_mean,
    'std': args.dataset_std,
    'noise': args.noise,
    'label_smooth': 0,
    'im_res': im_res
}
val_audio_conf = {
    'num_mel_bins': 128,
    'target_length': args.target_length,
    'freqm': 0,
    'timem': 0,
    'mixup': 0,
    'dataset': args.dataset,
    'mode': 'eval',
    'mean': args.dataset_mean,
    'std': args.dataset_std,
    'noise': False,
    'im_res': im_res
}

print('CAV-JEPA Training Configuration:')
print('  JEPA loss weight: {:.3f}'.format(args.jepa_loss_weight))
print('  Contrastive loss weight: {:.3f}'.format(args.contrast_loss_weight))
print('  Masking ratio: {:.3f}'.format(args.masking_ratio))
print('  Momentum: {:.4f} -> {:.4f}'.format(args.momentum_start, args.momentum_end))
print('  Predictor: depth={}, dim={}'.format(args.predictor_depth, args.predictor_dim))

# Data loaders
if args.bal == 'bal':
    print('Balanced sampler is being used')
    if args.weight_file is None:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('Balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval is not None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

# Model creation
if args.model == 'cav-jepa':
    print('Pretraining CAV-JEPA model with 11 modality-specific layers and 1 modality-sharing layer')
    audio_model = models.CAVJEPA(
        audio_length=args.target_length,
        modality_specific_depth=11,
        tr_pos=args.tr_pos,
        predictor_depth=args.predictor_depth,
        predictor_embed_dim=args.predictor_dim,
        momentum_start=args.momentum_start,
        momentum_end=args.momentum_end
    )
else:
    raise ValueError('Model not supported: {}'.format(args.model))

# Load pretrained weights (e.g., adapted I-JEPA/V-JEPA checkpoint)
if args.pretrain_path != 'None':
    mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print('Loaded pretrained weights from:', args.pretrain_path)
    print('Missing keys:', len(miss))
    print('Unexpected keys:', len(unexpected))

# Create experiment directory
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
