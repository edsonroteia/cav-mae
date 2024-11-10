# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import sys
import time
import torch
import dataloader_sync as dataloader
from dataloader_sync import eval_collate_fn
import models
from extract_features_sync import extract_features

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Keep only necessary arguments
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--dataset", type=str, default="audioset", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400"])
parser.add_argument("--dataset_mean", type=float, help="dataset mean for normalization")
parser.add_argument("--dataset_std", type=float, help="dataset std for normalization")
parser.add_argument("--target_length", type=int, help="input length in frames")
parser.add_argument("--noise", help='use noise', type=ast.literal_eval)
parser.add_argument("--exp-dir", type=str, default="", help="experiment directory")
parser.add_argument('-b', '--batch-size', default=48, type=int)
parser.add_argument('-w', '--num-workers', default=32, type=int)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--aggregate", type=str, default="None")
parser.add_argument("--n_register_tokens", type=int, default=4)
parser.add_argument("--cls_token", type=ast.literal_eval, default=True)
parser.add_argument("--total_frame", type=int, default=16)
parser.add_argument("--model_id", type=int, default=None)
parser.add_argument("--contrastive_head", type=ast.literal_eval, default=False)
parser.add_argument("--joint_layers", type=int, default=1)
parser.add_argument("--keep_register_tokens", type=ast.literal_eval, default=False)
args = parser.parse_args()

# Setup configs
im_res = 224
mode = 'retrieval'

audio_conf = {
    'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
    'dataset': args.dataset, 'mode': mode, 'mean': args.dataset_mean, 'std': args.dataset_std,
    'noise': args.noise, 'im_res': im_res, 'augmentation': False, 'total_frame': args.total_frame
}

# Initialize model and load weights
audio_model = models.CAVMAEFTSync(
    audio_length=args.target_length, 
    label_dim=args.n_class, 
    modality_specific_depth=11, 
    aggregate=args.aggregate,
    num_register_tokens=args.n_register_tokens,
    cls_token=args.cls_token,
    total_frame=args.total_frame,
    contrastive_head=args.contrastive_head,
    joint_layers=args.joint_layers,
    keep_register_tokens=args.keep_register_tokens
)

if args.pretrain_path != 'None':
    mdl_weight = torch.load(args.pretrain_path)
    keys_to_remove = ['module.pos_embed_a', 'module.decoder_pos_embed_a']
    for key in keys_to_remove:
        mdl_weight.pop(key, None)
    
    audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print(f"Missing keys: {miss}")
    print(f"Unexpected keys: {unexpected}")

# Create dataloaders
train_dataset = dataloader.AudiosetDataset(args.data_train, audio_conf, label_csv=args.label_csv)
val_dataset = dataloader.AudiosetDataset(args.data_val, audio_conf, label_csv=args.label_csv)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, collate_fn=eval_collate_fn   
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True, collate_fn=eval_collate_fn   
)

# Extract features
print("Extracting training features...")
train_features, train_labels = extract_features(audio_model, train_loader, args, 'train', args.model_id)

print("Extracting validation features...")
val_features, val_labels = extract_features(audio_model, val_loader, args, 'val', args.model_id)
