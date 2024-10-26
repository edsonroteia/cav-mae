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
from tqdm import tqdm
from itertools import product
from tqdm.auto import tqdm
import torch.nn.functional as F
import h5py  # Add this import at the top

# finetune cav-mae model



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

parser.add_argument("--label_smooth", type=float, default=0.0, help="label smoothing factor")
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--ftmode", type=str, default='multimodal', help="how to fine-tune the model")

parser.add_argument("--head_lr", type=float, default=50.0, help="learning rate ratio the newly initialized layers / pretrained weights")
parser.add_argument('--freeze_base', help='freeze the backbone or not', type=ast.literal_eval)
parser.add_argument('--skip_frame_agg', help='if do frame agg', type=ast.literal_eval)
parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (default: use all samples)")
parser.add_argument("--wandb_name", type=str, default=None)
parser.add_argument("--aggregate", type=str, default="None")
parser.add_argument("--n_register_tokens", type=int, default=4)
parser.add_argument("--augmentation", type=ast.literal_eval, default=True)

parser.add_argument("--neptune_tag", type=str, default="finetuning")
parser.add_argument("--cls_token", type=ast.literal_eval, default=True)
parser.add_argument("--total_frame", type=int, default=16)
parser.add_argument("--knn_k", type=int, default=10)

args = parser.parse_args()

run = neptune.init_run(
    project="junioroteia/CAV-MAE",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmNGE4NDA2NS1hYmE2LTQ3YWYtODllMC02ODk4NGNlODY0MDUifQ==",
    tags=["finetuning", args.neptune_tag],
)  # your credentials

# Add these variables after initializing the Neptune run
run["best_val_mAP"] = 0
run["best_val_ACC"] = 0
run["last_val_mAP"] = 0
run["last_val_ACC"] = 0

# all exp in this work is based on 224 * 224 image
im_res = 224

if args.aggregate != "None":
    mode = 'retrieval'
else:
    mode = 'train'

audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup,
              'dataset': args.dataset, 'mode':mode, 'mean':args.dataset_mean, 'std':args.dataset_std,
              'noise':args.noise, 'label_smooth': args.label_smooth, 'im_res': im_res, 'num_samples': args.num_samples, 'augmentation': args.augmentation, 'total_frame': args.total_frame}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':mode, 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': im_res, 'num_samples': args.num_samples, 'augmentation': False, 'total_frame': args.total_frame}

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
    audio_model = models.CAVMAEFTSync(audio_length=args.target_length, label_dim=args.n_class, modality_specific_depth=11, aggregate=args.aggregate, num_register_tokens=args.n_register_tokens, cls_token=args.cls_token, total_frame=args.total_frame)
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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_model = audio_model.to(device)
audio_model.eval()

# Function to extract features
def extract_features(loader):
    features = []
    labels = []
    video_ids = []  # To track which video each feature belongs to
    current_video_id = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features"):
            a_input, v_input, label, _, _ = batch
            a_input, v_input = a_input.to(device), v_input.to(device)
            
            # Get batch size and number of frames
            true_batch_size = label.size(0) // 10
            
            audio_outputs, visual_outputs = audio_model.module.forward_feat(a_input, v_input, 'av')
            # Concatenate the audio and visual features
            combined_features = torch.cat((audio_outputs.mean(dim=1), visual_outputs.mean(dim=1)), dim=1).squeeze(1)
            
            # Save each feature individually with its video ID
            for i in range(label.size(0)):
                video_ids.append(current_video_id + (i // 10))  # Integer division by 10 gives video ID
            
            features.append(combined_features.cpu())
            labels.append(label.cpu())
            
            current_video_id += true_batch_size
            
    features = torch.cat(features)
    labels = torch.cat(labels)
    video_ids = torch.tensor(video_ids)
    
    # Save everything to disk
    save_dict = {
        'features': features,
        'labels': labels,
        'video_ids': video_ids
    }
    torch.save(save_dict, os.path.join(args.exp_dir, 'raw_features.pth'))
    return save_dict

def aggregate_features(features_path):
    """
    Post-process the saved features to aggregate them by video ID
    """
    data = torch.load(features_path)
    features = data['features']
    labels = data['labels']
    video_ids = data['video_ids']
    
    unique_video_ids = torch.unique(video_ids)
    aggregated_features = []
    aggregated_labels = []
    
    for vid in tqdm(unique_video_ids, desc="Aggregating features"):
        # Get all features for this video
        vid_mask = video_ids == vid
        vid_features = features[vid_mask]
        
        # Concatenate all features for this video
        aggregated_features.append(vid_features.reshape(-1))
        
        # Take just one label (they're all the same for the video)
        aggregated_labels.append(labels[vid_mask][0])
    
    aggregated_features = torch.stack(aggregated_features)
    aggregated_labels = torch.stack(aggregated_labels)
    prefix = os.path.basename(features_path).split('.')[0].split('_')[0]
    print(f"Saving aggregated features for {prefix}...")
    # Save aggregated features
    torch.save({
        'features': aggregated_features,
        'labels': aggregated_labels
    }, os.path.join(os.path.dirname(features_path), f'{prefix}_aggregated_features.pth'))
    
    return aggregated_features, aggregated_labels

# Remove the H5 related functions and replace with torch-based functions
def save_features(features_dict, filepath):
    """Save features dictionary to torch file"""
    torch.save(features_dict, filepath)
    print(f"Features saved to {filepath}")

def load_features(filepath):
    """Load features dictionary from torch file"""
    data = torch.load(filepath)
    print(f"Features loaded from {filepath}")
    return data

# Modify the feature extraction section
feature_cache_dir = 'features_{}'.format(args.num_samples)
os.makedirs(feature_cache_dir, exist_ok=True)

train_feature_path = os.path.join(feature_cache_dir, 'train_raw_features.pth')
val_feature_path = os.path.join(feature_cache_dir, 'val_raw_features.pth')

# Extract or load training features
if os.path.exists(train_feature_path):
    user_input = input(f"Training features file {train_feature_path} already exists. Do you want to replace it? (y/n): ")
    if user_input.lower() == 'y':
        print("Extracting features for training set...")
        train_data = extract_features(train_loader)
        save_features(train_data, train_feature_path)
        print("Aggregating training features...")
        train_features, train_labels = aggregate_features(train_feature_path)
    else:
        print("Loading cached training features...")
        if os.path.exists(os.path.join(feature_cache_dir, 'train_aggregated_features.pth')):
            train_data = load_features(os.path.join(feature_cache_dir, 'train_aggregated_features.pth'))
            train_features, train_labels = train_data['features'], train_data['labels']
        else:
            print("Aggregating training features...")
            train_features, train_labels = aggregate_features(train_feature_path)
else:
    print("Extracting features for training set...")
    train_data = extract_features(train_loader)
    save_features(train_data, train_feature_path)
    print("Aggregating training features...")
    train_features, train_labels = aggregate_features(train_feature_path)

# Extract or load validation features
if os.path.exists(val_feature_path):
    user_input = input(f"Validation features file {val_feature_path} already exists. Do you want to replace it? (y/n): ")
    if user_input.lower() == 'y':
        print("Extracting features for validation set...")
        val_data = extract_features(val_loader)
        save_features(val_data, val_feature_path)
        print("Aggregating validation features...")
        val_features, val_labels = aggregate_features(val_feature_path)
    else:
        print("Loading cached validation features...")
        if os.path.exists(os.path.join(feature_cache_dir, 'val_aggregated_features.pth')):
            val_data = load_features(os.path.join(feature_cache_dir, 'val_aggregated_features.pth'))
            val_features, val_labels = val_data['features'], val_data['labels']
        else:
            print("Aggregating validation features...")
            val_features, val_labels = aggregate_features(val_feature_path)
else:
    print("Extracting features for validation set...")
    val_data = extract_features(val_loader)
    save_features(val_data, val_feature_path)
    print("Aggregating validation features...")
    val_features, val_labels = aggregate_features(val_feature_path)

# Modify the KNN classifier implementation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score, accuracy_score

class MultilabelKNN:
    def __init__(self, k, metric='euclidean', weights='uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
        
    def predict_proba(self, X):
        # Calculate distances based on the chosen metric
        if self.metric == 'cosine':
            # Normalize the vectors for cosine similarity
            X_norm = F.normalize(torch.Tensor(X), p=2, dim=1)
            X_train_norm = F.normalize(torch.Tensor(self.X_train), p=2, dim=1)
            distances = 1 - torch.mm(X_norm, X_train_norm.t())
        elif self.metric == 'manhattan':
            distances = torch.cdist(torch.Tensor(X), torch.Tensor(self.X_train), p=1)
        else:  # euclidean
            distances = torch.cdist(torch.Tensor(X), torch.Tensor(self.X_train), p=2)
        
        # Get k nearest neighbors
        _, indices = distances.topk(k=self.k, dim=1, largest=False)
        neighbor_labels = self.y_train[indices]
        
        if self.weights == 'distance':
            # Weight by inverse distance
            weights = 1.0 / (distances.gather(1, indices) + 1e-8)
            weights = weights.unsqueeze(-1).expand_as(neighbor_labels)
            probas = (neighbor_labels.float() * weights).sum(dim=1) / weights.sum(dim=1)
        else:  # uniform weights
            probas = neighbor_labels.float().mean(dim=1)
            
        return probas.numpy()

# Replace the original KNN classifier with multilabel version
print("Training multilabel KNN classifier...")
knn = MultilabelKNN(k=args.knn_k)
knn.fit(train_features, train_labels)

# Make predictions - now keeps the multilabel format
val_pred_proba = knn.predict_proba(val_features)

# Calculate metrics for multilabel classification
val_mAP = average_precision_score(val_labels.numpy(), val_pred_proba, average='macro')

# For multilabel accuracy, we'll use a threshold of 0.5
val_pred_binary = (val_pred_proba >= 0.5).astype(float)
val_acc = (val_pred_binary == val_labels.numpy()).mean()

# Add some debugging information
# print(f"Number of classes in KNN predictions: {len(knn.classes_)}")
# print(f"Shape of prediction matrix: {val_pred_proba.shape}")
# print(f"Shape of validation labels: {val_labels.shape}")

print(f"Validation mAP: {val_mAP:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Log metrics to Neptune
run["val_mAP"] = val_mAP
run["val_ACC"] = val_acc

# Save results
results = {
    "val_mAP": val_mAP,
    "val_ACC": val_acc,
    "knn_k": args.knn_k,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers,
}

with open(f"{args.exp_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {args.exp_dir}/results.json")

# Define hyperparameter ranges to iterate over
hyperparams = {
    'k_values': [5, 10, 15, 20],
    'distance_metrics': ['euclidean', 'cosine', 'manhattan'],
    'weights': ['uniform', 'distance'],
    'thresholds': [0.3, 0.4, 0.5, 0.6]
}

# Calculate total number of combinations
total_combinations = len(hyperparams['k_values']) * len(hyperparams['distance_metrics']) * \
                    len(hyperparams['weights']) * len(hyperparams['thresholds'])

print(f"\nStarting hyperparameter search with {total_combinations} combinations...")
results = {}

# Create progress bar for all combinations
pbar = tqdm(total=total_combinations, desc='Hyperparameter Search')

# Nested loops for hyperparameter combinations
for k, metric, weight, threshold in product(
    hyperparams['k_values'],
    hyperparams['distance_metrics'],
    hyperparams['weights'],
    hyperparams['thresholds']
):
    param_key = f"k{k}_metric{metric}_weight{weight}_thresh{threshold}"
    
    # Update progress bar description
    pbar.set_description(f"Testing {param_key}")
    
    # Modify MultilabelKNN to include these parameters
    knn = MultilabelKNN(
        k=k,
        metric=metric,
        weights=weight
    )
    knn.fit(train_features, train_labels)
    
    val_pred_proba = knn.predict_proba(val_features)
    
    # Calculate metrics
    val_mAP = average_precision_score(val_labels.numpy(), val_pred_proba, average='macro')
    val_pred_binary = (val_pred_proba >= threshold).astype(float)
    val_acc = (val_pred_binary == val_labels.numpy()).mean()
    
    # Store results
    results[param_key] = {
        "k": k,
        "distance_metric": metric,
        "weight_type": weight,
        "threshold": threshold,
        "val_mAP": val_mAP,
        "val_ACC": val_acc,
    }
    
    # Log metrics to Neptune
    run[f"metrics/{param_key}/val_mAP"] = val_mAP
    run[f"metrics/{param_key}/val_ACC"] = val_acc
    
    # Update progress bar
    pbar.update(1)
    
    # Add current best scores to progress bar postfix
    current_best_map = max(results.values(), key=lambda x: x['val_mAP'])['val_mAP']
    current_best_acc = max(results.values(), key=lambda x: x['val_ACC'])['val_ACC']
    pbar.set_postfix({
        'Best mAP': f"{current_best_map:.4f}",
        'Best ACC': f"{current_best_acc:.4f}"
    })

pbar.close()

# Save all hyperparameter results to a JSON file
with open(f"{args.exp_dir}/knn_hyperparams_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Find and print the best configurations
best_map_config = max(results.items(), key=lambda x: x[1]['val_mAP'])
best_acc_config = max(results.items(), key=lambda x: x[1]['val_ACC'])

print("\nBest mAP Configuration:")
print(f"Parameters: {best_map_config[0]}")
print(f"mAP: {best_map_config[1]['val_mAP']:.4f}")

print("\nBest Accuracy Configuration:")
print(f"Parameters: {best_acc_config[0]}")
print(f"Accuracy: {best_acc_config[1]['val_ACC']:.4f}")

run.stop()