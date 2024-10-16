# -*- coding: utf-8 -*-
# @Time    : 3/12/23 10:23 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : retrieval.py

# supervised (cav-mae pretrain and evaluated on AS) and zero-shot (cav-mae pretrained on AS-2M and eval retrieval on VGGSound) retrieval experiments

import argparse
import os
import models
import dataloader as dataloader
import dataloader_sync
from dataloader_sync import train_collate_fn
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch import nn
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from tabulate import tabulate
import argparse

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

# get mean
def get_sim_mat(a, b):
    B = a.shape[0]
    sim_mat = np.empty([B, B])
    for i in range(B):
        for j in range(B):
            sim_mat[i, j] = get_similarity(a[i, :], b[j, :])
    return sim_mat

def get_agg_sim_mat(a, b, strategy='max'):
    # a and b are tensors of shape (batch_size, num_frames, feature_dim)
    # e.g., torch.Size([100, 10, 768])
    B = a.shape[0]
    num_frames = a.shape[1]  # All videos have the same number of frames
    sim_mat = np.empty([B, B])
    
    for i in range(B):
        for j in range(B):
            # Compute similarity for all frame pairs
            if strategy == 'max' or strategy == 'mean':    
                frame_similarities = np.array([[get_similarity(a[i, k], b[j, l]) for l in range(num_frames)] for k in range(num_frames)])
            # Aggregate similarities based on the chosen strategy
            if strategy == 'max':
                sim_mat[i, j] = np.max(frame_similarities)
            elif strategy == 'mean':
                sim_mat[i, j] = np.mean(frame_similarities)
            elif strategy == 'diagonal_mean':
                # Compare elements from the diagonal of the video submatrix
                diagonal_similarities = [get_similarity(a[i, k], b[j, k]) for k in range(num_frames)]
                sim_mat[i, j] = np.mean(diagonal_similarities)
            elif strategy == 'diagonal_max':
                # Compare elements from the diagonal of the video submatrix
                diagonal_similarities = [get_similarity(a[i, k], b[j, k]) for k in range(num_frames)]
                sim_mat[i, j] = np.max(diagonal_similarities)
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    return sim_mat

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1
    return metrics

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

# direction: 'audio' means audio->visual retrieval, 'video' means visual->audio retrieval
def get_retrieval_result(audio_model, val_loader, direction='audio', model_type='pretrain', strategy='max', cls_token=False, local_matching=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        # Add tqdm progress bar
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            if 'sync' in model_type:
                a_input, v_input, labels, video_id, frame_indices = batch
            else:
                (a_input, v_input, labels) = batch
            if i == 0:
                print("A_shape", a_input.shape)
                print("V_shape", v_input.shape)
            if 'sync' in model_type:
                # flatten batch so we process all frames at the same time
                a_input = a_input.reshape(a_input.shape[0] * a_input.shape[1], a_input.shape[2], a_input.shape[3])
                v_input = v_input.reshape(v_input.shape[0] * v_input.shape[1], v_input.shape[2], v_input.shape[3], v_input.shape[4])

            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                if cls_token:
                    tokens_audio_output, tokens_video_output, cls_audio_output, cls_video_output = audio_model.module.forward_feat(audio_input, video_input)
                    if local_matching:
                        audio_output = torch.mean(tokens_audio_output, dim=1)
                        video_output = torch.mean(tokens_video_output, dim=1)
                    else:
                        audio_output = cls_audio_output
                        video_output = cls_video_output
                else:
                    # mean pool all patches
                    audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                    audio_output = torch.mean(audio_output, dim=1)
                    video_output = torch.mean(video_output, dim=1)
                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            
            if 'sync' in model_type:
                # Group features from the same video together
                num_frames = audio_output.shape[0] // len(video_id)
                audio_output = audio_output.view(len(video_id), num_frames, -1)
                video_output = video_output.view(len(video_id), num_frames, -1)
            
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)
    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    if direction == 'audio':
        # audio->visual retrieval
        sim_mat = get_agg_sim_mat(A_a_feat, A_v_feat, strategy=strategy) if 'sync' in model_type else get_sim_mat(A_a_feat, A_v_feat)
    elif direction == 'video':
        # visual->audio retrieval
        sim_mat = get_agg_sim_mat(A_v_feat, A_a_feat, strategy=strategy) if 'sync' in model_type else get_sim_mat(A_v_feat, A_a_feat)
    result = compute_metrics(sim_mat)
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def eval_retrieval(model, data, audio_conf, label_csv, direction, num_class, model_type='pretrain', batch_size=48, strategy='max', num_register_tokens=4, cls_token=False, local_matching=False):
    print(model)
    print(data)
    frame_use = 5
    # eval setting
    val_audio_conf = audio_conf
    val_audio_conf['frame_use'] = frame_use
    if 'sync' in model_type:
        val_loader = torch.utils.data.DataLoader(dataloader_sync.AudiosetDataset(data, label_csv=label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, collate_fn=train_collate_fn)
    else:
        val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(data, label_csv=label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    # cav-mae only been ssl pretrained

    if model_type == 'sync_pretrain_registers':
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=num_register_tokens, total_frame=audio_conf['total_frame'])
    elif model_type == 'sync_pretrain_registers_cls':
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=num_register_tokens, cls_token=True, total_frame=audio_conf['total_frame'])
    elif model_type == 'sync_pretrain_registers_cls_global_local':
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=num_register_tokens, cls_token=True, global_local_losses=True, total_frame=audio_conf['total_frame'])
    elif model_type == 'sync_pretrain':
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11, num_register_tokens=0, total_frame=audio_conf['total_frame'])
    elif model_type == 'pretrain':
        audio_model = models.CAVMAE(modality_specific_depth=11)
    # cav-mae only been ssl pretrained + supervisedly finetuned
    elif model_type == 'finetune':
        audio_model = models.CAVMAEFT(label_dim=num_class, modality_specific_depth=11)
    sdA = torch.load(model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    audio_model.eval()
    r1, r5, r10, mr = get_retrieval_result(audio_model, val_loader, direction, model_type, strategy, cls_token, local_matching)
    return r1 * 100, r5 * 100, r10 * 100, mr

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Toy example for argument parsing')
    
    parser.add_argument('--dataset', type=str, choices=['audioset', 'vggsound'], 
                        help='Dataset to use for retrieval')
    parser.add_argument('--strategy', type=str, 
                        help='Strategy for aggregation')
    parser.add_argument('--directions', type=str, nargs='+', 
                        help='Directions for evaluation')
    parser.add_argument('--nums_samples', type=int, nargs='+', 
                        help='Number of samples to test')
    args = parser.parse_args()

    # Print out the parsed arguments
    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print(f"Directions: {args.directions}")
    print(f"Number of Samples: {args.nums_samples}")

    dataset = args.dataset
    strategy = args.strategy
    directions = args.directions
    nums_samples = args.nums_samples

    # Hardcoded values for model paths (you may want to add these as command-line arguments in the future)
    model_names = {
        # 'model_1626_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs256-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_023019/models/audio_model.25.pth',
        # 'model_1626_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs256-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_023019/models/best_audio_model.pth',
        # 'model_1625_25': '/scratch/ssml/araujo/ICLR_exps/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs256-normTrue-c0.05-p1.0-tpFalse-mr-unstructured-0.75-20240912_022319/models/audio_model.25.pth',
        # 'model_1625_best': '/scratch/ssml/araujo/ICLR_exps/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs256-normTrue-c0.05-p1.0-tpFalse-mr-unstructured-0.75-20240912_022319/models/best_audio_model.pth',
        # 'model_1558_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75-20240910_082139/models/audio_model.25.pth',
        # 'model_1558_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75-20240910_082139/models/best_audio_model.pth',
        # 'model_1628_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs512-normTrue-c0.5-p1.0-tpFalse-mr-unstructured-0.75-20240912_024238/models/audio_model.25.pth',
        # 'model_1628_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs512-normTrue-c0.5-p1.0-tpFalse-mr-unstructured-0.75-20240912_024238/models/best_audio_model.pth',
        # 'model_1624_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_021700/models/audio_model.25.pth',
        # 'model_1624_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_021700/models/best_audio_model.pth',
        # 'model_1627_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_024000/models/audio_model.25.pth',
        # 'model_1627_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_024000/models/best_audio_model.pth',
        # 'model_1794_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs512-normTrue-c0.5-p1.0-tpFalse-mr-unstructured-0.75-20240915_010633/models/audio_model.21.pth',
        # 'model_1794_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr5e-4-epoch25-bs512-normTrue-c0.5-p1.0-tpFalse-mr-unstructured-0.75-20240915_010633/models/best_audio_model.pth',
        # 'model_1890_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240918_185818/models/best_audio_model.pth',
        # 'model_1890_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240918_185818/models/audio_model.25.pth',   
        # 'model_1921_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240920_204943/models/audio_model.25.pth',
        # 'model_1921_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240920_204943/models/best_audio_model.pth',
        # 'model_1919_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.05-p1.0-tpFalse-mr-unstructured-0.75-20240920_201244/models/audio_model.25.pth',
        # 'model_1919_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.05-p1.0-tpFalse-mr-unstructured-0.75-20240920_201244/models/best_audio_model.pth',
        # 'model_1920_25': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr4e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240920_201543/models/audio_model.25.pth',
        # 'model_1920_best': '/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr4e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240920_201543/models/best_audio_model.pth',
        # 'model_1919_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.05-p1.0-tpFalse-mr-unstructured-0.75-20240920_201244/models/audio_model.25.pth', 'sync_pretrain_registers'),
        # 'model_1970_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240922_181136/models/audio_model.25.pth', 'sync_pretrain_registers'),
        # 'model_1983_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240922_222603/models/audio_model.25.pth', 'sync_pretrain_registers_cls'),
        # 'model_1984_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.05-p1.0-tpFalse-mr-unstructured-0.75-20240922_222719/models/audio_model.25.pth', 'sync_pretrain_registers_cls'),
        # 'model_2145_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240925_112229/models/audio_model.25.pth', 'sync_pretrain_registers_cls_global_local'),
        # 'model_2145_25_local': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240925_112229/models/audio_model.25.pth', 'sync_pretrain_registers_cls_global_local'),
        # 'model_2145_25_both': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240925_112229/models/audio_model.25.pth', 'sync_pretrain_registers_cls_global_local'),
        # 'cav_mae++': ('/local/1314365/code/cav-mae/cav-mae-scale++.pth', 'pretrain'),
        # 'cav_mae+': ('/local/1314365/code/cav-mae/cav-mae-scale+.pth', 'pretrain'),
        # 'model_2618_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20241012_183505/models/audio_model.25.pth', 'sync_pretrain'),
        'model_2625_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20241012_184319/models/audio_model.25.pth', 'sync_pretrain'),
        'model_2626_25': ('/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20241012_184455/models/audio_model.25.pth', 'sync_pretrain'),
        }
    
    if len(model_names) == 0:
        print("Model names dictionary is empty. Searching for models in /scratch/ssml/araujo/exp/")
        base_dir = '/scratch/ssml/araujo/exp/'
        model_names = {}
        
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == 'best_audio_model.pth':
                    full_path = os.path.join(root, file)
                    timestamp = root.split('-')[-1]  # Extract timestamp from directory name
                    model_names[timestamp] = full_path

        if not model_names:
            print("No models found. Exiting.")
            exit(1)
        
        print(f"Found {len(model_names)} models to evaluate.")



    res = []

    if dataset == "audioset":
        data = 'datafilles/audioset_20k/cluster_nodes/audioset_eval_5_per_class_for_retrieval_cleaned.json'
        label_csv = 'datafilles/audioset_20k/cluster_nodes/class_labels_indices.csv'
        num_class = 527
    elif dataset == "vggsound":
        data = 'datafilles/vggsound/cluster_nodes/vgg_test_5_per_class_for_retrieval_cleaned.json'
        label_csv = 'datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv'
        num_class = 309
    else:
        print(f"Unsupported dataset: {dataset}")
        exit(1)

    for num_samples in tqdm(nums_samples, desc="Testing sample sizes"):
        for model_name, (model_path, model_type) in tqdm(model_names.items(), desc=f"Processing models for {dataset}", leave=False):
            if 'sync' in model_type:
                if '2625' in model_name:
                    target_length = 192
                elif '2626' in model_name:
                    target_length = 512
                else:
                    target_length = 96
            else:
                target_length = 1024

            if 'cls' in model_type:
                cls_token = True
            else:
                cls_token = False
            
            for direction in tqdm(directions, desc="Evaluating directions", leave=False):
                audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                            'mode': 'retrieval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5, 'num_samples': num_samples, 'total_frame': 16}
                if 'local' in model_name:
                    r1, r5, r10, mr = eval_retrieval(model_path, data, audio_conf=audio_conf, label_csv=label_csv, num_class=num_class, direction=direction, model_type=model_type, batch_size=100, strategy=strategy, num_register_tokens=8 if '1970' in model_name else 4, cls_token=cls_token, local_matching=True)
                else:
                    r1, r5, r10, mr = eval_retrieval(model_path, data, audio_conf=audio_conf, label_csv=label_csv, num_class=num_class, direction=direction, model_type=model_type, batch_size=100, strategy=strategy, num_register_tokens=8 if '1970' in model_name else 4, cls_token=cls_token, local_matching=False)
                res.append([model_name, dataset, direction, num_samples, r1, r5, r10, mr])
                res_sorted = sorted(res, key=lambda x: x[-1])  # Sort by MR
                print("\nCurrent Results Table:")
                print(tabulate(res_sorted, headers=["Model", "Dataset", "Direction", "Num Samples", "R@1", "R@5", "R@10", "MR"]))

    np.savetxt('./retrieval_result.csv', res, delimiter=',', fmt='%s')
