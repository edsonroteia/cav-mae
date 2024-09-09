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
import dataloader
import dataloader_sync
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch import nn
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def get_sim_mat(a, b):
    B = a.shape[0]
    sim_mat = np.empty([B, B])
    for i in range(B):
        for j in range(B):
            sim_mat[i, j] = get_similarity(a[i, :], b[j, :])
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

def get_retrieval_result(audio_model, val_loader, direction='audio'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                audio_output = torch.mean(audio_output, dim=1)
                video_output = torch.mean(video_output, dim=1)
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)
    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    if direction == 'audio':
        sim_mat = get_sim_mat(A_a_feat, A_v_feat)
    elif direction == 'video':
        sim_mat = get_sim_mat(A_v_feat, A_a_feat)
    result = compute_metrics(sim_mat)
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def get_sync_retrieval_result(audio_model, val_loader, direction='audio'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    video_ids = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing batches"):
            fbanks, images, _, batch_video_ids, _ = batch
            fbanks, images = fbanks.to(device), images.to(device)
            
            with autocast():
                audio_output, video_output = audio_model.module.forward_feat(fbanks, images)
                
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            
            A_a_feat.append(audio_output.cpu())
            A_v_feat.append(video_output.cpu())
            video_ids.extend(batch_video_ids)

    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)

    if direction == 'audio':
        sim_mat = get_sync_sim_mat(A_a_feat, A_v_feat, video_ids)
    elif direction == 'video':
        sim_mat = get_sync_sim_mat(A_v_feat, A_a_feat, video_ids, is_video_query=True)

    result = compute_metrics(sim_mat)
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def get_sync_sim_mat(query_feat, target_feat, video_ids, is_video_query=False):
    unique_video_ids = list(set(video_ids))
    num_queries = len(unique_video_ids)
    num_targets = len(unique_video_ids)
    sim_mat = torch.zeros((num_queries, num_targets))

    for i, query_video_id in enumerate(unique_video_ids):
        query_mask = torch.tensor([vid == query_video_id for vid in video_ids])
        query_features = query_feat[query_mask]

        for j, target_video_id in enumerate(unique_video_ids):
            target_mask = torch.tensor([vid == target_video_id for vid in video_ids])
            target_features = target_feat[target_mask]

            frame_similarities = torch.matmul(query_features, target_features.t())
            sim_mat[i, j] = frame_similarities.max()

    return sim_mat.numpy()

def eval_retrieval(model, data, audio_conf, label_csv, direction, num_class, model_type='pretrain', batch_size=48):
    print(model)
    print(data)
    frame_use = 10  # Use all frames (0 to 9)
    val_audio_conf = audio_conf
    val_audio_conf['frame_use'] = frame_use
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = nn.BCELoss()
    
    if model_type == 'sync_pretrain':
        val_dataset = dataloader_sync.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            collate_fn=dataloader_sync.eval_collate_fn
        )
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11)
    else:
        val_dataset = dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True
        )
        if model_type == 'pretrain':
            audio_model = models.CAVMAE(modality_specific_depth=11)
        elif model_type == 'finetune':
            audio_model = models.CAVMAEFT(label_dim=num_class, modality_specific_depth=11)

    sdA = torch.load(model, map_location=device)
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    audio_model.eval()
    
    if model_type == 'sync_pretrain':
        r1, r5, r10, mr = get_sync_retrieval_result(audio_model, val_loader, direction)
    else:
        r1, r5, r10, mr = get_retrieval_result(audio_model, val_loader, direction)
    
    return r1, r5, r10, mr

if __name__ == "__main__":
    model = 'cav-mae-scale++.pth'
    data = 'datafilles/vggsound/cluster_nodes/vgg_test_5_per_class_for_retrieval_cleaned.json'
    label_csv = 'datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv'
    dataset = 'vggsound'
    model_type = 'pretrain'

    target_length = 1024 if model_type != 'sync_pretrain' else 96

    res = []

    for direction in ['video', 'audio']:
        audio_conf = {
            'num_mel_bins': 128, 
            'target_length': target_length, 
            'freqm': 0, 
            'timem': 0, 
            'mixup': 0, 
            'dataset': dataset,
            'mode': 'eval', 
            'mean': -5.081, 
            'std': 4.4849, 
            'noise': False, 
            'im_res': 224, 
            'frame_use': 10
        }
        
        if dataset == "audioset":
            num_class = 527
        elif dataset == "vggsound":
            num_class = 309
        else:
            print(f"Unsupported dataset: {dataset}")
            exit(1)

        r1, r5, r10, mr = eval_retrieval(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=num_class, direction=direction, model_type=model_type, batch_size=100)
        res.append([dataset, direction, r1, r5, r10, mr])

    np.savetxt('./retrieval_result.csv', res, delimiter=',', fmt='%s')