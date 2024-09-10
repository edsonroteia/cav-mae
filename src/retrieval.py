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

# get mean
def get_agg_sim_mat(a, b):
    import pdb ; pdb.set_trace()
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

# direction: 'audio' means audio->visual retrieval, 'video' means visual->audio retrieval
def get_retrieval_result(audio_model, val_loader, direction='audio', model_type='pretrain'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        # Add tqdm progress bar
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            if model_type == 'sync_pretrain':
                a_input, v_input, labels, video_id, frame_indices = batch
            else:
                (a_input, v_input, labels) = batch
            if i == 0:
                print("A_shape", a_input.shape)
                print("V_shape", v_input.shape)
            if model_type == 'sync_pretrain':
                # flatten batch so we process all frames at the same time
                a_input = a_input.reshape(a_input.shape[0] * a_input.shape[1], a_input.shape[2], a_input.shape[3])
                v_input = v_input.reshape(v_input.shape[0] * v_input.shape[1], v_input.shape[2], v_input.shape[3], v_input.shape[4])

            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                # mean pool all patches
                audio_output = torch.mean(audio_output, dim=1)
                video_output = torch.mean(video_output, dim=1)
                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            
            if model_type == 'sync_pretrain':
                # Group features from the same video together
                num_frames = audio_output.shape[0] // len(video_id)
                audio_output = audio_output.view(len(video_id), num_frames, -1)
                A_a_feat.append(audio_output)
            
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)
    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    if direction == 'audio':
        # audio->visual retrieval
        sim_mat = get_agg_sim_mat(A_a_feat, A_v_feat) if model_type == 'sync_pretrain' else get_sim_mat(A_a_feat, A_v_feat)
    elif direction == 'video':
        # visual->audio retrieval
        sim_mat = get_agg_sim_mat(A_v_feat, A_a_feat) if model_type == 'sync_pretrain' else get_sim_mat(A_v_feat, A_a_feat)
    result = compute_metrics(sim_mat)
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def eval_retrieval(model, data, audio_conf, label_csv, direction, num_class, model_type='pretrain', batch_size=48):
    print(model)
    print(data)
    frame_use = 5
    # eval setting
    val_audio_conf = audio_conf
    val_audio_conf['frame_use'] = frame_use
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = torch.nn.BCELoss()
    if model_type == 'sync_pretrain':
        val_loader = torch.utils.data.DataLoader(dataloader_sync.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True, collate_fn=train_collate_fn)
    else:
        val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    # cav-mae only been ssl pretrained
    if model_type == 'sync_pretrain':
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11)
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
    r1, r5, r10, mr = get_retrieval_result(audio_model, val_loader, direction, model_type)
    return r1, r5, r10, mr

if __name__ == "__main__":
    # Hardcoded values
    model = '/local/1306531/models/best_audio_model.pth'
    #model = 'cav-mae-scale++.pth'
    data = 'datafilles/vggsound/cluster_nodes/vgg_test_5_per_class_for_retrieval_cleaned.json'
    label_csv = 'datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv'
    dataset = 'vggsound'
    model_type='sync_pretrain'
    #model_type = 'pretrain'

    if model_type == 'sync_pretrain':
        target_length = 96
    else:
        target_length = 1024

    res = []

    if dataset == "audioset":
        # for audioset
        for direction in ['video', 'audio']:
            audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                        'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}
            r1, r5, r10, mr = eval_retrieval(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=527, direction=direction, model_type=model_type, batch_size=100)
            res.append([dataset, direction, r1, r5, r10, mr])

    elif dataset == "vggsound":
        # for vggsound
        for direction in ['video', 'audio']:
            audio_conf = {'num_mel_bins': 128, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                        'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}
            r1, r5, r10, mr = eval_retrieval(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=309, direction=direction, model_type=model_type, batch_size=100)
            res.append([dataset, direction, r1, r5, r10, mr])
    else:
        print(f"Unsupported dataset: {dataset}")
        exit(1)

    np.savetxt('./retrieval_result.csv', res, delimiter=',', fmt='%s')