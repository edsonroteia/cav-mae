# -*- coding: utf-8 -*-
# @Time    : 3/12/23 10:23 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : retrieval.py

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
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def extract_features(audio_model, val_loader, model_type='pretrain', cls_token=False, local_matching=False):
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    video_ids = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            if 'sync' or 'enhanced' in model_type:
                a_input, v_input, labels, video_id, frame_indices = batch
                video_ids.extend(video_id)
            else:
                (a_input, v_input, labels) = batch
                video_ids.extend([f"video_{i}_{j}" for j in range(len(labels))])
            
            if i == 0:
                print("A_shape", a_input.shape)
                print("V_shape", v_input.shape)
                
            if 'sync' or 'enhanced' in model_type:
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
                    if 'enhanced' in model_type:
                        audio_output = audio_output[::16]
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
    
    return A_a_feat, A_v_feat, video_ids

def load_and_extract_features(model_path, data_path, audio_conf, label_csv, model_type='pretrain', batch_size=48, num_register_tokens=4, cls_token=False):
    # Load the model
    if 'ch' in model_type:
        model_type = model_type.replace('_ch', '')
        contrastive_heads = True
    else:
        contrastive_heads = False

    if model_type == 'sync_pretrain_registers':
        audio_model = models.CAVMAESync(audio_length=audio_conf['target_length'], modality_specific_depth=11, 
                                      num_register_tokens=num_register_tokens, total_frame=audio_conf['total_frame'],
                                      contrastive_heads=contrastive_heads)
    elif model_type == 'sync_pretrain_registers_cls':
        audio_model = models.CAVMAESync(audio_length=audio_conf['target_length'], modality_specific_depth=11, 
                                      num_register_tokens=num_register_tokens, cls_token=True, 
                                      total_frame=audio_conf['total_frame'],
                                      contrastive_heads=contrastive_heads)
    elif model_type == 'sync_pretrain':
        audio_model = models.CAVMAESync(audio_length=audio_conf['target_length'], modality_specific_depth=11, 
                                      num_register_tokens=0, total_frame=audio_conf['total_frame'])
    elif model_type == 'pretrain' or model_type == 'pretrain_enhanced':
        audio_model = models.CAVMAE(modality_specific_depth=11)
    
    # Load model weights
    sdA = torch.load(model_path, map_location=device)
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(
        dataloader_sync.AudiosetDataset(data_path, label_csv=label_csv, audio_conf=audio_conf),
        batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True,
        collate_fn=train_collate_fn
    )
    
    # Extract features
    audio_features, video_features, video_ids = extract_features(audio_model, val_loader, model_type, cls_token)
    
    return audio_features, video_features, video_ids

def visualize_tsne(audio_features, video_features, video_ids, output_dir):
    """
    Create TSNE visualizations for audio and video features
    Args:
        audio_features: numpy array of audio features
        video_features: numpy array of video features
        video_ids: list of video IDs for coloring
        output_dir: directory to save visualizations
    """
    # Convert features to 2D array if they're 3D (in case of sync models)
    if len(audio_features.shape) == 3:
        audio_features = audio_features.reshape(audio_features.shape[0], -1)
        video_features = video_features.reshape(video_features.shape[0], -1)
    
    # Create TSNE models
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    # Fit and transform the features
    audio_tsne = tsne.fit_transform(audio_features)
    video_tsne = tsne.fit_transform(video_features)
    
    # Create unique colors for each video ID
    unique_ids = list(set(video_ids))
    color_palette = sns.color_palette('husl', n_colors=len(unique_ids))
    id_to_color = dict(zip(unique_ids, color_palette))
    colors = [id_to_color[vid] for vid in video_ids]
    
    # Plot audio features
    plt.figure(figsize=(10, 10))
    plt.scatter(audio_tsne[:, 0], audio_tsne[:, 1], c=colors, alpha=0.6)
    plt.title('t-SNE Visualization of Audio Features')
    plt.savefig(os.path.join(output_dir, 'audio_tsne.png'))
    plt.close()
    
    # Plot video features
    plt.figure(figsize=(10, 10))
    plt.scatter(video_tsne[:, 0], video_tsne[:, 1], c=colors, alpha=0.6)
    plt.title('t-SNE Visualization of Video Features')
    plt.savefig(os.path.join(output_dir, 'video_tsne.png'))
    plt.close()
    
    # Plot both features side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.scatter(audio_tsne[:, 0], audio_tsne[:, 1], c=colors, alpha=0.6)
    ax1.set_title('Audio Features')
    
    ax2.scatter(video_tsne[:, 0], video_tsne[:, 1], c=colors, alpha=0.6)
    ax2.set_title('Video Features')
    
    plt.suptitle('t-SNE Visualization of Audio-Video Features')
    plt.savefig(os.path.join(output_dir, 'combined_tsne.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract audio and video features from a model')
    
    parser.add_argument('--dataset', type=str, choices=['audioset', 'vggsound'], 
                        help='Dataset to use for feature extraction')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Type of model architecture')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for processing')
    parser.add_argument('--output_dir', type=str, default='./features',
                        help='Directory to save extracted features')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up dataset-specific parameters
    if args.dataset == "audioset":
        data_path = 'datafilles/audioset_20k/cluster_nodes/audioset_eval_5_per_class_for_retrieval_cleaned.json'
        label_csv = 'datafilles/audioset_20k/cluster_nodes/class_labels_indices.csv'
    elif args.dataset == "vggsound":
        data_path = 'datafilles/vggsound/cluster_nodes/vgg_test_5_per_class_for_retrieval_cleaned.json'
        label_csv = 'datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv'
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Set up audio configuration
    if 'sync' in args.model_type:
        if '2s' in args.model_type:
            target_length = 192
        elif '3s' in args.model_type:
            target_length = 304
        elif '5s' in args.model_type:
            target_length = 512
        elif '7s' in args.model_type:
            target_length = 720
        elif '10s' in args.model_type:
            target_length = 1024
        else:
            target_length = 96
    else:
        target_length = 1024
    print("Using target_length:", target_length)

    # Clean up model type string
    model_type = args.model_type
    model_type = model_type.replace('_2s', '').replace('_3s', '').replace('_5s', '').replace('_7s', '').replace('_10s', '')
    cls_token = 'cls' in model_type

    # Set up audio configuration
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': target_length,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': args.dataset,
        'mode': 'retrieval',
        'mean': -5.081,
        'std': 4.4849,
        'noise': False,
        'im_res': 224,
        'frame_use': 5,
        'num_samples': args.num_samples,
        'total_frame': 16
    }

    # Extract features
    print(f"Extracting features from {args.model_path}")
    audio_features, video_features, video_ids = load_and_extract_features(
        args.model_path,
        data_path,
        audio_conf,
        label_csv,
        model_type=model_type,
        batch_size=args.batch_size,
        num_register_tokens=8 if '2918' in args.model_path else 4,
        cls_token=cls_token
    )

    # Save features
    output_base = os.path.join(args.output_dir, os.path.basename(args.model_path).replace('.pth', ''))
    np.save(f"{output_base}_audio_features.npy", audio_features.numpy())
    np.save(f"{output_base}_video_features.npy", video_features.numpy())
    print(f"Features saved to {output_base}_audio_features.npy and {output_base}_video_features.npy")

    # Create visualizations
    print("Creating t-SNE visualizations...")
    visualize_tsne(
        audio_features.numpy(), 
        video_features.numpy(), 
        video_ids, 
        args.output_dir
    )
    print(f"Visualizations saved to {args.output_dir}")


'''
python visualize_vectors.py \
    --dataset audioset \
    --model_path /scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20241027_025558/models/audio_model.25.pth \
    --model_type sync_pretrain_registers_cls_3s_ch \
    --num_samples 50 \
    --batch_size 50 \
    --output_dir ./features
'''