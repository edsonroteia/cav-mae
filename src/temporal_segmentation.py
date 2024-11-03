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
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import numpy as np
import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def extract_features(audio_model, val_loader, model_type='pretrain', cls_token=False, local_matching=False):
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    video_ids = []
    all_labels = []  # Renamed to avoid confusion
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing batches"):
            if 'sync' or 'enhanced' in model_type:
                a_input, v_input, batch_labels, video_id, frame_indices = batch
                video_ids.extend(video_id)
                all_labels.extend(batch_labels)  # Store batch labels
            else:
                (a_input, v_input, batch_labels) = batch
                video_ids.extend([f"video_{i}_{j}" for j in range(len(batch_labels))])
                all_labels.extend(batch_labels)  # Store batch labels
            
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
    
    return A_a_feat, A_v_feat, video_ids, all_labels

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
    audio_features, video_features, video_ids, labels = extract_features(audio_model, val_loader, model_type, cls_token)
    
    # Randomly select a video index
    random_idx = np.random.randint(0, len(video_ids))
    print(f"Randomly selected video index: {random_idx}")
    
    return audio_features, video_features, video_ids, labels, random_idx

def extract_single_video_features(audio_model, video_data, model_type='pretrain', cls_token=False):
    """Extract features from a single video"""
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    with torch.no_grad():
        a_input, v_input, batch_labels, video_id, frame_indices = video_data
        
        # Add batch dimension if not present
        if len(a_input.shape) == 2:
            a_input = a_input.unsqueeze(0)
        if len(v_input.shape) == 4:
            v_input = v_input.unsqueeze(0)
            
        # Reshape video input to merge batch and frame dimensions
        v_input = v_input.view(-1, v_input.shape[-3], v_input.shape[-2], v_input.shape[-1])  # [B*F, C, H, W]
            
        print("Audio input shape:", a_input.shape)
        print("Video input shape:", v_input.shape)
        
        # Move inputs to device
        audio_input, video_input = a_input.to(device), v_input.to(device)
        
        with autocast():
            if cls_token:
                tokens_audio_output, tokens_video_output, cls_audio_output, cls_video_output = audio_model.module.forward_feat(audio_input, video_input)
                audio_output = cls_audio_output
                video_output = cls_video_output
            else:
                audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                audio_output = torch.mean(audio_output, dim=1)
                video_output = torch.mean(video_output, dim=1)
            
            # Normalize features
            audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
            video_output = torch.nn.functional.normalize(video_output, dim=-1)
        
        # Combine audio and video features
        combined_features = torch.cat([audio_output, video_output], dim=-1)
        return combined_features.cpu()

def perform_temporal_segmentation(features, max_segments=5, distance_threshold=None):
    """
    Perform temporal segmentation using hierarchical clustering with adaptive number of segments
    based on cosine distances between features.
    
    Args:
        features: Normalized feature vectors
        max_segments: Maximum number of segments allowed
        distance_threshold: Optional threshold for cosine distance. If None, will be determined automatically.
    """
    # Normalize features
    features = F.normalize(features, dim=1)
    features_np = features.detach().cpu().numpy()
    
    # Compute pairwise cosine distances
    distances = pairwise_distances(features_np, metric='cosine')
    
    if distance_threshold is None:
        # Automatically determine threshold based on distance distribution
        # Using percentile to avoid outliers
        distance_threshold = np.percentile(distances[np.triu_indices_from(distances, k=1)], 75)
        print(f"Automatically determined distance threshold: {distance_threshold:.3f}")
    
    # Try different thresholds if the initial one gives too many or too few segments
    min_segments = 2  # We want at least 2 segments
    current_threshold = distance_threshold
    
    while True:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=current_threshold,
            metric='cosine',
            linkage='complete'
        )
        
        labels = clustering.fit_predict(features_np)
        n_segments = len(np.unique(labels))
        
        if n_segments <= max_segments and n_segments >= min_segments:
            print(f"Found {n_segments} segments with threshold {current_threshold:.3f}")
            break
        elif n_segments > max_segments:
            # If too many segments, increase threshold
            current_threshold *= 1.2
        else:
            # If too few segments, decrease threshold
            current_threshold *= 0.8
            
        print(f"Adjusting threshold to {current_threshold:.3f} (current segments: {n_segments})")
        
        # Prevent infinite loops
        if current_threshold > 1.0 or current_threshold < 0.1:
            print("Warning: Could not find optimal number of segments. Using K-means fallback.")
            # Fallback to K-means with fixed number of clusters
            kmeans = KMeans(n_clusters=max(min(max_segments, len(features_np)//2), 2))
            labels = kmeans.fit_predict(features_np)
            n_segments = len(np.unique(labels))
            break
    
    print(f"Final segmentation: {n_segments} segments")
    return labels

def visualize_segments(video_frames, audio_frames, segment_labels, output_path, video_label=None, class_labels=None):
    """
    Visualize video frames, audio spectrograms, and temporal segments together
    """
    unique_labels = np.unique(segment_labels)
    n_segments = len(unique_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_segments))
    n_frames = len(video_frames)
    
    # Create figure with 3 rows: video frames, audio spectrograms, and segment timeline
    fig = plt.figure(figsize=(2*n_frames, 10))
    gs = plt.GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0.3)

    # Create subfigures for frames and spectrograms
    gs_frames = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[0])
    gs_audio = gridspec.GridSpecFromSubplotSpec(1, n_frames, subplot_spec=gs[1])
    
    # Plot video frames
    frame_axes = []
    for i in range(n_frames):
        ax = fig.add_subplot(gs_frames[i])
        frame = video_frames[i].numpy().transpose(1, 2, 0).copy()
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f'Frame {i}')
        
        # Add colored border based on segment
        color = colors[segment_labels[i]]
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        frame_axes.append(ax)

    # Plot audio spectrograms
    audio_axes = []
    for i in range(n_frames):
        ax = fig.add_subplot(gs_audio[i])
        # Convert tensor to numpy if needed
        if torch.is_tensor(audio_frames[i]):
            spec = audio_frames[i].detach().cpu().numpy()
        else:
            spec = audio_frames[i].copy()
        
        if len(spec.shape) == 3:
            spec = spec[0]
        
        # spec = spec + 1e-8
        # spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        
        im = ax.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        ax.axis('off')
        
        # Add colored border based on segment
        color = colors[segment_labels[i]]
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        audio_axes.append(ax)

    # Add colorbar for spectrograms
    # cax = fig.add_axes([0.92, 0.4, 0.02, 0.2])
    # plt.colorbar(im, cax=cax, label='Magnitude (dB)')

    # Plot segment timeline
    ax_timeline = fig.add_subplot(gs[2])
    
    # Create segment blocks
    current_segment = segment_labels[0]
    segment_starts = [0]
    segment_lengths = [1]
    
    for i in range(1, len(segment_labels)):
        if segment_labels[i] == current_segment:
            segment_lengths[-1] += 1
        else:
            current_segment = segment_labels[i]
            segment_starts.append(i)
            segment_lengths.append(1)
    
    # Plot segments as colored blocks
    for i, label in enumerate(unique_labels):
        mask = segment_labels == label
        if np.any(mask):
            start = np.where(mask)[0][0]
            end = np.where(mask)[0][-1] + 1
            plt.axvspan(start, end, ymin=0, ymax=1, color=colors[i], alpha=0.3, 
                       label=f'Segment {label + 1}')
            # Or alternatively using Rectangle patch:
            # rect = plt.Rectangle((start, 0), end-start, 1, 
            #                     color=colors[i], alpha=0.3, 
            #                     label=f'Segment {label + 1}')
            # plt.gca().add_patch(rect)
    
    # Customize timeline appearance
    ax_timeline.set_xlim(0, n_frames)
    ax_timeline.set_ylim(-0.5, 0.5)
    ax_timeline.set_xlabel('Frame Index')
    ax_timeline.set_yticks([])
    
    # Add segment labels inside the blocks
    for start, length, segment in zip(segment_starts, segment_lengths, unique_labels):
        if length > 1:  # Only add text if there's enough space
            ax_timeline.text(start + length/2, 0, f'Segment\n{segment+1}',
                           ha='center', va='center')
    
    # Add title
    if video_label:
        plt.suptitle(f'Temporal Segmentation\nLabels: {class_labels}', y=0.95)
    else:
        plt.suptitle('Temporal Segmentation', y=0.95)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def load_class_labels(label_csv):
    """Load class labels from CSV file"""
    df = pd.read_csv(label_csv)
    # Create a mapping from index to display name
    return dict(zip(df['index'].astype(int), df['display_name']))

def get_active_class_names(label_vector, class_map):
    """Convert binary label vector to list of class names"""
    # Get indices where label is 1
    active_indices = torch.where(label_vector == 1)[0].cpu().numpy()
    # Convert indices to class names
    return [class_map[idx] for idx in active_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform temporal segmentation on a single video')
    
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
    parser.add_argument('--n_segments', type=int, default=5,
                        help='Number of temporal segments to create')
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

    # Check if features already exist
    output_base = os.path.join(args.output_dir, os.path.basename(args.model_path).replace('.pth', ''))
    audio_features_path = f"{output_base}_audio_features.npy"
    video_features_path = f"{output_base}_video_features.npy"

    if os.path.exists(audio_features_path) and os.path.exists(video_features_path):
        print("Loading pre-computed features...")
        audio_features = torch.from_numpy(np.load(audio_features_path))
        video_features = torch.from_numpy(np.load(video_features_path))
        video_ids = [f"video_{i}" for i in range(len(audio_features))]
        
        # Generate 100 unique random indices
        random_indices = np.random.choice(len(video_ids), size=100, replace=False)
        print(f"Randomly selected video indices: {random_indices}")

        # Load the model here
        if 'ch' in model_type:
            model_type = model_type.replace('_ch', '')
            contrastive_heads = True
        else:
            contrastive_heads = False

        if model_type == 'sync_pretrain_registers':
            audio_model = models.CAVMAESync(audio_length=audio_conf['target_length'], modality_specific_depth=11, 
                                          num_register_tokens=8 if '2918' in args.model_path else 4,
                                          total_frame=audio_conf['total_frame'],
                                          contrastive_heads=contrastive_heads)
        elif model_type == 'sync_pretrain_registers_cls':
            audio_model = models.CAVMAESync(audio_length=audio_conf['target_length'], modality_specific_depth=11, 
                                          num_register_tokens=8 if '2918' in args.model_path else 4,
                                          cls_token=True, 
                                          total_frame=audio_conf['total_frame'],
                                          contrastive_heads=contrastive_heads)
        elif model_type == 'sync_pretrain':
            audio_model = models.CAVMAESync(audio_length=audio_conf['target_length'], modality_specific_depth=11, 
                                          num_register_tokens=0, total_frame=audio_conf['total_frame'])
        elif model_type == 'pretrain' or model_type == 'pretrain_enhanced':
            audio_model = models.CAVMAE(modality_specific_depth=11)
            
        # Load model weights
        sdA = torch.load(args.model_path, map_location=device)
        if not isinstance(audio_model, torch.nn.DataParallel):
            audio_model = torch.nn.DataParallel(audio_model)
        msg = audio_model.load_state_dict(sdA, strict=False)
        print(msg)
    else:
        # Extract features
        print(f"Extracting features from {args.model_path}")
        audio_features, video_features, video_ids, labels, random_idx = load_and_extract_features(
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
        np.save(audio_features_path, audio_features.numpy())
        np.save(video_features_path, video_features.numpy())
        print(f"Features saved to {audio_features_path} and {video_features_path}")

    # Process multiple videos
    dataset = dataloader_sync.AudiosetDataset(data_path, label_csv=label_csv, audio_conf=audio_conf)
    
    # Load the class mapping
    class_map = load_class_labels(label_csv)

    for idx, random_idx in enumerate(random_indices):
        print(f"\nProcessing video {idx+1}/20 (index: {random_idx})")
        video_data = dataset[random_idx]
        
        # Get class names for this video
        # video_data[2] should be the label vector
        class_names = get_active_class_names(video_data[2], class_map)
        
        # Extract features and perform segmentation as before
        combined_features = extract_single_video_features(
            audio_model,
            video_data,
            model_type=model_type,
            cls_token=cls_token
        )

        segments = perform_temporal_segmentation(combined_features, args.n_segments)

        # Visualize with class names
        output_base = os.path.join(args.output_dir, f"video_{random_idx}")
        visualize_segments(
            video_data[1], 
            video_data[0], 
            segments, 
            f"{output_base}_segments.png", 
            video_label=f"Video {random_idx}",
            class_labels=class_names
        )

        print(f"Visualization saved for video {random_idx}")
        print(f"Classes: {', '.join(class_names)}")


'''
python src/temporal_segmentation.py \
    --dataset audioset \
    --model_path /scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20241027_025558/models/audio_model.25.pth \
    --model_type sync_pretrain_registers_cls_3s_ch \
    --num_samples 50 \
    --batch_size 50 \
    --output_dir ./features \
    --n_segments 5
'''