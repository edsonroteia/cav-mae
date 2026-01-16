#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run retrieval evaluation on VGGSound for CAV-MAE and CAV-JEPA models.

Usage:
    python src/run_retrieval.py --model_type cavmae --model_path path/to/model.pth
    python src/run_retrieval.py --model_type cavjepa --model_path path/to/model.pth
"""

import argparse
import os
import sys
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch import nn

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models
import dataloader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Return 0 for invalid vectors (missing data)
    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return cos_sim


def get_sim_mat(a, b):
    """Compute similarity matrix between two sets of vectors."""
    B = a.shape[0]
    sim_mat = np.empty([B, B])
    for i in range(B):
        for j in range(B):
            sim_mat[i, j] = get_similarity(a[i, :], b[j, :])
    return sim_mat


def filter_valid_samples(audio_feat, video_feat):
    """Filter out samples with zero-norm vectors (missing data)."""
    # Convert to float32 to avoid precision issues with float16 from autocast
    audio_feat = audio_feat.float()
    video_feat = video_feat.float()

    audio_norms = np.linalg.norm(audio_feat.numpy(), axis=1)
    video_norms = np.linalg.norm(video_feat.numpy(), axis=1)

    # Debug: print norm statistics
    print(f"  DEBUG: Audio norms - min: {audio_norms.min():.6f}, max: {audio_norms.max():.6f}, mean: {audio_norms.mean():.6f}")
    print(f"  DEBUG: Video norms - min: {video_norms.min():.6f}, max: {video_norms.max():.6f}, mean: {video_norms.mean():.6f}")
    print(f"  DEBUG: Audio norms > 0.5: {(audio_norms > 0.5).sum()}, Video norms > 0.5: {(video_norms > 0.5).sum()}")

    # Keep samples where both audio and video have non-zero norm
    # Use a threshold that accounts for L2-normalized vectors (should be ~1.0)
    valid_mask = (audio_norms > 0.5) & (video_norms > 0.5)

    num_invalid = (~valid_mask).sum()
    if num_invalid > 0:
        print(f"  Filtered out {num_invalid} samples with missing data")
    else:
        print(f"  All {len(audio_feat)} samples valid")

    return audio_feat[valid_mask], video_feat[valid_mask]


def compute_metrics(x):
    """Compute retrieval metrics from similarity matrix."""
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


def print_computed_metrics(metrics, direction):
    """Print retrieval metrics."""
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print(f'{direction}: R@1: {r1:.4f} - R@5: {r5:.4f} - R@10: {r10:.4f} - Median R: {mr}')
    return r1, r5, r10, mr


def get_retrieval_result(audio_model, val_loader, direction='audio'):
    """Run retrieval evaluation."""
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels) in enumerate(val_loader):
            if i % 10 == 0:
                print(f'  Processing batch {i}/{len(val_loader)}...')
            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                audio_output, video_output = audio_model.forward_feat(audio_input, video_input)
            # Convert to float32 before pooling/normalization to avoid precision issues
            audio_output = audio_output.float()
            video_output = video_output.float()
            # Debug first batch
            if i == 0:
                print(f"  DEBUG: Raw audio output shape: {audio_output.shape}, mean: {audio_output.mean():.6f}, std: {audio_output.std():.6f}")
                print(f"  DEBUG: Raw video output shape: {video_output.shape}, mean: {video_output.mean():.6f}, std: {video_output.std():.6f}")
            # Mean pool all patches
            audio_output = torch.mean(audio_output, dim=1)
            video_output = torch.mean(video_output, dim=1)
            if i == 0:
                print(f"  DEBUG: After pooling - audio mean: {audio_output.mean():.6f}, video mean: {video_output.mean():.6f}")
                print(f"  DEBUG: Audio norms before L2: {torch.norm(audio_output, dim=1)[:4]}")
            # L2 normalization
            audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
            video_output = torch.nn.functional.normalize(video_output, dim=-1)
            if i == 0:
                print(f"  DEBUG: After L2 norm - audio norms: {torch.norm(audio_output, dim=1)[:4]}")
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)

    # Filter out samples with missing data
    A_a_feat, A_v_feat = filter_valid_samples(A_a_feat, A_v_feat)

    print(f'  Computing similarity matrix ({A_a_feat.shape[0]} valid samples)...')
    if direction == 'audio':
        # audio -> visual retrieval
        sim_mat = get_sim_mat(A_a_feat.numpy(), A_v_feat.numpy())
    elif direction == 'video':
        # visual -> audio retrieval
        sim_mat = get_sim_mat(A_v_feat.numpy(), A_a_feat.numpy())

    result = compute_metrics(sim_mat)
    return print_computed_metrics(result, direction)


def eval_retrieval(model_path, model_type, data_json, label_csv, batch_size=48):
    """Run retrieval evaluation for a model."""
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Model Type: {model_type}")
    print(f"Data: {data_json}")
    print(f"{'='*60}")

    # Audio configuration for VGGSound
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': 1024,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': 'vggsound',
        'mode': 'eval',
        'mean': -5.081,
        'std': 4.4849,
        'noise': False,
        'im_res': 224,
        'frame_use': 5  # Use middle frame for eval
    }

    # Create dataloader
    print("\nLoading dataset...")
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(data_json, label_csv=label_csv, audio_conf=audio_conf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Load model
    print(f"\nLoading {model_type} model...")
    if model_type == 'cavmae':
        audio_model = models.CAVMAE(modality_specific_depth=11)
    elif model_type == 'cavjepa':
        audio_model = models.CAVJEPA(modality_specific_depth=11)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights (load to CPU first to avoid CUDA init issues)
    sdA = torch.load(model_path, map_location='cpu')
    # Handle DataParallel checkpoint keys (remove 'module.' prefix if present)
    if any(k.startswith('module.') for k in sdA.keys()):
        sdA = {k.replace('module.', ''): v for k, v in sdA.items()}
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(f"Load message: {msg}")
    audio_model.eval()

    # Run retrieval for both directions
    results = {}
    print("\n--- Audio -> Visual Retrieval ---")
    r1_a, r5_a, r10_a, mr_a = get_retrieval_result(audio_model, val_loader, direction='audio')
    results['a2v'] = {'R1': r1_a, 'R5': r5_a, 'R10': r10_a, 'MR': mr_a}

    print("\n--- Visual -> Audio Retrieval ---")
    r1_v, r5_v, r10_v, mr_v = get_retrieval_result(audio_model, val_loader, direction='video')
    results['v2a'] = {'R1': r1_v, 'R5': r5_v, 'R10': r10_v, 'MR': mr_v}

    return results


def main():
    parser = argparse.ArgumentParser(description='VGGSound Retrieval Evaluation')
    parser.add_argument('--model_type', type=str, required=True, choices=['cavmae', 'cavjepa'],
                        help='Model type: cavmae or cavjepa')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_json', type=str,
                        default='/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json',
                        help='Path to VGGSound retrieval JSON')
    parser.add_argument('--label_csv', type=str,
                        default='/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv',
                        help='Path to VGGSound label CSV')
    parser.add_argument('--batch_size', type=int, default=48,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results')

    args = parser.parse_args()

    # Run evaluation
    results = eval_retrieval(
        model_path=args.model_path,
        model_type=args.model_type,
        data_json=args.data_json,
        label_csv=args.label_csv,
        batch_size=args.batch_size
    )

    # Print summary
    print("\n" + "="*60)
    print("RETRIEVAL RESULTS SUMMARY")
    print("="*60)
    print(f"Audio->Visual: R@1={results['a2v']['R1']:.4f}, R@5={results['a2v']['R5']:.4f}, R@10={results['a2v']['R10']:.4f}, MR={results['a2v']['MR']}")
    print(f"Visual->Audio: R@1={results['v2a']['R1']:.4f}, R@5={results['v2a']['R5']:.4f}, R@10={results['v2a']['R10']:.4f}, MR={results['v2a']['MR']}")
    print("="*60)

    # Save to CSV if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("direction,R1,R5,R10,MR\n")
            f.write(f"a2v,{results['a2v']['R1']},{results['a2v']['R5']},{results['a2v']['R10']},{results['a2v']['MR']}\n")
            f.write(f"v2a,{results['v2a']['R1']},{results['v2a']['R5']},{results['v2a']['R10']},{results['v2a']['MR']}\n")
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
