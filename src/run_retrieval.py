#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run retrieval evaluation on VGGSound for CAV-MAE and CAV-JEPA models.

Usage:
    python src/run_retrieval.py --model_type cavmae --model_path path/to/model.pth
    python src/run_retrieval.py --model_type cavjepa --model_path path/to/model.pth
"""

import argparse
import csv
import os
import sys
import torch
import numpy as np
from torch.cuda.amp import autocast

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


def filter_valid_samples(audio_feat, video_feat, debug=False):
    """Filter out samples with zero-norm vectors (missing data)."""
    # Convert to float32 to avoid precision issues with float16 from autocast
    audio_feat = audio_feat.float()
    video_feat = video_feat.float()

    audio_norms = np.linalg.norm(audio_feat.numpy(), axis=1)
    video_norms = np.linalg.norm(video_feat.numpy(), axis=1)

    if debug:
        print(f"  DEBUG: Audio norms - min: {audio_norms.min():.6f}, max: {audio_norms.max():.6f}, mean: {audio_norms.mean():.6f}")
        print(f"  DEBUG: Video norms - min: {video_norms.min():.6f}, max: {video_norms.max():.6f}, mean: {video_norms.mean():.6f}")
        print(f"  DEBUG: Audio norms > 0.5: {(audio_norms > 0.5).sum()}, Video norms > 0.5: {(video_norms > 0.5).sum()}")

    # Keep samples where both audio and video have non-zero norm
    # Use a threshold that accounts for L2-normalized vectors (should be ~1.0)
    valid_mask = (audio_norms > 0.5) & (video_norms > 0.5)

    num_total = int(valid_mask.shape[0])
    num_valid = int(valid_mask.sum())
    num_invalid = num_total - num_valid
    num_audio_invalid = int((audio_norms <= 0.5).sum())
    num_video_invalid = int((video_norms <= 0.5).sum())

    quality_stats = {
        "total_samples": num_total,
        "valid_samples": num_valid,
        "invalid_samples": num_invalid,
        "invalid_rate": (num_invalid / num_total) if num_total > 0 else 0.0,
        "audio_invalid_samples": num_audio_invalid,
        "video_invalid_samples": num_video_invalid,
    }

    print(
        "  Data quality: "
        f"valid={num_valid}/{num_total} "
        f"({(100.0 * quality_stats['invalid_rate']):.2f}% invalid), "
        f"audio_invalid={num_audio_invalid}, video_invalid={num_video_invalid}"
    )

    if debug:
        if num_invalid > 0:
            print(f"  Filtered out {num_invalid} samples with missing data")
        else:
            print(f"  All {len(audio_feat)} samples valid")

    return audio_feat[valid_mask], video_feat[valid_mask], quality_stats


def compute_metrics(x):
    """Compute retrieval metrics from similarity matrix."""
    if x.shape[0] == 0:
        raise ValueError("No valid samples left after filtering; cannot compute retrieval metrics.")

    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    if len(ind) == 0:
        raise ValueError("Rank index array is empty; check input features and filtering.")

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


def extract_features(audio_model, val_loader, debug=False):
    """Extract pooled and normalized audio/video features for retrieval."""
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
            if debug and i == 0:
                print(f"  DEBUG: Raw audio output shape: {audio_output.shape}, mean: {audio_output.mean():.6f}, std: {audio_output.std():.6f}")
                print(f"  DEBUG: Raw video output shape: {video_output.shape}, mean: {video_output.mean():.6f}, std: {video_output.std():.6f}")
            # Mean pool all patches
            audio_output = torch.mean(audio_output, dim=1)
            video_output = torch.mean(video_output, dim=1)
            if debug and i == 0:
                print(f"  DEBUG: After pooling - audio mean: {audio_output.mean():.6f}, video mean: {video_output.mean():.6f}")
                print(f"  DEBUG: Audio norms before L2: {torch.norm(audio_output, dim=1)[:4]}")
            # L2 normalization
            audio_output = torch.nn.functional.normalize(audio_output, dim=-1)
            video_output = torch.nn.functional.normalize(video_output, dim=-1)
            if debug and i == 0:
                print(f"  DEBUG: After L2 norm - audio norms: {torch.norm(audio_output, dim=1)[:4]}")
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            A_a_feat.append(audio_output)
            A_v_feat.append(video_output)

    if len(A_a_feat) == 0 or len(A_v_feat) == 0:
        raise ValueError("No features extracted from dataloader.")

    return torch.cat(A_a_feat), torch.cat(A_v_feat)


def evaluate_direction(audio_feat, video_feat, direction='audio'):
    """Evaluate retrieval metrics in one direction from precomputed features."""
    print(f'  Computing similarity matrix ({audio_feat.shape[0]} valid samples)...')
    if direction == 'audio':
        # audio -> visual retrieval
        sim_mat = get_sim_mat(audio_feat.numpy(), video_feat.numpy())
    elif direction == 'video':
        # visual -> audio retrieval
        sim_mat = get_sim_mat(video_feat.numpy(), audio_feat.numpy())
    else:
        raise ValueError(f"Unknown direction: {direction}")

    result = compute_metrics(sim_mat)
    r1, r5, r10, mr = print_computed_metrics(result, direction)
    return {'R1': r1, 'R5': r5, 'R10': r10, 'MR': mr}


def eval_retrieval(model_path, model_type, data_json, label_csv, batch_size=48, debug=False):
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

    # Extract features once, then evaluate both directions.
    print("\nExtracting features...")
    audio_feat, video_feat = extract_features(audio_model, val_loader, debug=debug)

    # Filter out samples with missing data (shared for both directions)
    audio_feat, video_feat, quality_stats = filter_valid_samples(audio_feat, video_feat, debug=debug)

    # Run retrieval for both directions
    results = {}
    print("\n--- Audio -> Visual Retrieval ---")
    results['a2v'] = evaluate_direction(audio_feat, video_feat, direction='audio')

    print("\n--- Visual -> Audio Retrieval ---")
    results['v2a'] = evaluate_direction(audio_feat, video_feat, direction='video')

    return results, quality_stats


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
    parser.add_argument('--stats_output', type=str, default=None,
                        help='Optional output CSV file for data quality stats')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')

    args = parser.parse_args()

    # Run evaluation
    results, quality_stats = eval_retrieval(
        model_path=args.model_path,
        model_type=args.model_type,
        data_json=args.data_json,
        label_csv=args.label_csv,
        batch_size=args.batch_size,
        debug=args.debug
    )

    # Print summary
    print("\n" + "="*60)
    print("RETRIEVAL RESULTS SUMMARY")
    print("="*60)
    print(f"Audio->Visual: R@1={results['a2v']['R1']:.4f}, R@5={results['a2v']['R5']:.4f}, R@10={results['a2v']['R10']:.4f}, MR={results['a2v']['MR']}")
    print(f"Visual->Audio: R@1={results['v2a']['R1']:.4f}, R@5={results['v2a']['R5']:.4f}, R@10={results['v2a']['R10']:.4f}, MR={results['v2a']['MR']}")
    print(
        "Quality: "
        f"valid={quality_stats['valid_samples']}/{quality_stats['total_samples']}, "
        f"invalid_rate={100.0 * quality_stats['invalid_rate']:.2f}%"
    )
    print("="*60)

    # Save to CSV if requested
    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write("direction,R1,R5,R10,MR\n")
            f.write(f"a2v,{results['a2v']['R1']},{results['a2v']['R5']},{results['a2v']['R10']},{results['a2v']['MR']}\n")
            f.write(f"v2a,{results['v2a']['R1']},{results['v2a']['R5']},{results['v2a']['R10']},{results['v2a']['MR']}\n")
        print(f"\nResults saved to: {args.output}")

    if args.stats_output:
        stats_dir = os.path.dirname(args.stats_output)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(args.stats_output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key in [
                'total_samples',
                'valid_samples',
                'invalid_samples',
                'invalid_rate',
                'audio_invalid_samples',
                'video_invalid_samples',
            ]:
                writer.writerow([key, quality_stats[key]])
        print(f"Stats saved to: {args.stats_output}")


if __name__ == '__main__':
    main()
