#!/usr/bin/env python3
"""Parse CAV-MAE training logs and plot loss curves."""

import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log_file(log_path):
    """Parse a CAV-MAE training log file and extract metrics."""
    metrics = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            # Match training step lines
            # Format: Epoch: [epoch][step/total] ... Train Total Loss X.XXXX Train MAE Loss Audio X.XXXX ...
            match = re.search(
                r'Epoch: \[(\d+)\]\[(\d+)/\d+\].*'
                r'Train Total Loss ([\d.]+).*'
                r'Train MAE Loss Audio ([\d.]+).*'
                r'Train MAE Loss Visual ([\d.]+).*'
                r'Train Contrastive Loss ([\d.]+).*'
                r'Train Contrastive Acc ([\d.]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                metrics['epoch'].append(epoch)
                metrics['step'].append(step)
                metrics['total_loss'].append(float(match.group(3)))
                metrics['mae_audio'].append(float(match.group(4)))
                metrics['mae_visual'].append(float(match.group(5)))
                metrics['contrastive_loss'].append(float(match.group(6)))
                metrics['contrastive_acc'].append(float(match.group(7)))

            # Match validation lines
            val_match = re.search(r'Eval Total Loss: ([\d.]+)', line)
            if val_match:
                metrics['val_total_loss'].append(float(val_match.group(1)))

            val_cont_match = re.search(r'Eval Contrastive Accuracy: ([\d.]+)', line)
            if val_cont_match:
                metrics['val_contrastive_acc'].append(float(val_cont_match.group(1)))

    return metrics

def compute_epoch_averages(metrics):
    """Compute per-epoch average metrics."""
    epochs = sorted(set(metrics['epoch']))
    epoch_metrics = defaultdict(list)

    for ep in epochs:
        mask = [e == ep for e in metrics['epoch']]
        for key in ['total_loss', 'mae_audio', 'mae_visual', 'contrastive_loss', 'contrastive_acc']:
            values = [v for v, m in zip(metrics[key], mask) if m]
            epoch_metrics[key].append(np.mean(values))
        epoch_metrics['epoch'].append(ep)

    return epoch_metrics

def plot_training_curves(contrastive_metrics, mae_metrics, output_path):
    """Plot training curves comparing contrastive-only and MAE-only."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CAV-MAE Ablation: Contrastive-only vs MAE-only Training', fontsize=14, fontweight='bold')

    cont_epoch = compute_epoch_averages(contrastive_metrics)
    mae_epoch = compute_epoch_averages(mae_metrics)

    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(cont_epoch['epoch'], cont_epoch['total_loss'], 'b-o', label='Contrastive-only', linewidth=2, markersize=4)
    ax1.plot(mae_epoch['epoch'], mae_epoch['total_loss'], 'r-s', label='MAE-only', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Contrastive Loss (only for contrastive-only model)
    ax2 = axes[0, 1]
    ax2.plot(cont_epoch['epoch'], cont_epoch['contrastive_loss'], 'b-o', label='Contrastive Loss', linewidth=2, markersize=4)
    ax2.plot(cont_epoch['epoch'], [acc * 10 for acc in cont_epoch['contrastive_acc']], 'g-^', label='Contrastive Acc (×10)', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss / Accuracy×10')
    ax2.set_title('Contrastive-only: Loss & Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: MAE Losses (only for MAE-only model)
    ax3 = axes[1, 0]
    ax3.plot(mae_epoch['epoch'], mae_epoch['mae_audio'], 'r-o', label='Audio MAE', linewidth=2, markersize=4)
    ax3.plot(mae_epoch['epoch'], mae_epoch['mae_visual'], 'm-s', label='Visual MAE', linewidth=2, markersize=4)
    ax3.plot(mae_epoch['epoch'], mae_epoch['total_loss'], 'k-^', label='Total MAE', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAE Loss')
    ax3.set_title('MAE-only: Reconstruction Losses')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Training dynamics (step-level for first epoch)
    ax4 = axes[1, 1]
    # Get first 1000 steps of each
    cont_steps = min(1000, len(contrastive_metrics['total_loss']))
    mae_steps = min(1000, len(mae_metrics['total_loss']))
    ax4.plot(range(cont_steps), contrastive_metrics['total_loss'][:cont_steps], 'b-', alpha=0.7, label='Contrastive-only')
    ax4.plot(range(mae_steps), mae_metrics['total_loss'][:mae_steps], 'r-', alpha=0.7, label='MAE-only')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Total Loss')
    ax4.set_title('First 1000 Steps: Loss Dynamics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    base_dir = '/weka/kuehne/kqr867/code/cav-mae/egs/audioset'

    # Log files
    contrastive_log = os.path.join(base_dir, 'log/312887_contrastive_only.txt')
    mae_log = os.path.join(base_dir, 'log/312886_mae_only.txt')

    print("Parsing contrastive-only log...")
    cont_metrics = parse_log_file(contrastive_log)
    print(f"  Found {len(cont_metrics['epoch'])} training steps")

    print("Parsing MAE-only log...")
    mae_metrics = parse_log_file(mae_log)
    print(f"  Found {len(mae_metrics['epoch'])} training steps")

    # Plot
    output_path = os.path.join(base_dir, 'training_curves_ablation.png')
    plot_training_curves(cont_metrics, mae_metrics, output_path)

    # Print summary
    cont_epoch = compute_epoch_averages(cont_metrics)
    mae_epoch = compute_epoch_averages(mae_metrics)

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print("\nContrastive-only (Final Epoch 25):")
    print(f"  Contrastive Loss: {cont_epoch['contrastive_loss'][-1]:.4f}")
    print(f"  Contrastive Acc:  {cont_epoch['contrastive_acc'][-1]*100:.2f}%")

    print("\nMAE-only (Final Epoch 25):")
    print(f"  Audio MAE:  {mae_epoch['mae_audio'][-1]:.4f}")
    print(f"  Visual MAE: {mae_epoch['mae_visual'][-1]:.4f}")
    print(f"  Total MAE:  {mae_epoch['total_loss'][-1]:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
