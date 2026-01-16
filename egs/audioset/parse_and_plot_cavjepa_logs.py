#!/usr/bin/env python3
"""Parse CAV-JEPA training logs and plot loss curves."""

import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_cavjepa_log_file(log_path):
    """Parse a CAV-JEPA training log file and extract metrics."""
    metrics = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            # Match training step lines
            # Format: Epoch: [epoch][step/total] ... Train Total Loss X.XXXX Train JEPA Loss Audio X.XXXX ...
            match = re.search(
                r'Epoch: \[(\d+)\]\[(\d+)/(\d+)\].*'
                r'Train Total Loss ([\d.]+).*'
                r'Train JEPA Loss Audio ([\d.]+).*'
                r'Train JEPA Loss Visual ([\d.]+).*'
                r'Train Contrastive Loss ([\d.]+).*'
                r'Train Contrastive Acc ([\d.]+).*'
                r'Momentum ([\d.]+)',
                line
            )
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                total_steps = int(match.group(3))
                metrics['epoch'].append(epoch)
                metrics['step'].append(step)
                metrics['total_steps'].append(total_steps)
                metrics['total_loss'].append(float(match.group(4)))
                metrics['jepa_audio'].append(float(match.group(5)))
                metrics['jepa_visual'].append(float(match.group(6)))
                metrics['contrastive_loss'].append(float(match.group(7)))
                metrics['contrastive_acc'].append(float(match.group(8)))
                metrics['momentum'].append(float(match.group(9)))

            # Match validation lines if present
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
        for key in ['total_loss', 'jepa_audio', 'jepa_visual', 'contrastive_loss', 'contrastive_acc', 'momentum']:
            values = [v for v, m in zip(metrics[key], mask) if m]
            if values:
                epoch_metrics[key].append(np.mean(values))
        epoch_metrics['epoch'].append(ep)

    return epoch_metrics


def plot_cavjepa_training_curves(metrics, output_path):
    """Plot CAV-JEPA training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('CAV-JEPA Training Progress (Job 312964)', fontsize=14, fontweight='bold')

    epoch_avg = compute_epoch_averages(metrics)
    epochs = epoch_avg['epoch']

    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, epoch_avg['total_loss'], 'b-o', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss (JEPA + 0.01×Contrastive)')
    ax1.grid(True, alpha=0.3)

    # Plot 2: JEPA Losses (Audio vs Visual)
    ax2 = axes[0, 1]
    ax2.plot(epochs, epoch_avg['jepa_audio'], 'r-o', label='JEPA Audio', linewidth=2, markersize=5)
    ax2.plot(epochs, epoch_avg['jepa_visual'], 'g-s', label='JEPA Visual', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('JEPA Loss')
    ax2.set_title('JEPA Losses by Modality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Contrastive Loss
    ax3 = axes[0, 2]
    ax3.plot(epochs, epoch_avg['contrastive_loss'], 'm-o', linewidth=2, markersize=5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Contrastive Loss')
    ax3.set_title('Contrastive Loss')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Contrastive Accuracy
    ax4 = axes[1, 0]
    acc_pct = [a * 100 for a in epoch_avg['contrastive_acc']]
    ax4.plot(epochs, acc_pct, 'c-o', linewidth=2, markersize=5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Contrastive Accuracy')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])

    # Plot 5: Momentum Schedule
    ax5 = axes[1, 1]
    ax5.plot(epochs, epoch_avg['momentum'], 'k-o', linewidth=2, markersize=5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Momentum')
    ax5.set_title('EMA Momentum Schedule')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Step-level dynamics (first 2000 steps)
    ax6 = axes[1, 2]
    n_steps = min(2000, len(metrics['total_loss']))
    ax6.plot(range(n_steps), metrics['total_loss'][:n_steps], 'b-', alpha=0.7, label='Total Loss')
    ax6.plot(range(n_steps), metrics['contrastive_loss'][:n_steps], 'm-', alpha=0.5, label='Contrastive')
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('Loss')
    ax6.set_title('First 2000 Steps: Loss Dynamics')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def print_training_summary(metrics):
    """Print training summary statistics."""
    epoch_avg = compute_epoch_averages(metrics)

    print("\n" + "="*70)
    print("CAV-JEPA TRAINING SUMMARY (Job 312964)")
    print("="*70)

    if len(epoch_avg['epoch']) == 0:
        print("No data found!")
        return

    print(f"\nProgress: {len(epoch_avg['epoch'])} epochs completed")
    print(f"Total training steps logged: {len(metrics['epoch'])}")

    # First vs Latest epoch comparison
    print("\n" + "-"*70)
    print("EPOCH COMPARISON:")
    print("-"*70)
    print(f"{'Metric':<25} {'Epoch 1':>12} {'Latest':>12} {'Change':>12}")
    print("-"*70)

    for key, label in [
        ('total_loss', 'Total Loss'),
        ('jepa_audio', 'JEPA Audio Loss'),
        ('jepa_visual', 'JEPA Visual Loss'),
        ('contrastive_loss', 'Contrastive Loss'),
        ('contrastive_acc', 'Contrastive Acc'),
        ('momentum', 'Momentum')
    ]:
        first = epoch_avg[key][0]
        last = epoch_avg[key][-1]
        if key == 'contrastive_acc':
            change = f"{(last - first) * 100:+.1f}%"
            print(f"{label:<25} {first*100:>11.1f}% {last*100:>11.1f}% {change:>12}")
        else:
            pct_change = ((last - first) / first * 100) if first != 0 else 0
            print(f"{label:<25} {first:>12.4f} {last:>12.4f} {pct_change:>+11.1f}%")

    print("-"*70)

    # Per-epoch details
    print("\nPER-EPOCH AVERAGES:")
    print("-"*70)
    print(f"{'Epoch':>5} {'Total':>10} {'JEPA-A':>10} {'JEPA-V':>10} {'Contr':>10} {'Acc%':>8} {'Mom':>10}")
    print("-"*70)

    for i, ep in enumerate(epoch_avg['epoch']):
        print(f"{ep:>5} {epoch_avg['total_loss'][i]:>10.4f} "
              f"{epoch_avg['jepa_audio'][i]:>10.4f} {epoch_avg['jepa_visual'][i]:>10.4f} "
              f"{epoch_avg['contrastive_loss'][i]:>10.4f} {epoch_avg['contrastive_acc'][i]*100:>7.1f}% "
              f"{epoch_avg['momentum'][i]:>10.6f}")

    print("="*70)


def main():
    base_dir = '/weka/kuehne/kqr867/code/cav-mae/egs/audioset'

    # CAV-JEPA log file
    cavjepa_log = os.path.join(base_dir, 'log/312964_cavjepa_pretrain.txt')

    if not os.path.exists(cavjepa_log):
        print(f"Log file not found: {cavjepa_log}")
        return

    print(f"Parsing CAV-JEPA log: {cavjepa_log}")
    metrics = parse_cavjepa_log_file(cavjepa_log)
    print(f"  Found {len(metrics['epoch'])} training steps")

    if len(metrics['epoch']) == 0:
        print("No training data found in log file!")
        return

    # Print summary
    print_training_summary(metrics)

    # Plot
    output_path = os.path.join(base_dir, 'training_curves_cavjepa.png')
    plot_cavjepa_training_curves(metrics, output_path)


if __name__ == '__main__':
    main()
