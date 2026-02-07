#!/usr/bin/env python3
"""Generate Pareto frontier plots for CAV-MAE model merging experiments.

Reads retrieval R@1 from CSV files and classification mAP from SFT result files,
then generates combined and per-scale Pareto frontier plots.
"""

import csv
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

BASE_DIR = Path(__file__).resolve().parent
EXP_DIR = BASE_DIR / "exp"
RETRIEVAL_DIR = EXP_DIR / "retrieval_results"


def load_retrieval_results():
    """Load retrieval R@1 from all_results_summary.csv and fine-grained CSVs."""
    retrieval = {}

    # Main summary CSV: model, sweep, direction, R@1, R@5, R@10, MR
    summary_path = RETRIEVAL_DIR / "all_results_summary.csv"
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            sweep = row["sweep"]
            direction = row["direction"]
            r1 = float(row["R@1"])

            key = (model, sweep)
            if key not in retrieval:
                retrieval[key] = {}
            retrieval[key][direction] = r1

    # Fine-grained CSVs (plusplus only)
    for alpha in ["0.05", "0.15"]:
        fg_path = RETRIEVAL_DIR / f"plusplus_merged_weighted_{alpha}.csv"
        if fg_path.exists():
            with open(fg_path) as f:
                reader = csv.DictReader(f)
                key = (f"merged_weighted_{alpha}", "finegrained-plusplus")
                retrieval[key] = {}
                for row in reader:
                    direction = row["direction"]
                    r1 = float(row["R1"])
                    retrieval[key][direction] = r1

    return retrieval


def load_sft_results():
    """Load best mAP from each SFT experiment's result.csv.

    result.csv format: acc, mAP, mAUC, lr (no header, space-separated scientific notation)
    """
    sft = {}
    for result_path in sorted(glob.glob(str(EXP_DIR / "sft-*/result.csv"))):
        dirname = os.path.basename(os.path.dirname(result_path))
        # Extract model name: sft-MODELNAME-5e-5-bs36-epoch15-TIMESTAMP
        # Remove prefix "sft-" and suffix "-5e-5-bs36-epoch15-YYYYMMDD_HHMMSS"
        parts = dirname.split("-5e-5-bs36-epoch15-")
        if len(parts) != 2:
            continue
        model_name = parts[0]  # e.g., "sft-cavonly" or "sft-plusplus-merged_weighted_0.05"

        best_map = 0.0
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = line.split(",")
                if len(vals) >= 2:
                    map_val = float(vals[1])
                    if map_val > best_map:
                        best_map = map_val

        if best_map > 0:
            sft[model_name] = best_map * 100  # Convert to percentage

    return sft


def compute_avg_r1(retrieval_data):
    """Compute average R@1 across a2v and v2a directions."""
    a2v = retrieval_data.get("a2v", 0)
    v2a = retrieval_data.get("v2a", 0)
    return (a2v + v2a) / 2 * 100  # Convert to percentage


def build_model_data(retrieval, sft):
    """Map retrieval and SFT results to create (R@1, mAP, name, scale) tuples."""
    models = []

    # Define mappings: (retrieval_model, retrieval_sweep) -> (sft_key, display_name, scale)
    mapping = [
        # Base scale
        (("contrastive_only_lr1e-4", "base"), "sft-cavonly", "Contrastive Only", "Base"),
        (("merged_weighted_alpha0.0", "weighted"), "sft-base-merged_weighted_alpha0.0", "Weighted α=0.0 (=CAV)", "Base"),
        (("merged_weighted_alpha0.1", "weighted"), "sft-cavmerged-alpha0.1", "Weighted α=0.1", "Base"),
        (("merged_fisher_direct", "fisher"), "sft-base-fisher_direct", "Fisher Direct", "Base"),
        (("merged_fisher_taskvec", "fisher"), "sft-fisher-taskvec", "Fisher TaskVec", "Base"),

        # Original scale (paper models)
        (("original_scalep", "base"), "sft-scalep-original", "CAV-MAE (paper)", "Original"),
        (("original_scalepp", "base"), "sft-scalepp-original", "CAV-MAE++ (paper)", "Original"),

        # Scale++ (plusplus)
        (("contrastive++", "plusplus"), "sft-plusplus-contrastive++", "Contrastive++", "Scale++"),
        (("merged_weighted_0.0", "merged-plusplus"), "sft-merged_weighted_0.0", "Merged α=0.0", "Scale++"),
        (("merged_weighted_0.1", "merged-plusplus"), "sft-merged_weighted_0.1", "Merged α=0.1", "Scale++"),
        (("merged_weighted_0.2", "merged-plusplus"), "sft-plusplus-merged_weighted_0.2", "Merged α=0.2", "Scale++"),
        (("merged_fisher", "merged-plusplus"), "sft-merged_fisher", "Fisher Merge", "Scale++"),
        # Fine-grained
        (("merged_weighted_0.05", "finegrained-plusplus"), "sft-plusplus-merged_weighted_0.05", "Merged α=0.05", "Scale++"),
        (("merged_weighted_0.15", "finegrained-plusplus"), "sft-plusplus-merged_weighted_0.15", "Merged α=0.15", "Scale++"),
    ]

    # Handle Fisher TaskVec - pick the second (correct) SFT run
    fisher_taskvec_candidates = [k for k in sft if k.startswith("sft-fisher-taskvec")]
    if len(fisher_taskvec_candidates) >= 2:
        # Sort by timestamp, pick the later one
        fisher_taskvec_candidates.sort()
        fisher_taskvec_key = fisher_taskvec_candidates[-1]
    elif fisher_taskvec_candidates:
        fisher_taskvec_key = fisher_taskvec_candidates[0]
    else:
        fisher_taskvec_key = None

    for (ret_model, ret_sweep), sft_key, display_name, scale in mapping:
        ret_key = (ret_model, ret_sweep)
        if ret_key not in retrieval:
            print(f"  WARNING: No retrieval data for {ret_model} (sweep={ret_sweep})")
            continue

        # Special handling for Fisher TaskVec
        actual_sft_key = sft_key
        if "fisher-taskvec" in sft_key and fisher_taskvec_key:
            actual_sft_key = fisher_taskvec_key

        if actual_sft_key not in sft:
            print(f"  WARNING: No SFT data for {actual_sft_key} (display: {display_name})")
            continue

        r1 = compute_avg_r1(retrieval[ret_key])
        map_val = sft[actual_sft_key]
        models.append((r1, map_val, display_name, scale))

    return models


def find_pareto_optimal(models):
    """Find Pareto-optimal models (maximize both R@1 and mAP)."""
    pareto = []
    for i, (r1_i, map_i, name_i, scale_i) in enumerate(models):
        dominated = False
        for j, (r1_j, map_j, name_j, scale_j) in enumerate(models):
            if i == j:
                continue
            if r1_j >= r1_i and map_j >= map_i and (r1_j > r1_i or map_j > map_i):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto


SCALE_COLORS = {
    "Base": "#2196F3",       # Blue
    "Original": "#FF9800",   # Orange
    "Scale++": "#4CAF50",    # Green
}

SCALE_MARKERS = {
    "Base": "o",
    "Original": "s",
    "Scale++": "D",
}


def plot_combined(models, pareto_indices, output_prefix):
    """Single plot with all models, color-coded by scale."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot all points by scale
    for scale in ["Base", "Original", "Scale++"]:
        xs = [m[0] for m in models if m[3] == scale]
        ys = [m[1] for m in models if m[3] == scale]
        names = [m[2] for m in models if m[3] == scale]
        ax.scatter(xs, ys, c=SCALE_COLORS[scale], marker=SCALE_MARKERS[scale],
                   s=80, label=scale, zorder=5, edgecolors='white', linewidths=0.5)

    # Draw Pareto frontier line
    pareto_points = sorted([(models[i][0], models[i][1]) for i in pareto_indices], key=lambda p: p[0])
    if pareto_points:
        px, py = zip(*pareto_points)
        ax.plot(px, py, 'k--', alpha=0.4, linewidth=1.5, zorder=3, label='Pareto frontier')

    # Highlight Pareto-optimal points
    for i in pareto_indices:
        r1, map_val, name, scale = models[i]
        ax.scatter([r1], [map_val], c=SCALE_COLORS[scale], marker=SCALE_MARKERS[scale],
                   s=160, zorder=6, edgecolors='red', linewidths=2)

    # Label all points
    for r1, map_val, name, scale in models:
        is_pareto = any(models[i][2] == name for i in pareto_indices)
        fontweight = 'bold' if is_pareto else 'normal'
        # Offset labels to avoid overlap
        offset_x, offset_y = 0.3, 0.15
        if name == "CAV-MAE++ (paper)":
            offset_x, offset_y = -0.3, -0.5
        elif name == "CAV-MAE (paper)":
            offset_x, offset_y = 0.3, -0.4
        elif name == "Merged α=0.0":
            offset_x, offset_y = 0.3, -0.4
        elif name == "Contrastive++":
            offset_x, offset_y = 0.3, 0.2
        elif name == "Merged α=0.05":
            offset_x, offset_y = 0.3, 0.2
        elif name == "Fisher Direct":
            offset_x, offset_y = 0.3, -0.4
        elif name == "Fisher TaskVec":
            offset_x, offset_y = -0.3, 0.3
        elif "α=0.1" in name and scale == "Scale++":
            offset_x, offset_y = 0.3, 0.3
        elif "α=0.15" in name:
            offset_x, offset_y = 0.3, -0.4

        ax.annotate(name, (r1, map_val),
                    xytext=(offset_x, offset_y), textcoords='offset fontsize',
                    fontsize=8, fontweight=fontweight, alpha=0.85,
                    arrowprops=dict(arrowstyle='-', alpha=0.3, lw=0.5) if abs(offset_x) > 0.5 else None)

    ax.set_xlabel("Retrieval R@1 (%)", fontsize=12)
    ax.set_ylabel("Classification mAP (%)", fontsize=12)
    ax.set_title("CAV-MAE Model Merging: Retrieval vs Classification Trade-off", fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=9)

    # Add annotation for the gap
    ax.annotate('', xy=(17.8, 49.9), xytext=(17.8, 44.8),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5, alpha=0.5))
    ax.text(18.3, 47.3, '~5% mAP\ngap', fontsize=9, color='red', alpha=0.7, ha='left')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{output_prefix}.{ext}", dpi=200, bbox_inches='tight')
        print(f"  Saved {output_prefix}.{ext}")
    plt.close(fig)


def plot_by_scale(models, pareto_indices, output_prefix):
    """Three subplots, one per scale."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    scales = ["Base", "Original", "Scale++"]

    # Global min/max for consistent axes
    all_r1 = [m[0] for m in models]
    all_map = [m[1] for m in models]

    for ax, scale in zip(axes, scales):
        scale_models = [(r1, map_val, name, s) for r1, map_val, name, s in models if s == scale]
        scale_pareto = [i for i in pareto_indices if models[i][3] == scale]

        if not scale_models:
            ax.set_title(f"{scale}\n(no models)")
            continue

        # Plot points
        xs = [m[0] for m in scale_models]
        ys = [m[1] for m in scale_models]
        ax.scatter(xs, ys, c=SCALE_COLORS[scale], marker=SCALE_MARKERS[scale],
                   s=80, zorder=5, edgecolors='white', linewidths=0.5)

        # Pareto frontier
        pareto_pts = sorted([(models[i][0], models[i][1]) for i in scale_pareto], key=lambda p: p[0])
        if pareto_pts:
            px, py = zip(*pareto_pts)
            ax.plot(px, py, 'k--', alpha=0.4, linewidth=1.5, zorder=3)

        # Highlight Pareto-optimal
        for i in scale_pareto:
            r1, map_val, name, s = models[i]
            ax.scatter([r1], [map_val], c=SCALE_COLORS[scale], marker=SCALE_MARKERS[scale],
                       s=160, zorder=6, edgecolors='red', linewidths=2)

        # Label points
        for r1, map_val, name, s in scale_models:
            is_pareto = any(models[i][2] == name for i in scale_pareto)
            fontweight = 'bold' if is_pareto else 'normal'
            ax.annotate(name, (r1, map_val),
                        xytext=(0.3, 0.3), textcoords='offset fontsize',
                        fontsize=8, fontweight=fontweight, alpha=0.85)

        # Add paper baselines as reference (faint) on all subplots
        paper_models = [(r1, map_val, name) for r1, map_val, name, s in models if s == "Original"]
        if scale != "Original" and paper_models:
            for pr1, pmap, pname in paper_models:
                ax.scatter([pr1], [pmap], c='gray', marker='x', s=40, alpha=0.4, zorder=2)
                ax.annotate(pname, (pr1, pmap), fontsize=7, alpha=0.4,
                            xytext=(0.2, -0.5), textcoords='offset fontsize')

        ax.set_title(f"{scale}", fontsize=13, fontweight='bold', color=SCALE_COLORS[scale])
        ax.set_xlabel("Retrieval R@1 (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=9)

    axes[0].set_ylabel("Classification mAP (%)")

    fig.suptitle("Pareto Frontiers by Model Scale", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{output_prefix}.{ext}", dpi=200, bbox_inches='tight')
        print(f"  Saved {output_prefix}.{ext}")
    plt.close(fig)


def main():
    print("Loading retrieval results...")
    retrieval = load_retrieval_results()
    print(f"  Loaded {len(retrieval)} model/sweep combinations")

    print("\nLoading SFT results...")
    sft = load_sft_results()
    print(f"  Loaded {len(sft)} SFT experiments:")
    for k, v in sorted(sft.items()):
        print(f"    {k}: {v:.2f}% mAP")

    print("\nBuilding model data...")
    models = build_model_data(retrieval, sft)
    print(f"  Matched {len(models)} models with both retrieval and classification data:")
    for r1, map_val, name, scale in sorted(models, key=lambda m: -m[0]):
        print(f"    {name:30s} ({scale:8s}): R@1={r1:.2f}%, mAP={map_val:.2f}%")

    print("\nFinding Pareto-optimal models...")
    pareto = find_pareto_optimal(models)
    print(f"  {len(pareto)} Pareto-optimal models:")
    for i in pareto:
        r1, map_val, name, scale = models[i]
        print(f"    {name:30s} ({scale:8s}): R@1={r1:.2f}%, mAP={map_val:.2f}%")

    print("\nGenerating plots...")
    output_combined = str(EXP_DIR / "pareto_combined")
    output_by_scale = str(EXP_DIR / "pareto_by_scale")

    plot_combined(models, pareto, output_combined)
    plot_by_scale(models, pareto, output_by_scale)

    print("\nDone!")


if __name__ == "__main__":
    main()
