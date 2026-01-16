# -*- coding: utf-8 -*-
# Model merging script for CAV-MAE
# Supports: Simple averaging, Weighted averaging, Task arithmetic
#
# Usage examples:
#   python merge_models.py --method simple --model_mae mae.pth --model_contrastive contrastive.pth --output merged.pth
#   python merge_models.py --method weighted --model_mae mae.pth --model_contrastive contrastive.pth --alpha 0.7 --output merged.pth
#   python merge_models.py --method task_arithmetic --model_mae mae.pth --model_contrastive contrastive.pth --model_base base.pth --lambda_scale 1.0 --output merged.pth

import argparse
import torch
import os
from collections import OrderedDict


def load_model(path, device='cpu'):
    """Load model state dict from path."""
    print(f"Loading model from: {path}")
    return torch.load(path, map_location=device)


# Layers that should be taken from contrastive model only (not merged with MAE)
# These layers don't receive gradients during MAE-only training
CONTRASTIVE_ONLY_LAYERS = [
    'norm_a.weight', 'norm_a.bias',
    'norm_v.weight', 'norm_v.bias',
    'module.norm_a.weight', 'module.norm_a.bias',
    'module.norm_v.weight', 'module.norm_v.bias',
]


def is_contrastive_only_layer(key):
    """Check if this layer should be taken from contrastive model only."""
    return any(key.endswith(suffix) for suffix in ['norm_a.weight', 'norm_a.bias', 'norm_v.weight', 'norm_v.bias'])


def simple_average(model_a, model_b, use_contrastive_for_norms=False):
    """
    Simple averaging: (model_mae + model_contrastive) / 2

    Args:
        model_a: MAE model weights (state_dict)
        model_b: Contrastive model weights (state_dict)
        use_contrastive_for_norms: If True, take norm_a/norm_v from contrastive only

    Returns:
        Merged state_dict
    """
    merged = OrderedDict()

    # Get all keys from both models
    all_keys = set(model_a.keys()) | set(model_b.keys())

    norm_layers_overridden = []
    for key in all_keys:
        if key in model_a and key in model_b:
            # Check if this is a norm layer that should come from contrastive only
            if use_contrastive_for_norms and is_contrastive_only_layer(key):
                merged[key] = model_b[key]  # Use contrastive weights
                norm_layers_overridden.append(key)
            else:
                # Both models have this parameter - average them
                merged[key] = (model_a[key] + model_b[key]) / 2.0
        elif key in model_a:
            # Only in model_a
            merged[key] = model_a[key]
        else:
            # Only in model_b
            merged[key] = model_b[key]

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


def weighted_average(model_a, model_b, alpha=0.5, use_contrastive_for_norms=False):
    """
    Weighted averaging: alpha * model_mae + (1-alpha) * model_contrastive

    Args:
        model_a: MAE model weights (state_dict)
        model_b: Contrastive model weights (state_dict)
        alpha: Weight for MAE model (default 0.5)
               alpha=1.0 means 100% MAE, alpha=0.0 means 100% Contrastive
        use_contrastive_for_norms: If True, take norm_a/norm_v from contrastive only

    Returns:
        Merged state_dict
    """
    merged = OrderedDict()

    all_keys = set(model_a.keys()) | set(model_b.keys())

    norm_layers_overridden = []
    for key in all_keys:
        if key in model_a and key in model_b:
            if use_contrastive_for_norms and is_contrastive_only_layer(key):
                merged[key] = model_b[key]  # Use contrastive weights
                norm_layers_overridden.append(key)
            else:
                merged[key] = alpha * model_a[key] + (1 - alpha) * model_b[key]
        elif key in model_a:
            merged[key] = model_a[key]
        else:
            merged[key] = model_b[key]

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


def task_arithmetic(base_model, task_mae, task_contrastive, lambda_scale=1.0, use_contrastive_for_norms=False):
    """
    Task arithmetic: base + lambda * (tau_mae + tau_contrastive)
    where tau = trained - base (task vector)

    Reference: "Editing Models with Task Arithmetic" (Ilharco et al., 2023)

    Args:
        base_model: Base/initial model weights (the checkpoint used to initialize training)
        task_mae: MAE-trained model weights
        task_contrastive: Contrastive-trained model weights
        lambda_scale: Scaling factor for task vectors (default 1.0)
                      Higher values = stronger effect of both objectives
        use_contrastive_for_norms: If True, take norm_a/norm_v from contrastive only

    Returns:
        Merged state_dict
    """
    merged = OrderedDict()

    norm_layers_overridden = []
    for key in base_model.keys():
        # Get weights from each model, defaulting to base if not present
        base_weight = base_model[key]
        mae_weight = task_mae.get(key, base_weight)
        contrastive_weight = task_contrastive.get(key, base_weight)

        # For norm layers, use contrastive only (MAE doesn't train these)
        if use_contrastive_for_norms and is_contrastive_only_layer(key):
            merged[key] = contrastive_weight
            norm_layers_overridden.append(key)
        else:
            # Compute task vectors (delta from base)
            tau_mae = mae_weight - base_weight
            tau_contrastive = contrastive_weight - base_weight

            # Apply task arithmetic: base + lambda * (sum of task vectors)
            merged[key] = base_weight + lambda_scale * (tau_mae + tau_contrastive)

    # Include any keys only in task models but not in base
    for model, name in [(task_mae, 'MAE'), (task_contrastive, 'Contrastive')]:
        for key in model.keys():
            if key not in merged:
                print(f"Warning: Key '{key}' found in {name} model but not in base. Using {name} weights directly.")
                merged[key] = model[key]

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


def print_model_stats(state_dict, name="Model"):
    """Print statistics about a model state dict."""
    num_params = len(state_dict)
    total_elements = sum(p.numel() for p in state_dict.values())
    print(f"{name}: {num_params} parameters, {total_elements:,} total elements")


def verify_merge(model_a, model_b, merged, method, alpha=None):
    """Verify that the merge was performed correctly by spot-checking a parameter."""
    # Find a shared parameter to verify
    shared_keys = set(model_a.keys()) & set(model_b.keys())
    if not shared_keys:
        print("Warning: No shared keys to verify merge")
        return

    test_key = list(shared_keys)[0]

    if method == 'simple':
        expected = (model_a[test_key] + model_b[test_key]) / 2.0
    elif method == 'weighted' and alpha is not None:
        expected = alpha * model_a[test_key] + (1 - alpha) * model_b[test_key]
    else:
        return  # Skip verification for task_arithmetic (needs base model)

    if torch.allclose(merged[test_key], expected, atol=1e-6):
        print(f"Verification passed for key: {test_key}")
    else:
        print(f"Warning: Verification failed for key: {test_key}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge CAV-MAE models trained with different objectives',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--method', type=str, required=True,
                        choices=['simple', 'weighted', 'task_arithmetic'],
                        help='Merging method to use')

    parser.add_argument('--model_mae', type=str, required=True,
                        help='Path to MAE-only trained model')

    parser.add_argument('--model_contrastive', type=str, required=True,
                        help='Path to Contrastive-only trained model')

    parser.add_argument('--model_base', type=str, default=None,
                        help='Path to base model (required for task_arithmetic)')

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for MAE model in weighted averaging (0.0-1.0)')

    parser.add_argument('--lambda_scale', type=float, default=1.0,
                        help='Scaling factor for task vectors in task arithmetic')

    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged model')

    parser.add_argument('--use_contrastive_for_norms', action='store_true',
                        help='Take norm_a/norm_v weights from contrastive model only (fixes MAE norm collapse issue)')

    args = parser.parse_args()

    # Validate arguments
    if args.method == 'task_arithmetic' and args.model_base is None:
        raise ValueError("--model_base is required for task_arithmetic method")

    if args.method == 'weighted' and not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0.0 and 1.0")

    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)

    model_mae = load_model(args.model_mae)
    print_model_stats(model_mae, "MAE model")

    model_contrastive = load_model(args.model_contrastive)
    print_model_stats(model_contrastive, "Contrastive model")

    # Perform merging
    print("\n" + "="*60)
    print(f"Merging with method: {args.method}")
    print("="*60)

    if args.use_contrastive_for_norms:
        print("NOTE: Using contrastive-only weights for norm_a/norm_v (fixes MAE collapse)")

    if args.method == 'simple':
        print("Formula: merged = (model_mae + model_contrastive) / 2")
        merged = simple_average(model_mae, model_contrastive, args.use_contrastive_for_norms)
        verify_merge(model_mae, model_contrastive, merged, 'simple')

    elif args.method == 'weighted':
        print(f"Formula: merged = {args.alpha} * model_mae + {1-args.alpha} * model_contrastive")
        merged = weighted_average(model_mae, model_contrastive, args.alpha, args.use_contrastive_for_norms)
        verify_merge(model_mae, model_contrastive, merged, 'weighted', args.alpha)

    elif args.method == 'task_arithmetic':
        model_base = load_model(args.model_base)
        print_model_stats(model_base, "Base model")
        print(f"Formula: merged = base + {args.lambda_scale} * (tau_mae + tau_contrastive)")
        print("         where tau = trained_model - base_model")
        merged = task_arithmetic(model_base, model_mae, model_contrastive, args.lambda_scale, args.use_contrastive_for_norms)

    # Save merged model
    print("\n" + "="*60)
    print("Saving merged model...")
    print("="*60)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    torch.save(merged, args.output)
    print(f"Merged model saved to: {args.output}")

    # Print final statistics
    print_model_stats(merged, "Merged model")

    print("\n" + "="*60)
    print("Merge complete!")
    print("="*60)


if __name__ == '__main__':
    main()
