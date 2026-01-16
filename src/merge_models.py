# -*- coding: utf-8 -*-
# Model merging script for CAV-MAE
# Supports: Simple averaging, Weighted averaging, Task arithmetic,
#           Layer-wise alpha, DARE-TIES, Fisher-weighted merge,
#           Orthogonal conflict-aware merge
#
# Usage examples:
#   python merge_models.py --method simple --model_mae mae.pth --model_contrastive contrastive.pth --output merged.pth
#   python merge_models.py --method weighted --model_mae mae.pth --model_contrastive contrastive.pth --alpha 0.7 --output merged.pth
#   python merge_models.py --method task_arithmetic --model_mae mae.pth --model_contrastive contrastive.pth --model_base base.pth --lambda_scale 1.0 --output merged.pth
#   python merge_models.py --method layerwise --model_mae mae.pth --model_contrastive contrastive.pth --model_base base.pth --output merged.pth
#   python merge_models.py --method dare_ties --model_mae mae.pth --model_contrastive contrastive.pth --model_base base.pth --dare_keep_ratio 0.2 --output merged.pth
#   python merge_models.py --method fisher --model_mae mae.pth --model_contrastive contrastive.pth --fisher_mae fisher_mae.pth --fisher_contrastive fisher_con.pth --output merged.pth
#   python merge_models.py --method orthogonal --model_mae mae.pth --model_contrastive contrastive.pth --model_base base.pth --ortho_beta 1.0 --ortho_tau 0.0 --output merged.pth

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


def strip_module_prefix(key):
    if key.startswith("module."):
        return key[len("module."):]
    return key


def group_key(key, group_mode="block"):
    if group_mode == "param":
        return key
    key = strip_module_prefix(key)
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] in {"blocks_a", "blocks_v", "blocks_u", "decoder_blocks"}:
        return ".".join(parts[:2])
    return parts[0]


def compute_group_norms(base_model, task_mae, task_contrastive, group_mode="block"):
    norms = {}
    for key in base_model.keys():
        group = group_key(key, group_mode=group_mode)
        base_weight = base_model[key]
        mae_weight = task_mae.get(key, base_weight)
        contrastive_weight = task_contrastive.get(key, base_weight)
        tau_mae = mae_weight - base_weight
        tau_con = contrastive_weight - base_weight
        mae_sq = torch.sum(tau_mae * tau_mae).item()
        con_sq = torch.sum(tau_con * tau_con).item()
        if group not in norms:
            norms[group] = [0.0, 0.0]
        norms[group][0] += mae_sq
        norms[group][1] += con_sq
    return norms


def layerwise_merge(base_model, task_mae, task_contrastive, eps=1e-8, group_mode="block", use_contrastive_for_norms=False):
    """
    Layer-wise alpha merge based on task-vector magnitudes.

    alpha_g = ||tau_mae|| / (||tau_mae|| + ||tau_con||)
    merged = base + alpha_g * tau_mae + (1 - alpha_g) * tau_con
    """
    merged = OrderedDict()
    group_norms = compute_group_norms(base_model, task_mae, task_contrastive, group_mode=group_mode)

    norm_layers_overridden = []
    for key in base_model.keys():
        base_weight = base_model[key]
        mae_weight = task_mae.get(key, base_weight)
        contrastive_weight = task_contrastive.get(key, base_weight)

        if use_contrastive_for_norms and is_contrastive_only_layer(key):
            merged[key] = contrastive_weight
            norm_layers_overridden.append(key)
            continue

        group = group_key(key, group_mode=group_mode)
        mae_sq, con_sq = group_norms[group]
        alpha = mae_sq / (mae_sq + con_sq + eps)
        tau_mae = mae_weight - base_weight
        tau_con = contrastive_weight - base_weight
        merged[key] = base_weight + alpha * tau_mae + (1 - alpha) * tau_con

    for model, name in [(task_mae, 'MAE'), (task_contrastive, 'Contrastive')]:
        for key in model.keys():
            if key not in merged:
                print(f"Warning: Key '{key}' found in {name} model but not in base. Using {name} weights directly.")
                merged[key] = model[key]

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


def apply_dare(tau, keep_ratio=0.2, rescale=False):
    if keep_ratio >= 1.0:
        return tau
    flat = tau.abs().flatten()
    if flat.numel() == 0:
        return tau
    k = max(1, int(keep_ratio * flat.numel()))
    if k >= flat.numel():
        return tau
    threshold = torch.topk(flat, k, largest=True).values.min()
    mask = tau.abs() >= threshold
    pruned = tau * mask
    if rescale:
        pruned = pruned / keep_ratio
    return pruned


def ties_merge(task_vectors, reduce="mean", eps=1e-8):
    stacked = torch.stack(task_vectors, dim=0)
    sign = torch.sign(stacked.sum(dim=0))
    merged = torch.zeros_like(task_vectors[0])
    count = torch.zeros_like(task_vectors[0])
    for tau in task_vectors:
        mask = (torch.sign(tau) == sign) & (sign != 0)
        merged = merged + tau * mask
        count = count + mask.float()
    if reduce == "mean":
        merged = merged / torch.clamp(count, min=1.0)
    return merged


def dare_ties_merge(base_model, task_mae, task_contrastive, keep_ratio=0.2, rescale=False, reduce="mean",
                    lambda_scale=1.0, use_contrastive_for_norms=False):
    """
    DARE-TIES: sparsify task vectors then merge with TIES sign-consensus.
    merged = base + lambda * TIES(DARE(tau_mae), DARE(tau_con))
    """
    merged = OrderedDict()
    norm_layers_overridden = []
    for key in base_model.keys():
        base_weight = base_model[key]
        mae_weight = task_mae.get(key, base_weight)
        contrastive_weight = task_contrastive.get(key, base_weight)

        if use_contrastive_for_norms and is_contrastive_only_layer(key):
            merged[key] = contrastive_weight
            norm_layers_overridden.append(key)
            continue

        tau_mae = apply_dare(mae_weight - base_weight, keep_ratio=keep_ratio, rescale=rescale)
        tau_con = apply_dare(contrastive_weight - base_weight, keep_ratio=keep_ratio, rescale=rescale)
        merged_tau = ties_merge([tau_mae, tau_con], reduce=reduce)
        merged[key] = base_weight + lambda_scale * merged_tau

    for model, name in [(task_mae, 'MAE'), (task_contrastive, 'Contrastive')]:
        for key in model.keys():
            if key not in merged:
                print(f"Warning: Key '{key}' found in {name} model but not in base. Using {name} weights directly.")
                merged[key] = model[key]

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


def compute_cosine_similarity(vec_a, vec_b, eps=1e-8):
    """
    Compute cosine similarity between two tensors (flattened).
    Returns a scalar value in [-1, 1].
    """
    flat_a = vec_a.flatten().float()
    flat_b = vec_b.flatten().float()

    norm_a = torch.norm(flat_a)
    norm_b = torch.norm(flat_b)

    if norm_a < eps or norm_b < eps:
        return 0.0  # Treat zero vectors as orthogonal

    return (torch.dot(flat_a, flat_b) / (norm_a * norm_b)).item()


def project_onto(vec_source, vec_target, eps=1e-8):
    """
    Project vec_source onto vec_target.
    proj_{target}(source) = (source · target / ||target||²) * target

    Returns a tensor with the same shape as vec_source.
    """
    original_shape = vec_source.shape
    flat_source = vec_source.flatten().float()
    flat_target = vec_target.flatten().float()

    target_norm_sq = torch.dot(flat_target, flat_target)

    if target_norm_sq < eps:
        # Target is zero vector, return zero projection
        return torch.zeros_like(vec_source)

    scale = torch.dot(flat_source, flat_target) / target_norm_sq
    projection = scale * flat_target

    return projection.reshape(original_shape).to(vec_source.dtype)


def orthogonal_task_vector_merge(base_model, task_mae, task_contrastive,
                                  beta=1.0, tau=0.0, eps=1e-8,
                                  use_contrastive_for_norms=False,
                                  group_mode="param", verbose=False):
    """
    Conflict-aware orthogonal task-vector merge.

    For each parameter/group:
      Δ_c = W_contrastive - W_base  (contrastive task vector)
      Δ_m = W_mae - W_base          (MAE task vector)
      s = cos(Δ_c, Δ_m)             (cosine similarity)

      If s >= τ (aligned objectives):
        W_merged = W_base + Δ_m + β·Δ_c
      If s < τ (conflicting objectives):
        Δ_c_⊥ = Δ_c - proj_{Δ_m}(Δ_c)  (orthogonal component)
        W_merged = W_base + Δ_m + β·Δ_c_⊥

    This preserves MAE features while only adding the non-conflicting
    components of contrastive learning.

    Args:
        base_model: Base/initial model weights (pre-training checkpoint)
        task_mae: MAE-only trained model weights
        task_contrastive: Contrastive-only trained model weights
        beta: Scaling factor for contrastive task vector (default 1.0)
        tau: Conflict threshold - below this cosine similarity, project away conflicts (default 0.0)
        eps: Numerical stability epsilon
        use_contrastive_for_norms: If True, take norm_a/norm_v from contrastive only
        group_mode: "param" for per-parameter, "block" for per-block grouping
        verbose: If True, print conflict statistics

    Returns:
        Merged state_dict
    """
    merged = OrderedDict()

    # Statistics tracking
    stats = {
        'aligned': 0,       # cos >= tau
        'conflicting': 0,   # cos < tau
        'zero_mae': 0,      # MAE task vector is ~zero
        'zero_con': 0,      # Contrastive task vector is ~zero
        'similarities': []  # List of (key, cos_sim) for analysis
    }

    norm_layers_overridden = []

    # If using block grouping, we need to compute group-level statistics first
    if group_mode == "block":
        # Compute task vectors per group, then merge at group level
        group_task_vecs = {}
        for key in base_model.keys():
            group = group_key(key, group_mode="block")
            if group not in group_task_vecs:
                group_task_vecs[group] = {'mae': [], 'con': [], 'keys': []}

            base_weight = base_model[key]
            mae_weight = task_mae.get(key, base_weight)
            con_weight = task_contrastive.get(key, base_weight)

            group_task_vecs[group]['mae'].append((mae_weight - base_weight).flatten())
            group_task_vecs[group]['con'].append((con_weight - base_weight).flatten())
            group_task_vecs[group]['keys'].append(key)

        # Compute per-group cosine similarities
        group_cos_sims = {}
        for group, vecs in group_task_vecs.items():
            cat_mae = torch.cat(vecs['mae'])
            cat_con = torch.cat(vecs['con'])
            group_cos_sims[group] = compute_cosine_similarity(cat_mae, cat_con, eps)

    for key in base_model.keys():
        base_weight = base_model[key]
        mae_weight = task_mae.get(key, base_weight)
        con_weight = task_contrastive.get(key, base_weight)

        # Handle norm layers specially
        if use_contrastive_for_norms and is_contrastive_only_layer(key):
            merged[key] = con_weight
            norm_layers_overridden.append(key)
            continue

        # Compute task vectors
        tau_mae = mae_weight - base_weight
        tau_con = con_weight - base_weight

        # Check for zero vectors
        mae_norm = torch.norm(tau_mae.flatten().float())
        con_norm = torch.norm(tau_con.flatten().float())

        if mae_norm < eps:
            # MAE didn't change this parameter much, use contrastive directly
            merged[key] = base_weight + beta * tau_con
            stats['zero_mae'] += 1
            continue

        if con_norm < eps:
            # Contrastive didn't change this parameter much, use MAE directly
            merged[key] = base_weight + tau_mae
            stats['zero_con'] += 1
            continue

        # Get cosine similarity (per-param or per-group)
        if group_mode == "block":
            group = group_key(key, group_mode="block")
            cos_sim = group_cos_sims[group]
        else:
            cos_sim = compute_cosine_similarity(tau_mae, tau_con, eps)

        stats['similarities'].append((key, cos_sim))

        if cos_sim >= tau:
            # Objectives are aligned - simply add both task vectors
            merged[key] = base_weight + tau_mae + beta * tau_con
            stats['aligned'] += 1
        else:
            # Objectives conflict - project away the conflicting component
            # Δ_c_⊥ = Δ_c - proj_{Δ_m}(Δ_c)
            projection = project_onto(tau_con, tau_mae, eps)
            tau_con_orthogonal = tau_con - projection

            merged[key] = base_weight + tau_mae + beta * tau_con_orthogonal
            stats['conflicting'] += 1

    # Handle keys only in task models
    for model, name in [(task_mae, 'MAE'), (task_contrastive, 'Contrastive')]:
        for key in model.keys():
            if key not in merged:
                print(f"Warning: Key '{key}' found in {name} model but not in base. Using {name} weights directly.")
                merged[key] = model[key]

    # Print statistics
    if verbose or True:  # Always print for now
        print(f"\n  Orthogonal Merge Statistics (beta={beta}, tau={tau}):")
        print(f"    Aligned parameters (cos >= {tau}): {stats['aligned']}")
        print(f"    Conflicting parameters (cos < {tau}): {stats['conflicting']}")
        print(f"    Zero MAE task vectors: {stats['zero_mae']}")
        print(f"    Zero Contrastive task vectors: {stats['zero_con']}")

        if stats['similarities']:
            sims = [s for _, s in stats['similarities']]
            print(f"    Cosine similarity range: [{min(sims):.4f}, {max(sims):.4f}]")
            print(f"    Mean cosine similarity: {sum(sims)/len(sims):.4f}")

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


def fisher_merge(model_mae, model_contrastive, fisher_mae, fisher_contrastive, base_model=None,
                 eps=1e-8, use_contrastive_for_norms=False):
    """
    Fisher-weighted merge:
      merged = (F_mae * W_mae + F_con * W_con) / (F_mae + F_con)
    If base_model is provided, it operates on task vectors:
      merged = base + (F_mae * tau_mae + F_con * tau_con) / (F_mae + F_con)
    """
    merged = OrderedDict()
    all_keys = set(model_mae.keys()) | set(model_contrastive.keys())
    if base_model is not None:
        all_keys |= set(base_model.keys())

    norm_layers_overridden = []
    for key in all_keys:
        if key in model_mae:
            mae_weight = model_mae[key]
        elif base_model is not None and key in base_model:
            mae_weight = base_model[key]
        else:
            mae_weight = None

        if key in model_contrastive:
            contrastive_weight = model_contrastive[key]
        elif base_model is not None and key in base_model:
            contrastive_weight = base_model[key]
        else:
            contrastive_weight = None

        if mae_weight is None and contrastive_weight is None:
            continue

        if use_contrastive_for_norms and is_contrastive_only_layer(key) and contrastive_weight is not None:
            merged[key] = contrastive_weight
            norm_layers_overridden.append(key)
            continue

        if mae_weight is None:
            merged[key] = contrastive_weight
            continue
        if contrastive_weight is None:
            merged[key] = mae_weight
            continue

        f_mae = fisher_mae.get(key)
        f_con = fisher_contrastive.get(key)

        if f_mae is None and f_con is None:
            merged[key] = (mae_weight + contrastive_weight) / 2.0
            continue

        if f_mae is None:
            f_mae = torch.zeros_like(mae_weight)
        if f_con is None:
            f_con = torch.zeros_like(contrastive_weight)

        denom = f_mae + f_con
        if base_model is not None and key in base_model:
            base_weight = base_model[key]
            tau_mae = mae_weight - base_weight
            tau_con = contrastive_weight - base_weight
            merged_weight = base_weight + (f_mae * tau_mae + f_con * tau_con) / (denom + eps)
            fallback = base_weight + 0.5 * (tau_mae + tau_con)
        else:
            merged_weight = (f_mae * mae_weight + f_con * contrastive_weight) / (denom + eps)
            fallback = 0.5 * (mae_weight + contrastive_weight)

        merged[key] = torch.where(denom > eps, merged_weight, fallback)

    if norm_layers_overridden:
        print(f"  Using contrastive-only weights for: {norm_layers_overridden}")

    return merged


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
                        choices=['simple', 'weighted', 'task_arithmetic', 'layerwise', 'dare_ties', 'fisher', 'orthogonal'],
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

    parser.add_argument('--layerwise_group', type=str, default='block', choices=['block', 'param'],
                        help='Grouping strategy for layer-wise alpha')
    parser.add_argument('--layerwise_eps', type=float, default=1e-8,
                        help='Numerical stability epsilon for layer-wise alpha')

    parser.add_argument('--dare_keep_ratio', type=float, default=0.2,
                        help='Fraction of parameters to keep per tensor for DARE')
    parser.add_argument('--dare_rescale', action='store_true',
                        help='Rescale DARE-pruned task vectors by 1/keep_ratio')
    parser.add_argument('--ties_reduce', type=str, default='mean', choices=['mean', 'sum'],
                        help='Reduce strategy for TIES merge')

    parser.add_argument('--fisher_mae', type=str, default=None,
                        help='Path to Fisher diagonal for MAE model (state_dict-like)')
    parser.add_argument('--fisher_contrastive', type=str, default=None,
                        help='Path to Fisher diagonal for contrastive model (state_dict-like)')
    parser.add_argument('--fisher_eps', type=float, default=1e-8,
                        help='Numerical stability epsilon for fisher merge')

    # Orthogonal (conflict-aware) merge arguments
    parser.add_argument('--ortho_beta', type=float, default=1.0,
                        help='Scaling factor for contrastive task vector in orthogonal merge (default 1.0)')
    parser.add_argument('--ortho_tau', type=float, default=0.0,
                        help='Conflict threshold for orthogonal merge. Below this cosine similarity, project away conflicts (default 0.0)')
    parser.add_argument('--ortho_group', type=str, default='param', choices=['param', 'block'],
                        help='Grouping strategy for orthogonal merge: "param" for per-parameter, "block" for per-block')

    args = parser.parse_args()

    # Validate arguments
    if args.method in ['task_arithmetic', 'layerwise', 'dare_ties', 'orthogonal'] and args.model_base is None:
        raise ValueError("--model_base is required for task_arithmetic, layerwise, dare_ties, and orthogonal methods")

    if args.method == 'weighted' and not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0.0 and 1.0")

    if args.method == 'dare_ties' and not (0.0 < args.dare_keep_ratio <= 1.0):
        raise ValueError("--dare_keep_ratio must be within (0.0, 1.0]")

    if args.method == 'fisher':
        if args.fisher_mae is None or args.fisher_contrastive is None:
            raise ValueError("--fisher_mae and --fisher_contrastive are required for fisher method")

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

    elif args.method == 'layerwise':
        model_base = load_model(args.model_base)
        print_model_stats(model_base, "Base model")
        print("Formula: merged = base + alpha_g * tau_mae + (1 - alpha_g) * tau_contrastive")
        print("         where alpha_g is computed per group from task-vector magnitudes")
        merged = layerwise_merge(model_base, model_mae, model_contrastive,
                                 eps=args.layerwise_eps, group_mode=args.layerwise_group,
                                 use_contrastive_for_norms=args.use_contrastive_for_norms)

    elif args.method == 'dare_ties':
        model_base = load_model(args.model_base)
        print_model_stats(model_base, "Base model")
        print(f"Formula: merged = base + {args.lambda_scale} * TIES(DARE(tau_mae), DARE(tau_contrastive))")
        print(f"         keep_ratio={args.dare_keep_ratio}, rescale={args.dare_rescale}, reduce={args.ties_reduce}")
        merged = dare_ties_merge(model_base, model_mae, model_contrastive,
                                 keep_ratio=args.dare_keep_ratio, rescale=args.dare_rescale,
                                 reduce=args.ties_reduce, lambda_scale=args.lambda_scale,
                                 use_contrastive_for_norms=args.use_contrastive_for_norms)

    elif args.method == 'fisher':
        fisher_mae = load_model(args.fisher_mae)
        fisher_con = load_model(args.fisher_contrastive)
        model_base = None
        if args.model_base is not None:
            model_base = load_model(args.model_base)
            print_model_stats(model_base, "Base model")
        print("Formula: merged = (F_mae * W_mae + F_con * W_con) / (F_mae + F_con)")
        if model_base is not None:
            print("         operating on task vectors relative to base model")
        merged = fisher_merge(model_mae, model_contrastive, fisher_mae, fisher_con,
                              base_model=model_base, eps=args.fisher_eps,
                              use_contrastive_for_norms=args.use_contrastive_for_norms)

    elif args.method == 'orthogonal':
        model_base = load_model(args.model_base)
        print_model_stats(model_base, "Base model")
        print(f"Formula: Conflict-aware orthogonal merge (beta={args.ortho_beta}, tau={args.ortho_tau})")
        print(f"         For cos(tau_mae, tau_con) >= {args.ortho_tau}: merged = base + tau_mae + {args.ortho_beta}*tau_con")
        print(f"         For cos(tau_mae, tau_con) < {args.ortho_tau}: merged = base + tau_mae + {args.ortho_beta}*tau_con_orthogonal")
        print(f"         where tau_con_orthogonal = tau_con - proj_{{tau_mae}}(tau_con)")
        merged = orthogonal_task_vector_merge(model_base, model_mae, model_contrastive,
                                               beta=args.ortho_beta, tau=args.ortho_tau,
                                               use_contrastive_for_norms=args.use_contrastive_for_norms,
                                               group_mode=args.ortho_group)

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
