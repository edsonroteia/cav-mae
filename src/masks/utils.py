# -*- coding: utf-8 -*-
# @Time    : 1/19/26
# @Author  : Adapted from I-JEPA (Meta)
# @Description: Mask utility functions

import torch


def apply_masks(x, masks):
    """
    Apply masks to select patches from a tensor.

    Args:
        x: tensor of shape [B, N, D] (batch, num_patches, features)
        masks: list of tensors containing indices of patches to keep

    Returns:
        Concatenated selected patches
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def create_binary_mask(indices, num_patches, batch_size, device='cpu'):
    """
    Convert index-based mask to binary mask.

    Args:
        indices: [B, num_keep] indices of patches to keep
        num_patches: total number of patches
        batch_size: batch size
        device: device to place output on

    Returns:
        binary_mask: [B, num_patches] where 1 = keep, 0 = mask
    """
    binary_mask = torch.zeros(batch_size, num_patches, device=device)
    for b in range(batch_size):
        binary_mask[b, indices[b]] = 1
    return binary_mask


def binary_to_indices(binary_mask):
    """
    Convert binary mask to index-based mask.

    Args:
        binary_mask: [B, num_patches] where 1 = keep, 0 = mask

    Returns:
        indices: list of [num_keep_i] tensors with indices of kept patches
        restore_indices: [B, num_patches] indices to restore original order
    """
    batch_size = binary_mask.shape[0]
    indices = []
    restore_indices = []

    for b in range(batch_size):
        keep_idx = torch.nonzero(binary_mask[b]).squeeze(-1)
        indices.append(keep_idx)

        # Create restore indices
        restore = torch.argsort(torch.cat([
            keep_idx,
            torch.nonzero(1 - binary_mask[b]).squeeze(-1)
        ]))
        restore_indices.append(restore)

    return indices, torch.stack(restore_indices)


def expand_mask_to_full(x_masked, mask, num_patches, fill_value=0.0):
    """
    Expand masked tensor back to full sequence length.

    Args:
        x_masked: [B, num_keep, D] tensor with only kept patches
        mask: [B, num_patches] binary mask (1 = keep)
        num_patches: total number of patches
        fill_value: value to fill masked positions

    Returns:
        x_full: [B, num_patches, D] full sequence with fill_value at masked positions
    """
    B, _, D = x_masked.shape

    # Create full tensor with fill value
    x_full = torch.full((B, num_patches, D), fill_value,
                        dtype=x_masked.dtype, device=x_masked.device)

    # Get keep indices
    for b in range(B):
        keep_idx = torch.nonzero(mask[b]).squeeze(-1)
        x_full[b, keep_idx] = x_masked[b]

    return x_full


def mask_ratio_to_keep_count(num_patches, mask_ratio):
    """
    Convert mask ratio to number of patches to keep.

    Args:
        num_patches: total number of patches
        mask_ratio: fraction to mask (0.75 = mask 75%, keep 25%)

    Returns:
        num_keep: number of patches to keep
    """
    return int(num_patches * (1 - mask_ratio))


def sample_random_mask(batch_size, num_patches, num_keep, device='cpu'):
    """
    Sample random mask (unstructured masking).

    Args:
        batch_size: number of samples
        num_patches: total patches per sample
        num_keep: number of patches to keep
        device: device for output

    Returns:
        keep_indices: [B, num_keep] indices of patches to keep
        restore_indices: [B, num_patches] indices to restore original order
        binary_mask: [B, num_patches] binary mask (1 = keep)
    """
    # Random noise for shuffling
    noise = torch.rand(batch_size, num_patches, device=device)
    shuffle_ids = torch.argsort(noise, dim=1)
    restore_ids = torch.argsort(shuffle_ids, dim=1)

    # Keep first num_keep after shuffling
    keep_indices = shuffle_ids[:, :num_keep]

    # Create binary mask
    binary_mask = torch.zeros(batch_size, num_patches, device=device)
    binary_mask.scatter_(1, keep_indices, 1.0)

    return keep_indices, restore_ids, binary_mask
