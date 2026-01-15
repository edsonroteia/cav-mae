# -*- coding: utf-8 -*-
# @Time    : 1/15/26
# @Author  : Based on adapt_vmae_weights.py
# @File    : adapt_jepa_weights.py
# @Description: Adapt I-JEPA/V-JEPA/A-JEPA weights for CAV-JEPA initialization

"""
Adapt JEPA checkpoints for CAV-JEPA initialization.

Supports 4 initialization options:
  Option A: --visual_ckpt ijepa.pth --audio_ckpt ajepa.pth  (recommended)
  Option B: --visual_ckpt ijepa.pth --audio_ckpt ijepa.pth  (fallback)
  Option C: --visual_ckpt vjepa.pth --audio_ckpt vjepa.pth  (video-focused)
  Option D: --visual_ckpt vjepa.pth --audio_ckpt ajepa.pth  (hybrid)

Download checkpoints from:
  - I-JEPA: https://github.com/facebookresearch/ijepa
  - V-JEPA: https://github.com/facebookresearch/jepa
  - A-JEPA: https://github.com/facebookresearch/AudioMAE (or similar)
"""

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import models


def load_jepa_checkpoint(ckpt_path):
    """Load a JEPA checkpoint and extract the encoder weights."""
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'encoder' in ckpt:
        state_dict = ckpt['encoder']
    elif 'target_encoder' in ckpt:
        # Some JEPA checkpoints save target encoder separately
        state_dict = ckpt['target_encoder']
    else:
        # Assume the checkpoint is the state dict itself
        state_dict = ckpt

    # Remove 'module.' prefix if present (from DDP)
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        # Also handle 'encoder.' prefix
        if k.startswith('encoder.'):
            k = k[8:]
        if k.startswith('target_encoder.'):
            k = k[15:]
        cleaned_state_dict[k] = v

    return cleaned_state_dict


def adapt_weights_for_modality(jepa_weights, modality, modal_specific_layer=11):
    """
    Adapt JEPA encoder weights for a specific modality (audio or visual).

    Args:
        jepa_weights: State dict from JEPA checkpoint
        modality: 'a' for audio, 'v' for visual
        modal_specific_layer: Number of modality-specific layers (default 11)

    Returns:
        Adapted weights dict
    """
    adapted = OrderedDict()
    prefix = f'blocks_{modality}.'

    for key, value in jepa_weights.items():
        # Handle transformer blocks
        if key.startswith('blocks.'):
            parts = key.split('.')
            block_id = int(parts[1])
            rest = '.'.join(parts[2:])

            if block_id < modal_specific_layer:
                # Modality-specific blocks
                new_key = f'{prefix}{block_id}.{rest}'
                adapted[new_key] = value.detach().clone()
            else:
                # Unified blocks (only add once, will be handled separately)
                unified_id = block_id - modal_specific_layer
                new_key = f'blocks_u.{unified_id}.{rest}'
                if new_key not in adapted:
                    adapted[new_key] = value.detach().clone()

                # Also add modality-specific norm layers for unified blocks
                if 'norm1.weight' in rest:
                    adapted[f'blocks_u.{unified_id}.norm1_{modality}.weight'] = value.detach().clone()
                elif 'norm1.bias' in rest:
                    adapted[f'blocks_u.{unified_id}.norm1_{modality}.bias'] = value.detach().clone()
                elif 'norm2.weight' in rest:
                    adapted[f'blocks_u.{unified_id}.norm2_{modality}.weight'] = value.detach().clone()
                elif 'norm2.bias' in rest:
                    adapted[f'blocks_u.{unified_id}.norm2_{modality}.bias'] = value.detach().clone()

        # Handle final norm
        elif key == 'norm.weight':
            adapted[f'norm_{modality}.weight'] = value.detach().clone()
        elif key == 'norm.bias':
            adapted[f'norm_{modality}.bias'] = value.detach().clone()

        # Handle patch embedding
        elif key.startswith('patch_embed.'):
            rest = key[len('patch_embed.'):]
            new_key = f'patch_embed_{modality}.{rest}'

            # For audio, we need to adapt RGB (3 channels) to 1 channel
            if modality == 'a' and 'proj.weight' in rest:
                # Sum RGB channels to get single channel
                adapted[new_key] = torch.sum(value, dim=1, keepdim=True).detach().clone()
            else:
                adapted[new_key] = value.detach().clone()

        # Handle positional embedding
        elif key == 'pos_embed':
            # Remove CLS token if present (first token)
            if value.shape[1] == 197:  # 196 patches + 1 CLS
                adapted[f'pos_embed_{modality}'] = value[:, 1:, :].detach().clone()
            else:
                adapted[f'pos_embed_{modality}'] = value.detach().clone()

    return adapted


def create_target_encoder_weights(context_weights):
    """
    Create target encoder weights by copying context encoder weights.

    Args:
        context_weights: State dict with context encoder weights

    Returns:
        Target encoder weights dict
    """
    target_weights = OrderedDict()

    # Map context encoder keys to target encoder keys
    key_mapping = {
        'patch_embed_a.': 'target_patch_embed_a.',
        'patch_embed_v.': 'target_patch_embed_v.',
        'modality_a': 'target_modality_a',
        'modality_v': 'target_modality_v',
        'pos_embed_a': 'target_pos_embed_a',
        'pos_embed_v': 'target_pos_embed_v',
        'blocks_a.': 'target_blocks_a.',
        'blocks_v.': 'target_blocks_v.',
        'blocks_u.': 'target_blocks_u.',
        'norm_a.': 'target_norm_a.',
        'norm_v.': 'target_norm_v.',
    }

    for key, value in context_weights.items():
        for src_prefix, tgt_prefix in key_mapping.items():
            if key.startswith(src_prefix) or key == src_prefix.rstrip('.'):
                if key == src_prefix.rstrip('.'):
                    new_key = tgt_prefix.rstrip('.')
                else:
                    new_key = key.replace(src_prefix, tgt_prefix, 1)
                target_weights[new_key] = value.detach().clone()
                break

    return target_weights


def adapt_jepa_for_cavjepa(visual_ckpt_path, audio_ckpt_path, output_path,
                           modal_specific_layer=11, predictor_depth=4, predictor_dim=384):
    """
    Main function to adapt JEPA checkpoints for CAV-JEPA.

    Args:
        visual_ckpt_path: Path to visual JEPA checkpoint (I-JEPA or V-JEPA)
        audio_ckpt_path: Path to audio JEPA checkpoint (A-JEPA or I-JEPA/V-JEPA)
        output_path: Path to save adapted weights
        modal_specific_layer: Number of modality-specific layers
        predictor_depth: Depth of predictor network
        predictor_dim: Dimension of predictor network
    """
    print(f"\n=== Adapting JEPA weights for CAV-JEPA ===")
    print(f"Visual checkpoint: {visual_ckpt_path}")
    print(f"Audio checkpoint: {audio_ckpt_path}")
    print(f"Modal-specific layers: {modal_specific_layer}")
    print(f"Predictor: depth={predictor_depth}, dim={predictor_dim}")

    # Load checkpoints
    visual_weights = load_jepa_checkpoint(visual_ckpt_path)
    audio_weights = load_jepa_checkpoint(audio_ckpt_path)

    print(f"\nVisual checkpoint keys: {len(visual_weights)}")
    print(f"Audio checkpoint keys: {len(audio_weights)}")

    # Adapt weights for each modality
    print("\nAdapting visual weights...")
    visual_adapted = adapt_weights_for_modality(visual_weights, 'v', modal_specific_layer)

    print("Adapting audio weights...")
    audio_adapted = adapt_weights_for_modality(audio_weights, 'a', modal_specific_layer)

    # Merge adapted weights
    merged_weights = OrderedDict()
    merged_weights.update(visual_adapted)
    merged_weights.update(audio_adapted)

    # Handle unified blocks - use visual weights as base
    for key in visual_adapted:
        if key.startswith('blocks_u.') and 'norm1_' not in key and 'norm2_' not in key:
            if key not in merged_weights:
                merged_weights[key] = visual_adapted[key]

    # Create target encoder weights (copy of context encoder)
    print("Creating target encoder weights...")
    target_weights = create_target_encoder_weights(merged_weights)
    merged_weights.update(target_weights)

    print(f"\nTotal adapted weights: {len(merged_weights)}")

    # Create CAV-JEPA model to verify loading
    print("\nCreating CAV-JEPA model...")
    model = models.CAVJEPA(
        modality_specific_depth=modal_specific_layer,
        predictor_depth=predictor_depth,
        predictor_embed_dim=predictor_dim
    )

    # Load adapted weights
    print("Loading adapted weights into model...")
    missing, unexpected = model.load_state_dict(merged_weights, strict=False)

    print(f"\nMissing keys ({len(missing)}):")
    for k in missing[:20]:  # Show first 20
        print(f"  {k}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")

    print(f"\nUnexpected keys ({len(unexpected)}):")
    for k in unexpected[:20]:
        print(f"  {k}")
    if len(unexpected) > 20:
        print(f"  ... and {len(unexpected) - 20} more")

    # Save adapted weights
    print(f"\nSaving to {output_path}...")
    torch.save(model.state_dict(), output_path)
    print("Done!")

    return model


def main():
    parser = argparse.ArgumentParser(description='Adapt JEPA weights for CAV-JEPA')
    parser.add_argument('--visual_ckpt', type=str, required=True,
                        help='Path to visual JEPA checkpoint (I-JEPA or V-JEPA)')
    parser.add_argument('--audio_ckpt', type=str, required=True,
                        help='Path to audio JEPA checkpoint (A-JEPA, I-JEPA, or V-JEPA)')
    parser.add_argument('--output', type=str, default='cav_jepa_init.pth',
                        help='Output path for adapted weights')
    parser.add_argument('--modal_specific_depth', type=int, default=11,
                        help='Number of modality-specific layers')
    parser.add_argument('--predictor_depth', type=int, default=4,
                        help='Predictor network depth')
    parser.add_argument('--predictor_dim', type=int, default=384,
                        help='Predictor network dimension')

    args = parser.parse_args()

    adapt_jepa_for_cavjepa(
        visual_ckpt_path=args.visual_ckpt,
        audio_ckpt_path=args.audio_ckpt,
        output_path=args.output,
        modal_specific_layer=args.modal_specific_depth,
        predictor_depth=args.predictor_depth,
        predictor_dim=args.predictor_dim
    )


if __name__ == '__main__':
    main()
