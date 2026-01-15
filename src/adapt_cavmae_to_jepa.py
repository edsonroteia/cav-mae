# -*- coding: utf-8 -*-
# @Time    : 1/15/26
# @File    : adapt_cavmae_to_jepa.py
# @Description: Adapt CAV-MAE initialization weights for CAV-JEPA
#
# This script takes the same ImageMAE-based initialization used by CAV-MAE
# and adapts it for CAV-JEPA by:
# 1. Keeping all encoder weights (blocks_a, blocks_v, blocks_u, patch_embed, pos_embed, norm)
# 2. Creating target encoder as copy of context encoder
# 3. Skipping decoder weights (JEPA uses predictor instead)
# 4. Initializing predictor randomly

import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models


def adapt_cavmae_for_jepa(cavmae_ckpt_path, output_path, predictor_depth=4, predictor_dim=384):
    """
    Adapt CAV-MAE initialization weights for CAV-JEPA.

    Args:
        cavmae_ckpt_path: Path to CAV-MAE initialization checkpoint (e.g., IN-initial.pth)
        output_path: Path to save adapted weights
        predictor_depth: Depth of JEPA predictor
        predictor_dim: Dimension of JEPA predictor
    """
    print(f"=== Adapting CAV-MAE weights for CAV-JEPA ===")
    print(f"Input: {cavmae_ckpt_path}")
    print(f"Output: {output_path}")
    print(f"Predictor: depth={predictor_depth}, dim={predictor_dim}")

    # Load CAV-MAE checkpoint
    print("\nLoading CAV-MAE checkpoint...")
    ckpt = torch.load(cavmae_ckpt_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # Remove 'module.' prefix if present
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if k.startswith('module.') else k
        cleaned[new_k] = v
    state_dict = cleaned

    print(f"Loaded {len(state_dict)} keys from CAV-MAE checkpoint")

    # Separate encoder and decoder weights
    encoder_keys = []
    decoder_keys = []

    for k in state_dict.keys():
        if any(x in k for x in ['decoder_', 'mask_token']):
            decoder_keys.append(k)
        else:
            encoder_keys.append(k)

    print(f"Encoder keys: {len(encoder_keys)}")
    print(f"Decoder keys (skipped): {len(decoder_keys)}")

    # Create CAV-JEPA model
    print("\nCreating CAV-JEPA model...")
    model = models.CAVJEPA(
        audio_length=1024,
        modality_specific_depth=11,
        predictor_depth=predictor_depth,
        predictor_embed_dim=predictor_dim,
        tr_pos=False
    )

    # Build adapted state dict
    adapted = OrderedDict()

    # Copy encoder weights to context encoder (direct mapping)
    print("\nAdapting encoder weights for context encoder...")
    for k in encoder_keys:
        if k in state_dict:
            adapted[k] = state_dict[k].clone()

    # Copy encoder weights to target encoder
    print("Creating target encoder weights (copy of context encoder)...")
    target_mapping = {
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
        'norm.': 'target_norm.',  # unified norm if exists
    }

    for k in encoder_keys:
        for src_prefix, tgt_prefix in target_mapping.items():
            if k.startswith(src_prefix) or k == src_prefix.rstrip('.'):
                if k == src_prefix.rstrip('.'):
                    new_k = tgt_prefix.rstrip('.')
                else:
                    new_k = k.replace(src_prefix, tgt_prefix, 1)
                adapted[new_k] = state_dict[k].clone()
                break

    print(f"Total adapted weights: {len(adapted)}")

    # Load adapted weights into model
    print("\nLoading adapted weights into CAV-JEPA model...")
    missing, unexpected = model.load_state_dict(adapted, strict=False)

    print(f"\nMissing keys ({len(missing)}):")
    # Group missing keys by category
    predictor_missing = [k for k in missing if 'predictor' in k]
    other_missing = [k for k in missing if 'predictor' not in k]

    print(f"  - Predictor keys (expected, will be random init): {len(predictor_missing)}")
    if other_missing:
        print(f"  - Other missing keys: {len(other_missing)}")
        for k in other_missing[:10]:
            print(f"      {k}")

    if unexpected:
        print(f"\nUnexpected keys ({len(unexpected)}):")
        for k in unexpected[:10]:
            print(f"    {k}")

    # Save the full model state (including randomly initialized predictor)
    print(f"\nSaving adapted weights to {output_path}...")
    torch.save(model.state_dict(), output_path)

    # Verify the saved checkpoint
    print("\nVerifying saved checkpoint...")
    verify_ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
    print(f"Saved checkpoint has {len(verify_ckpt)} keys")

    print("\n=== Adaptation complete! ===")
    print(f"You can now use this checkpoint with:")
    print(f"  --pretrain_path {output_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Adapt CAV-MAE weights for CAV-JEPA')
    parser.add_argument('--cavmae_ckpt', type=str, required=True,
                        help='Path to CAV-MAE initialization checkpoint (e.g., IN-initial.pth)')
    parser.add_argument('--output', type=str, default='cav_jepa_from_mae_init.pth',
                        help='Output path for adapted weights')
    parser.add_argument('--predictor_depth', type=int, default=4,
                        help='Predictor network depth')
    parser.add_argument('--predictor_dim', type=int, default=384,
                        help='Predictor network dimension')

    args = parser.parse_args()

    adapt_cavmae_for_jepa(
        cavmae_ckpt_path=args.cavmae_ckpt,
        output_path=args.output,
        predictor_depth=args.predictor_depth,
        predictor_dim=args.predictor_dim
    )


if __name__ == '__main__':
    main()
