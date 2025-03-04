import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import models
from dataloader_sync import AudiosetDataset, eval_collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os

def get_attention_maps(model, layer_idx, head_idx=None):
    """
    Extract attention maps from a specific layer and optionally a specific head
    """
    attention_maps = {'attn_weights': None, 'count': 0}  # Use running average instead of list
    
    def hook_fn(module, input, output):
        print(f"Hook called for layer {layer_idx}")
        
        try:
            # For the Attention module in timm, the attention weights are computed inside
            # We need to recompute them here
            
            # Get query, key, value projections
            qkv = module.qkv(input[0])
            B, N, C = input[0].shape
            num_heads = module.num_heads
            head_dim = C // num_heads
            
            # Reshape qkv to get q, k, v separately
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Compute attention weights
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            
            # Use running average to handle different sequence lengths
            if attention_maps['attn_weights'] is None:
                attention_maps['attn_weights'] = attn.detach().mean(dim=0).cpu()  # Average over batch
                attention_maps['count'] = 1
            else:
                # Only update if the sequence length matches
                if attn.shape[2] == attention_maps['attn_weights'].shape[1]:
                    attention_maps['attn_weights'] = (attention_maps['attn_weights'] * attention_maps['count'] + 
                                                     attn.detach().mean(dim=0).cpu()) / (attention_maps['count'] + 1)
                    attention_maps['count'] += 1
                else:
                    print(f"Skipping batch with different sequence length: {attn.shape[2]} vs {attention_maps['attn_weights'].shape[1]}")
            
            print(f"Attention shape: {attn.shape}, Running count: {attention_maps['count']}")
            
        except Exception as e:
            print(f"Error in hook: {e}")
        
        # Return the original output to not affect the forward pass
        return output
    
    # Register hook on the specified layer
    if isinstance(model, nn.DataParallel):
        target_layer = model.module.blocks_u[layer_idx]
    else:
        target_layer = model.blocks_u[layer_idx]
    
    hook_handle = target_layer.attn.register_forward_hook(hook_fn)
    
    return attention_maps, hook_handle

def visualize_attention(attention_maps, save_dir, prefix="", num_register_tokens=8, patch_size=14):
    """
    Visualize attention maps for CLS token and register tokens, averaged across all heads
    Each token gets its own separate plot without a colorbar
    For patch token, we use the middle patch instead of a random one
    """
    # Convert to absolute path and ensure directory exists
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Test file creation to verify write permissions
    test_file_path = os.path.join(save_dir, "test_write.txt")
    try:
        with open(test_file_path, 'w') as f:
            f.write("Testing write permissions")
        os.remove(test_file_path)
        print(f"Directory {save_dir} is writable")
    except Exception as e:
        print(f"Error writing to directory {save_dir}: {e}")
        return
    
    # Handle dictionary of attention maps
    if isinstance(attention_maps, dict):
        print(f"Processing attention maps dictionary with keys: {list(attention_maps.keys())}")
        
        # Extract the attention weights for the last layer
        if 'attn_weights' in attention_maps:
            attn = attention_maps['attn_weights']
            
            # Debug token structure
            num_heads, seq_len, _ = attn.shape
            print(f"Attention shape: {attn.shape}")
            print(f"Total sequence length: {seq_len}")
            
            # Based on the sequence length, determine the modality
            if seq_len == 208 + num_register_tokens + 1:  # Audio
                modality = "audio"
                grid_height, grid_width = 8, 26
                num_patch_tokens = 208
            elif seq_len == 196 + num_register_tokens + 1:  # Image
                modality = "visual"
                grid_height, grid_width = 14, 14
                num_patch_tokens = 196
            else:
                print(f"Warning: Unknown modality with sequence length {seq_len}")
                # Try to guess a reasonable grid
                grid_height = int(np.sqrt(seq_len))
                grid_width = seq_len // grid_height
                if grid_height * grid_width < seq_len:
                    grid_width += 1
                modality = "unknown"
                num_patch_tokens = seq_len - num_register_tokens - 1
            
            print(f"Detected modality: {modality}")
            print(f"Grid dimensions: {grid_height}x{grid_width}")
            
            # Average attention across all heads
            avg_attn = attn.mean(dim=0)  # Average across heads
            
            # Use the middle patch token
            middle_row = grid_height // 2
            middle_col = grid_width // 2
            middle_patch_idx = 1 + middle_row * grid_width + middle_col  # +1 for CLS token
            print(f"Using middle patch token at position ({middle_row}, {middle_col}), index: {middle_patch_idx}")
            
            # Define tokens to visualize
            tokens_to_visualize = [
                (0, "CLS"),  # CLS token
                (middle_patch_idx, "Middle_Patch")  # Middle patch token
            ]
            
            # Add register tokens
            for i in range(num_register_tokens):
                reg_idx = seq_len - num_register_tokens + i
                tokens_to_visualize.append((reg_idx, f"Register_{i}"))
            
            # Create a separate figure for each token
            for token_idx, token_name in tokens_to_visualize:
                plt.figure(figsize=(8, 6))
                
                # Get attention from this token to all patch tokens
                token_attn = avg_attn[token_idx]
                
                # Reshape to grid (only the patch tokens, excluding CLS and register tokens)
                attn_map = token_attn[1:1+num_patch_tokens].reshape(grid_height, grid_width)
                plt.imshow(attn_map, cmap='viridis')
                plt.title(f"{token_name} → Patches ({modality.capitalize()})")
                plt.axis('off')
                
                # Save the figure
                save_path = os.path.join(save_dir, f"{prefix}{modality}_{token_name}_attention.png")
                try:
                    plt.savefig(save_path, bbox_inches='tight')
                    print(f"Saved {token_name} attention map to {save_path}")
                except Exception as e:
                    print(f"Error saving {token_name} attention map: {e}")
                plt.close()
            
            # Also save raw attention data for further analysis
            np_save_path = os.path.join(save_dir, f"{prefix}{modality}_attention_raw.npy")
            try:
                np.save(np_save_path, attn.cpu().numpy() if hasattr(attn, 'cpu') else attn)
                print(f"Saved raw attention data to {np_save_path}")
            except Exception as e:
                print(f"Error saving raw attention data: {e}")
        else:
            print("Error: 'attn_weights' key not found in attention_maps dictionary")
    else:
        print(f"Error: attention_maps should be a dictionary, but got {type(attention_maps)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize attention maps for registers and CLS token')
    parser.add_argument('--pretrain_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--data_val', type=str, required=True, help='Path to validation data JSON')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to label CSV file')
    parser.add_argument('--n_register_tokens', type=int, default=4, help='Number of register tokens')
    parser.add_argument('--cls_token', type=eval, default=False, help='Whether to use CLS token')
    parser.add_argument('--total_frame', type=int, default=16, help='Total number of frames')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to process')
    parser.add_argument('--save_dir', type=str, default='attention_maps', help='Directory to save attention maps')
    parser.add_argument('--layer_idx', type=int, default=0, help='Layer index to extract attention maps from')
    parser.add_argument('--target_length', type=int, default=1024, help='Target length for audio')
    
    args = parser.parse_args()
    
    # Check if pretrain_path is a model name in models.csv
    if not os.path.exists(args.pretrain_path):
        # Try to read the path from models.csv
        try:
            models_df = pd.read_csv('models.csv')
            model_row = models_df[models_df['name'] == args.pretrain_path]
            if not model_row.empty:
                args.pretrain_path = model_row['path'].values[0]
                print(f"Found model path in CSV: {args.pretrain_path}")
            else:
                raise ValueError(f"Model {args.pretrain_path} not found in models.csv")
        except Exception as e:
            print(f"Error reading from models.csv: {e}")
            raise
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up data loader
    val_audio_conf = {
        'num_mel_bins': 128, 
        'target_length': args.target_length, 
        'freqm': 0, 
        'timem': 0, 
        'mixup': 0, 
        'dataset': 'audioset',
        'mode': 'retrieval', 
        'mean': -5.081, 
        'std': 4.4849, 
        'noise': False, 
        'im_res': 224, 
        'num_samples': args.num_samples, 
        'augmentation': False, 
        'total_frame': args.total_frame
    }
    
    val_dataset = AudiosetDataset(args.data_val, val_audio_conf, label_csv=args.label_csv)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=eval_collate_fn
    )
    
    # Create model
    audio_model = models.CAVMAEFTSync(
        audio_length=args.target_length, 
        label_dim=527,  # AudioSet has 527 classes
        modality_specific_depth=11, 
        aggregate='None', 
        num_register_tokens=args.n_register_tokens, 
        cls_token=args.cls_token, 
        total_frame=args.total_frame, 
        contrastive_head=False, 
        joint_layers=1, 
        keep_register_tokens=True,  # Keep register tokens for visualization
        mode='multimodal'
    )
    
    # Load pretrained weights
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    
    mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print(f"Missing keys: {miss}")
    print(f"Unexpected keys: {unexpected}")
    
    audio_model = audio_model.to(device)
    audio_model.eval()
    
    # Process more batches to get a better average
    with torch.no_grad():
        # First pass for audio attention
        print("Processing audio attention maps...")
        # Register hook for audio attention
        audio_attention_maps, audio_hook_handle = get_attention_maps(audio_model, args.layer_idx)
        
        # Process more samples for audio
        num_samples_to_process = 50  # Increased from 20
        for i, batch in enumerate(tqdm(val_loader, desc="Processing audio batches")):
            if batch is None:
                print(f"Skipping empty batch {i}")
                continue
            
            a_input, v_input, _, _, _ = batch
            a_input, v_input = a_input.to(device), v_input.to(device)
            
            print(f"Processing audio batch {i}, input shapes: a={a_input.shape}, v={v_input.shape}")
            
            try:
                # Forward pass with audio only
                audio_model.module.forward_feat(a_input, None, mode='a')
                print(f"Successfully processed audio batch {i}")
            except Exception as e:
                print(f"Error processing audio batch {i}: {e}")
            
            if i >= num_samples_to_process:  # Process more batches for a better average
                break
        
        # Remove audio hook and visualize
        audio_hook_handle.remove()
        visualize_attention(
            audio_attention_maps, 
            args.save_dir, 
            f'layer_{args.layer_idx}_audio_', 
            args.n_register_tokens, 
            args.cls_token
        )
        
        # Clear any residual state
        torch.cuda.empty_cache()
        
        # Second pass for visual attention - register new hook
        print("Processing visual attention maps...")
        visual_attention_maps, visual_hook_handle = get_attention_maps(audio_model, args.layer_idx)
        
        # Process more samples for visual
        for i, batch in enumerate(tqdm(val_loader, desc="Processing visual batches")):
            if batch is None:
                print(f"Skipping empty batch {i}")
                continue
            
            a_input, v_input, _, _, _ = batch
            a_input, v_input = a_input.to(device), v_input.to(device)
            
            if v_input is None or v_input.shape[0] == 0:
                print(f"Skipping batch {i} with empty visual input")
                continue
                
            print(f"Processing visual batch {i}, input shapes: a={a_input.shape}, v={v_input.shape}")
            
            try:
                # Forward pass with visual only
                audio_model.module.forward_feat(None, v_input, mode='v')
                print(f"Successfully processed visual batch {i}")
            except Exception as e:
                print(f"Error processing visual batch {i}: {e}")
            
            if i >= num_samples_to_process:  # Process more batches for a better average
                break
        
        # Check if we captured any visual attention maps
        if visual_attention_maps['attn_weights'] is None:
            print("WARNING: No visual attention maps were captured!")
        else:
            print(f"Visual attention maps shape: {visual_attention_maps['attn_weights'].shape}")
        
        # Remove visual hook and visualize
        visual_hook_handle.remove()
        visualize_attention(
            visual_attention_maps, 
            args.save_dir, 
            f'layer_{args.layer_idx}_visual_', 
            args.n_register_tokens, 
            args.cls_token
        )
    
    # Add check for existing output directory
    save_dir = os.path.abspath(args.save_dir)
    print(f"Will save output to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Check directory permissions
    try:
        test_file = os.path.join(save_dir, "permission_test.txt")
        with open(test_file, "w") as f:
            f.write("Testing write permissions")
        os.remove(test_file)
        print(f"Directory {save_dir} is writable")
    except Exception as e:
        print(f"Warning: Cannot write to directory {save_dir}: {e}")
    
    print(f"Audio and visual attention maps saved to {args.save_dir}")

if __name__ == '__main__':
    main() 