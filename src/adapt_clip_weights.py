import torch
from transformers import CLIPVisionModel
from collections import OrderedDict
import sys

MODAL_SPECIFIC_LAYER = 11  # 11 layers specific, 1 shared (total 12)

# Load CLIP vision model
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
clip_weights = clip_model.state_dict()
initial_weights = torch.load('IN-initial.pth')

additional_weights = OrderedDict()

# Handle CLS token (class embedding in CLIP)
cls_token = clip_weights['vision_model.embeddings.class_embedding']
additional_weights['cls_token_a'] = cls_token.detach().clone()
additional_weights['cls_token_v'] = cls_token.detach().clone()
additional_weights['cls_token_av'] = cls_token.detach().clone()

# Handle modality-specific layers
for key in clip_weights.keys():
    if 'encoder.layers' in key:
        layer_id = int(key.split('.')[3])
        remaining_key = '.'.join(key.split('.')[4:])
        
        # Convert CLIP attention naming to CAV-MAE naming
        if 'self_attn' in remaining_key:
            if any(x in remaining_key for x in ['q_proj', 'k_proj', 'v_proj']):
                continue  # Skip individual q,k,v - we'll handle them together
            elif 'out_proj' in remaining_key:
                remaining_key = remaining_key.replace('self_attn.out_proj', 'attn.proj')
        
        # Convert layer norm naming
        remaining_key = remaining_key.replace('layer_norm', 'norm')
        
        if layer_id <= MODAL_SPECIFIC_LAYER-1:
            additional_weights[f'module.blocks_a.{layer_id}.{remaining_key}'] = clip_weights[key].detach().clone()
            additional_weights[f'module.blocks_v.{layer_id}.{remaining_key}'] = clip_weights[key].detach().clone()
        else:
            new_id = layer_id - MODAL_SPECIFIC_LAYER
            additional_weights[f'module.blocks_u.{new_id}.{remaining_key}'] = clip_weights[key].detach().clone()

# Combine q,k,v into qkv for each layer
for layer_id in range(12):
    prefix = f'vision_model.encoder.layers.{layer_id}.self_attn'
    if layer_id <= MODAL_SPECIFIC_LAYER-1:
        for modal in ['a', 'v']:
            q = clip_weights[f'{prefix}.q_proj.weight']
            k = clip_weights[f'{prefix}.k_proj.weight']
            v = clip_weights[f'{prefix}.v_proj.weight']
            qkv_weight = torch.cat([q, k, v], dim=0)
            additional_weights[f'module.blocks_{modal}.{layer_id}.attn.qkv.weight'] = qkv_weight
            
            q_bias = clip_weights[f'{prefix}.q_proj.bias']
            k_bias = clip_weights[f'{prefix}.k_proj.bias']
            v_bias = clip_weights[f'{prefix}.v_proj.bias']
            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            additional_weights[f'module.blocks_{modal}.{layer_id}.attn.qkv.bias'] = qkv_bias
    else:
        new_id = layer_id - MODAL_SPECIFIC_LAYER
        q = clip_weights[f'{prefix}.q_proj.weight']
        k = clip_weights[f'{prefix}.k_proj.weight']
        v = clip_weights[f'{prefix}.v_proj.weight']
        qkv_weight = torch.cat([q, k, v], dim=0)
        additional_weights[f'module.blocks_u.{new_id}.attn.qkv.weight'] = qkv_weight
        
        q_bias = clip_weights[f'{prefix}.q_proj.bias']
        k_bias = clip_weights[f'{prefix}.k_proj.bias']
        v_bias = clip_weights[f'{prefix}.v_proj.bias']
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        additional_weights[f'module.blocks_u.{new_id}.attn.qkv.bias'] = qkv_bias

# Handle embeddings and norms
for key in clip_weights.keys():
    if 'patch_embedding' in key:
        additional_weights[key.replace('vision_model.embeddings.patch_embedding', 'module.patch_embed_a.proj')] = clip_weights[key].detach().clone()
        additional_weights[key.replace('vision_model.embeddings.patch_embedding', 'module.patch_embed_v.proj')] = clip_weights[key].detach().clone()
    elif 'position_embedding' in key:
        additional_weights[key.replace('vision_model.embeddings.position_embedding', 'module.pos_embed_a')] = clip_weights[key].detach().clone()
        additional_weights[key.replace('vision_model.embeddings.position_embedding', 'module.pos_embed_v')] = clip_weights[key].detach().clone()
    elif 'pre_layrnorm' in key:
        additional_weights[key.replace('vision_model.pre_layrnorm', 'module.norm_a')] = clip_weights[key].detach().clone()
        additional_weights[key.replace('vision_model.pre_layrnorm', 'module.norm_v')] = clip_weights[key].detach().clone()
    elif 'post_layernorm' in key:
        additional_weights[key.replace('vision_model.post_layernorm', 'module.norm')] = clip_weights[key].detach().clone()

# Initialize modality tokens (these are learned embeddings)
additional_weights['modality_a'] = torch.zeros_like(clip_weights['vision_model.embeddings.position_embedding.weight'][0])
additional_weights['modality_v'] = torch.zeros_like(clip_weights['vision_model.embeddings.position_embedding.weight'][0])

# Copy all decoder-related weights from IN-initial
for k, v in initial_weights.items():
    if any(x in k for x in ['decoder', 'mask_token', 'modality_', 'register_tokens']):
        additional_weights[k] = v.clone()

# Initialize modality-specific norms for each block
for block_type in ['blocks_a', 'blocks_v', 'blocks_u']:
    num_blocks = 11 if block_type != 'blocks_u' else 1
    for i in range(num_blocks):
        for norm_type in ['norm1', 'norm2']:
            for modality in ['a', 'v']:
                norm_weight = initial_weights[f'module.{block_type}.{i}.{norm_type}.weight'].clone()
                norm_bias = initial_weights[f'module.{block_type}.{i}.{norm_type}.bias'].clone()
                additional_weights[f'module.{block_type}.{i}.{norm_type}_{modality}.weight'] = norm_weight
                additional_weights[f'module.{block_type}.{i}.{norm_type}_{modality}.bias'] = norm_bias

# Add position embeddings and patch embed bias
additional_weights['module.pos_embed_a'] = initial_weights['module.pos_embed_a']
additional_weights['module.pos_embed_v'] = initial_weights['module.pos_embed_v']
additional_weights['module.patch_embed_a.proj.bias'] = initial_weights['module.patch_embed_a.proj.bias']
additional_weights['module.patch_embed_v.proj.bias'] = initial_weights['module.patch_embed_v.proj.bias']

# Remove unnecessary keys
for key in ['cls_token_a', 'cls_token_v', 'cls_token_av', 'modality_a', 'modality_v']:
    if key in additional_weights:
        del additional_weights[key]

# Adjust patch embedding from 3 to 1 channel
patch_embed = additional_weights['module.patch_embed_a.proj.weight']
additional_weights['module.patch_embed_a.proj.weight'] = patch_embed.mean(dim=1, keepdim=True)
additional_weights['module.patch_embed_v.proj.weight'] = patch_embed.mean(dim=1, keepdim=True)

# Fix position embedding keys and shapes
if 'module.pos_embed_a.weight' in additional_weights:
    additional_weights['module.pos_embed_a'] = additional_weights.pop('module.pos_embed_a.weight')
if 'module.pos_embed_v.weight' in additional_weights:
    additional_weights['module.pos_embed_v'] = additional_weights.pop('module.pos_embed_v.weight')

# Fix position embedding shapes
pos_embed_a = additional_weights['module.pos_embed_a']
pos_embed_v = additional_weights['module.pos_embed_v']

print(f"Original pos_embed_a shape: {pos_embed_a.shape}, total elements: {pos_embed_a.numel()}")
print(f"Original pos_embed_v shape: {pos_embed_v.shape}, total elements: {pos_embed_v.numel()}")

# Get shapes from working weights
working_weights = torch.load('IN-initial.pth')
working_shape_a = working_weights['module.pos_embed_a'].shape
working_shape_v = working_weights['module.pos_embed_v'].shape

print(f"Target shapes - A: {working_shape_a}, V: {working_shape_v}")

# Copy position embeddings directly from working weights
additional_weights['module.pos_embed_a'] = working_weights['module.pos_embed_a'].clone()
additional_weights['module.pos_embed_v'] = working_weights['module.pos_embed_v'].clone()

# Fix patch embedding shape from 1 to 3 channels
patch_embed_v = additional_weights['module.patch_embed_v.proj.weight']
patch_embed_a = additional_weights['module.patch_embed_a.proj.weight']

if patch_embed_v.shape[1] == 1:  # If it's 1 channel, expand to 3
    # Repeat the single channel 3 times
    additional_weights['module.patch_embed_v.proj.weight'] = patch_embed_v.repeat(1, 3, 1, 1)
    # additional_weights['module.patch_embed_a.proj.weight'] = patch_embed_a.repeat(1, 3, 1, 1)

print(f"Final patch_embed_v shape: {additional_weights['module.patch_embed_v.proj.weight'].shape}")
print(f"Final patch_embed_a shape: {additional_weights['module.patch_embed_a.proj.weight'].shape}")

# Save the adapted weights
torch.save(additional_weights, 'adapted_clip_weights.pth')

# Redirect stdout to file
with open('weight_differences.txt', 'w') as f:
    sys.stdout = f
    
    # Load working weights
    working_weights = torch.load('IN-initial.pth')
    current_weights = torch.load('adapted_clip_weights.pth')

    # Compare structure
    print("=== Missing Keys ===")
    for k in working_weights.keys():
        if k not in current_weights:
            print(f"Missing: {k}")

    print("\n=== Extra Keys ===")
    for k in current_weights.keys():
        if k not in working_weights:
            print(f"Extra: {k}")

    print("\n=== Shape Mismatches ===")
    for k in working_weights.keys():
        if k in current_weights:
            if working_weights[k].shape != current_weights[k].shape:
                print(f"{k}: Working {working_weights[k].shape} vs Current {current_weights[k].shape}")

    # Reset stdout
    sys.stdout = sys.__stdout__
