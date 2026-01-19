# -*- coding: utf-8 -*-
# @Time    : 1/15/26
# @Author  : Based on CAV-MAE by Yuan Gong (MIT)
# @File    : cav_jepa.py
# @Description: CAV-JEPA v2 - Improved JEPA with proper initialization and masking
#
# Key improvements over v1:
# - Random initialization option (init_mode='random') to avoid MAE pretrain bias
# - Learnable mask tokens in predictor (instead of zeros)
# - Target representation normalization (collapse prevention)
# - Depth-scaled initialization (following I-JEPA)

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Attention, Mlp
from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Block(nn.Module):
    """Transformer block with modality-specific normalization support"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = timm.models.layers.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x


class PredictorBlock(nn.Module):
    """Lightweight predictor block for JEPA"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CAVJEPA(nn.Module):
    """CAV-JEPA: Contrastive Audio-Visual Joint-Embedding Predictive Architecture

    Replaces MAE (pixel reconstruction) with JEPA (latent prediction) while keeping
    contrastive learning. Key differences from CAV-MAE:
    - Target encoder (EMA of context encoder) instead of decoder
    - Lightweight predictor instead of heavy decoder
    - Loss in latent space instead of pixel space

    v2 Improvements:
    - init_mode='random': Fresh random initialization (avoids MAE bias)
    - init_mode='mae': Load from MAE checkpoint (original behavior)
    - Learnable mask tokens in predictor
    - Target representation normalization
    - Depth-scaled initialization for better training dynamics
    """

    def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 predictor_embed_dim=384, predictor_depth=4, predictor_num_heads=6,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, tr_pos=False,
                 momentum_start=0.996, momentum_end=0.999,
                 init_mode='mae', normalize_targets=True, init_std=0.02):
        super().__init__()
        print('A CAV-JEPA Model (v2)')
        print('Init mode:', init_mode)
        print('Predictor: depth={}, dim={}'.format(predictor_depth, predictor_embed_dim))
        print('Momentum: {} -> {}'.format(momentum_start, momentum_end))
        print('Learnable Positional Embedding:', tr_pos)
        print('Normalize targets:', normalize_targets)

        # Store config
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.momentum = momentum_start
        self.momentum_start = momentum_start
        self.momentum_end = momentum_end
        self.init_mode = init_mode
        self.normalize_targets = normalize_targets
        self.init_std = init_std
        self.modality_specific_depth = modality_specific_depth

        # ========== Context Encoder (receives masked input) ==========
        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)

        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(
            self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)

        # Audio-specific blocks
        self.blocks_a = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        # Visual-specific blocks
        self.blocks_v = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        # Unified blocks
        self.blocks_u = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(12 - modality_specific_depth)
        ])

        # Normalization layers
        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        # ========== Target Encoder (EMA copy, receives unmasked input) ==========
        # Will be initialized as copy of context encoder
        self.target_patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.target_patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.target_patch_embed_a.num_patches = self.patch_embed_a.num_patches

        self.target_modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.target_modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.target_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=False)
        self.target_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=False)

        self.target_blocks_a = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        self.target_blocks_v = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        self.target_blocks_u = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(12 - modality_specific_depth)
        ])

        self.target_norm_a = norm_layer(embed_dim)
        self.target_norm_v = norm_layer(embed_dim)

        # ========== Predictor (lightweight, predicts target representations) ==========
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Learnable mask tokens (key I-JEPA component)
        # These replace zeros at masked positions - provides learnable queries for prediction
        self.mask_token_a = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.mask_token_v = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.predictor_pos_embed_a = nn.Parameter(
            torch.zeros(1, self.patch_embed_a.num_patches, predictor_embed_dim), requires_grad=False)
        self.predictor_pos_embed_v = nn.Parameter(
            torch.zeros(1, self.patch_embed_v.num_patches, predictor_embed_dim), requires_grad=False)

        self.predictor_blocks = nn.ModuleList([
            PredictorBlock(predictor_embed_dim, predictor_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(predictor_depth)
        ])
        self.predictor_depth = predictor_depth  # Store for depth-scaled init

        self.predictor_norm = norm_layer(predictor_embed_dim)

        # Project predictor output back to encoder dimension for loss computation
        self.predictor_proj_a = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        self.predictor_proj_v = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Initialize weights
        self.initialize_weights()

        # Copy context encoder to target encoder (no gradients for target)
        self._init_target_encoder()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def initialize_weights(self):
        """Initialize weights based on init_mode.

        init_mode='mae': Xavier uniform (compatible with MAE checkpoint loading)
        init_mode='random': Truncated normal with depth-scaled rescaling (I-JEPA style)
        """
        # Context encoder positional embeddings
        pos_embed_a = get_2d_sincos_pos_embed(
            self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches / 8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(
            self.pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches ** .5),
            int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        # Target encoder positional embeddings (same as context)
        self.target_pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))
        self.target_pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        # Predictor positional embeddings
        pred_pos_embed_a = get_2d_sincos_pos_embed(
            self.predictor_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches / 8), cls_token=False)
        self.predictor_pos_embed_a.data.copy_(torch.from_numpy(pred_pos_embed_a).float().unsqueeze(0))

        pred_pos_embed_v = get_2d_sincos_pos_embed(
            self.predictor_pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches ** .5),
            int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.predictor_pos_embed_v.data.copy_(torch.from_numpy(pred_pos_embed_v).float().unsqueeze(0))

        # Initialize patch embeddings
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize modality tokens
        trunc_normal_(self.modality_a, std=self.init_std)
        trunc_normal_(self.modality_v, std=self.init_std)

        # Initialize mask tokens (critical for predictor)
        trunc_normal_(self.mask_token_a, std=self.init_std)
        trunc_normal_(self.mask_token_v, std=self.init_std)

        # Initialize all linear layers and layer norms
        self.apply(self._init_weights)

        # Apply depth-scaled initialization for encoder and predictor blocks (I-JEPA style)
        if self.init_mode == 'random':
            self._apply_depth_scaled_init()

    def _apply_depth_scaled_init(self):
        """Apply depth-scaled rescaling to attention and MLP layers.

        Following I-JEPA: rescale by 1/sqrt(2*layer_id) for better gradient flow.
        This helps with training stability when starting from random init.
        """
        def rescale(param, layer_id):
            param.data.div_(math.sqrt(2.0 * layer_id))

        # Rescale audio encoder blocks
        for layer_id, blk in enumerate(self.blocks_a):
            rescale(blk.attn.proj.weight, layer_id + 1)
            rescale(blk.mlp.fc2.weight, layer_id + 1)

        # Rescale video encoder blocks
        for layer_id, blk in enumerate(self.blocks_v):
            rescale(blk.attn.proj.weight, layer_id + 1)
            rescale(blk.mlp.fc2.weight, layer_id + 1)

        # Rescale unified blocks (continue layer counting)
        base_layer = self.modality_specific_depth
        for layer_id, blk in enumerate(self.blocks_u):
            rescale(blk.attn.proj.weight, base_layer + layer_id + 1)
            rescale(blk.mlp.fc2.weight, base_layer + layer_id + 1)

        # Rescale predictor blocks
        for layer_id, blk in enumerate(self.predictor_blocks):
            rescale(blk.attn.proj.weight, layer_id + 1)
            rescale(blk.mlp.fc2.weight, layer_id + 1)

        print('Applied depth-scaled initialization (I-JEPA style)')

    def _init_weights(self, m):
        """Initialize weights based on init_mode."""
        if isinstance(m, nn.Linear):
            if self.init_mode == 'random':
                # I-JEPA style: truncated normal
                trunc_normal_(m.weight, std=self.init_std)
            else:
                # Original style: xavier uniform (compatible with MAE weights)
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.init_mode == 'random':
                trunc_normal_(m.weight, std=self.init_std)
            else:
                torch.nn.init.xavier_uniform_(m.weight.view([m.weight.shape[0], -1]))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_target_encoder(self):
        """Initialize target encoder as copy of context encoder (no gradients)"""
        # Copy patch embeddings
        self.target_patch_embed_a.load_state_dict(self.patch_embed_a.state_dict())
        self.target_patch_embed_v.load_state_dict(self.patch_embed_v.state_dict())

        # Copy modality tokens
        self.target_modality_a.data.copy_(self.modality_a.data)
        self.target_modality_v.data.copy_(self.modality_v.data)

        # Copy blocks
        for target_blk, ctx_blk in zip(self.target_blocks_a, self.blocks_a):
            target_blk.load_state_dict(ctx_blk.state_dict())
        for target_blk, ctx_blk in zip(self.target_blocks_v, self.blocks_v):
            target_blk.load_state_dict(ctx_blk.state_dict())
        for target_blk, ctx_blk in zip(self.target_blocks_u, self.blocks_u):
            target_blk.load_state_dict(ctx_blk.state_dict())

        # Copy norms
        self.target_norm_a.load_state_dict(self.norm_a.state_dict())
        self.target_norm_v.load_state_dict(self.norm_v.state_dict())

        # Freeze target encoder
        for param in self.target_parameters():
            param.requires_grad = False

    def target_parameters(self):
        """Iterator over target encoder parameters"""
        yield from self.target_patch_embed_a.parameters()
        yield from self.target_patch_embed_v.parameters()
        yield self.target_modality_a
        yield self.target_modality_v
        yield self.target_pos_embed_a
        yield self.target_pos_embed_v
        for blk in self.target_blocks_a:
            yield from blk.parameters()
        for blk in self.target_blocks_v:
            yield from blk.parameters()
        for blk in self.target_blocks_u:
            yield from blk.parameters()
        yield from self.target_norm_a.parameters()
        yield from self.target_norm_v.parameters()

    def context_parameters(self):
        """Iterator over context encoder parameters (for EMA source)"""
        yield from self.patch_embed_a.parameters()
        yield from self.patch_embed_v.parameters()
        yield self.modality_a
        yield self.modality_v
        yield self.pos_embed_a
        yield self.pos_embed_v
        for blk in self.blocks_a:
            yield from blk.parameters()
        for blk in self.blocks_v:
            yield from blk.parameters()
        for blk in self.blocks_u:
            yield from blk.parameters()
        yield from self.norm_a.parameters()
        yield from self.norm_v.parameters()

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder from context encoder.

        Reference I-JEPA implementation:
        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        This uses in-place operations for memory efficiency.
        """
        m = self.momentum
        for param_ctx, param_tgt in zip(self.context_parameters(), self.target_parameters()):
            param_tgt.data.mul_(m).add_((1. - m) * param_ctx.detach().data)

    def update_momentum(self, momentum):
        """Update the EMA momentum value"""
        self.momentum = momentum

    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence
        Returns: x_masked, mask, ids_restore
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_context_encoder(self, a, v, mask_ratio_a, mask_ratio_v):
        """Forward pass through context encoder with masking"""
        # Embed patches
        a = a.unsqueeze(1).transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # Apply masking
        a_masked, mask_a, ids_restore_a, ids_keep_a = self.random_masking_unstructured(a, mask_ratio_a)
        v_masked, mask_v, ids_restore_v, ids_keep_v = self.random_masking_unstructured(v, mask_ratio_v)

        # Modality-specific blocks
        for blk in self.blocks_a:
            a_masked = blk(a_masked)
        for blk in self.blocks_v:
            v_masked = blk(v_masked)

        # For contrastive: use modality-specific normalization on masked features
        a_contrast = a_masked
        for blk in self.blocks_u:
            a_contrast = blk(a_contrast, 'a')
        a_contrast = self.norm_a(a_contrast)

        v_contrast = v_masked
        for blk in self.blocks_u:
            v_contrast = blk(v_contrast, 'v')
        v_contrast = self.norm_v(v_contrast)

        return a_masked, v_masked, mask_a, mask_v, ids_restore_a, ids_restore_v, ids_keep_a, ids_keep_v, a_contrast, v_contrast

    @torch.no_grad()
    def forward_target_encoder(self, a, v):
        """Forward pass through target encoder (no masking, no gradients)

        Returns normalized target representations when normalize_targets=True.
        Normalization prevents representation collapse by focusing learning on
        representation direction rather than magnitude.
        """
        # Embed patches
        a = a.unsqueeze(1).transpose(2, 3)
        a = self.target_patch_embed_a(a)
        a = a + self.target_pos_embed_a
        a = a + self.target_modality_a

        v = self.target_patch_embed_v(v)
        v = v + self.target_pos_embed_v
        v = v + self.target_modality_v

        # Modality-specific blocks
        for blk in self.target_blocks_a:
            a = blk(a)
        for blk in self.target_blocks_v:
            v = blk(v)

        # Use modality-specific normalization in unified blocks
        for blk in self.target_blocks_u:
            a = blk(a, 'a')
        a = self.target_norm_a(a)

        for blk in self.target_blocks_u:
            v = blk(v, 'v')
        v = self.target_norm_v(v)

        # Apply target normalization to prevent collapse (I-JEPA key component)
        # This normalizes each target representation to have zero mean and unit variance
        if self.normalize_targets:
            a = F.layer_norm(a, (a.size(-1),))
            v = F.layer_norm(v, (v.size(-1),))

        return a, v

    def forward_predictor(self, a_ctx, v_ctx, mask_a, mask_v, ids_restore_a, ids_restore_v):
        """Forward pass through predictor to predict masked patch representations.

        Uses learnable mask tokens (I-JEPA style) instead of zeros for masked positions.
        This provides better gradient signal for masked position predictions.
        """
        N = a_ctx.shape[0]

        # Project context features to predictor dimension
        a_ctx = self.predictor_embed(a_ctx)
        v_ctx = self.predictor_embed(v_ctx)

        # Get number of masked patches
        num_mask_a = int(mask_a[0].sum())
        num_mask_v = int(mask_v[0].sum())

        # Calculate visible patches
        visible_a = self.patch_embed_a.num_patches - num_mask_a
        visible_v = self.patch_embed_v.num_patches - num_mask_v

        # Create full sequences initialized with LEARNABLE MASK TOKENS (not zeros!)
        # This is a key I-JEPA improvement - mask tokens provide learnable queries
        a_expanded = self.mask_token_a.expand(N, self.patch_embed_a.num_patches, -1).clone()
        v_expanded = self.mask_token_v.expand(N, self.patch_embed_v.num_patches, -1).clone()

        # Scatter context features at visible positions
        # ids_restore tells us where each token should go
        ids_keep_a = torch.argsort(ids_restore_a, dim=1)[:, :visible_a]
        a_expanded.scatter_(1, ids_keep_a.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim), a_ctx)

        ids_keep_v = torch.argsort(ids_restore_v, dim=1)[:, :visible_v]
        v_expanded.scatter_(1, ids_keep_v.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim), v_ctx)

        # Add positional embeddings
        a_expanded = a_expanded + self.predictor_pos_embed_a
        v_expanded = v_expanded + self.predictor_pos_embed_v

        # Concatenate audio and visual for joint prediction
        x = torch.cat([a_expanded, v_expanded], dim=1)

        # Apply predictor blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Split back to audio and visual
        pred_a = x[:, :self.patch_embed_a.num_patches, :]
        pred_v = x[:, self.patch_embed_a.num_patches:, :]

        # Project back to encoder dimension
        pred_a = self.predictor_proj_a(pred_a)
        pred_v = self.predictor_proj_v(pred_v)

        return pred_a, pred_v

    def forward_jepa_loss(self, pred, target, mask):
        """
        Compute JEPA loss following the reference I-JEPA implementation.
        Uses smooth_l1_loss (Huber loss) which is more robust to outliers.

        pred: [N, L, D] - predicted representations
        target: [N, L, D] - target representations (from target encoder)
        mask: [N, L] - binary mask (1 = masked/predict, 0 = visible)

        Reference: https://github.com/facebookresearch/ijepa
        """
        # Compute smooth L1 loss (Huber loss) - same as reference I-JEPA
        # This is more robust to outliers than MSE
        loss = F.smooth_l1_loss(pred, target, reduction='none')
        loss = loss.mean(dim=-1)  # [N, L], mean over embedding dim

        # Average over masked positions only
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
        """Compute contrastive loss between audio and visual representations"""
        audio_rep = F.normalize(audio_rep, dim=-1)
        video_rep = F.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, video_rep.t()) / 0.05

        if not bidirect_contrast:
            nce = -torch.mean(torch.diag(F.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(
                torch.argmax(F.softmax(total, dim=0), dim=0),
                torch.arange(0, total.shape[0], device=audio_rep.device)
            )) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(F.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(F.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(
                torch.argmax(F.softmax(total, dim=0), dim=0),
                torch.arange(0, total.shape[0], device=audio_rep.device)
            )) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(
                torch.argmax(F.softmax(total.t(), dim=0), dim=0),
                torch.arange(0, total.shape[0], device=audio_rep.device)
            )) / total.shape[0]
            return (nce_1 + nce_2) / 2, (c_acc_1 + c_acc_2) / 2

    def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75,
                jepa_loss_weight=1.0, contrast_loss_weight=0.01):
        """
        Forward pass for CAV-JEPA training.

        Args:
            audio: [N, T, F] audio spectrogram
            imgs: [N, C, H, W] visual frames
            mask_ratio_a: masking ratio for audio
            mask_ratio_v: masking ratio for visual
            jepa_loss_weight: weight for JEPA loss
            contrast_loss_weight: weight for contrastive loss

        Returns:
            loss: total loss
            loss_jepa: JEPA loss
            loss_jepa_a: JEPA loss for audio
            loss_jepa_v: JEPA loss for visual
            loss_c: contrastive loss
            mask_a: audio mask
            mask_v: visual mask
            c_acc: contrastive accuracy
        """
        # 1. Forward through target encoder (no masking, no gradients)
        target_a, target_v = self.forward_target_encoder(audio, imgs)

        # 2. Forward through context encoder (with masking)
        (a_ctx, v_ctx, mask_a, mask_v, ids_restore_a, ids_restore_v,
         ids_keep_a, ids_keep_v, a_contrast, v_contrast) = self.forward_context_encoder(
            audio, imgs, mask_ratio_a, mask_ratio_v)

        # 3. Forward through predictor
        pred_a, pred_v = self.forward_predictor(
            a_ctx, v_ctx, mask_a, mask_v, ids_restore_a, ids_restore_v)

        # 4. Compute JEPA loss (on masked positions)
        if jepa_loss_weight != 0:
            loss_jepa_a = self.forward_jepa_loss(pred_a, target_a, mask_a)
            loss_jepa_v = self.forward_jepa_loss(pred_v, target_v, mask_v)
            loss_jepa = jepa_loss_weight * (loss_jepa_a + loss_jepa_v)
        else:
            loss_jepa_a = torch.tensor(0.0, device=audio.device)
            loss_jepa_v = torch.tensor(0.0, device=audio.device)
            loss_jepa = torch.tensor(0.0, device=audio.device)

        # 5. Compute contrastive loss (on visible positions, mean pooled)
        if contrast_loss_weight != 0:
            loss_c, c_acc = self.forward_contrastive(
                a_contrast.mean(dim=1), v_contrast.mean(dim=1))
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c = torch.tensor(0.0, device=audio.device)
            c_acc = torch.tensor(0.0, device=audio.device)

        # 6. Total loss
        loss = loss_jepa + loss_c

        return loss, loss_jepa, loss_jepa_a, loss_jepa_v, loss_c, mask_a, mask_v, c_acc

    def forward_feat(self, a, v):
        """Extract features for retrieval (no masking)"""
        # Embed patches
        a = a.unsqueeze(1).transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # Modality-specific blocks
        for blk in self.blocks_a:
            a = blk(a)
        for blk in self.blocks_v:
            v = blk(v)

        # Unified blocks with modality-specific normalization
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)

        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_v(v)

        return a, v


# Fine-tuning model (same as CAVMAEFT but loads from JEPA checkpoint)
class CAVJEPAFT(nn.Module):
    """CAV-JEPA Fine-tuning model (no predictor/target encoder needed)"""

    def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, tr_pos=True):
        super().__init__()
        print('CAV-JEPA Fine-tuning Model')

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)

        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(
            self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, embed_dim), requires_grad=tr_pos)

        self.blocks_a = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        self.blocks_v = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(modality_specific_depth)
        ])
        self.blocks_u = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(12 - modality_specific_depth)
        ])

        self.norm_a = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        self.norm = norm_layer(embed_dim)

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(
            self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches / 8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(
            self.pos_embed_v.shape[-1],
            int(self.patch_embed_v.num_patches ** .5),
            int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))

        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, a, v, mode='multimodal'):
        if mode == 'multimodal':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_v:
                v = blk(v)

            x = torch.cat((a, v), dim=1)
            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)
            x = x.mean(dim=1)
            return self.mlp_head(x)

        elif mode == 'audioonly':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            return self.mlp_head(a.mean(dim=1))

        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v

            for blk in self.blocks_v:
                v = blk(v)
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            return self.mlp_head(v.mean(dim=1))

    def forward_feat(self, a, v, mode='av'):
        if mode == 'av':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_v:
                v = blk(v)

            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            return a, v

        elif mode == 'a':
            a = a.unsqueeze(1).transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            return a
