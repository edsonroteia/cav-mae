# -*- coding: utf-8 -*-
# @Time    : 3/11/23 4:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : cav_mae.py

import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
from .modeling_pretrain import PretrainVisionTransformerEncoder, PretrainVisionTransformerDecoder
from .modeling_finetune import VisionTransformer
from .modeling_finetune import PatchEmbed as VMAEPatchEmbed
from .modeling_finetune import get_sinusoid_encoding_table

class PatchEmbed(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=384):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm1_a = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm2_a = norm_layer(dim)
        self.norm2_v = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, modality=None):
        if modality == None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif modality == 'a':
            x = x + self.drop_path(self.attn(self.norm1_a(x)))
            x = x + self.drop_path(self.mlp(self.norm2_a(x)))
        elif modality == 'v':
            x = x + self.drop_path(self.attn(self.norm1_v(x)))
            x = x + self.drop_path(self.mlp(self.norm2_v(x)))
        return x

# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class CAVMAEv2(nn.Module):
    """ CAV-MAEv2 Model
    """
    def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, num_frames=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False):
        super().__init__()
        print('A CAV-MAEv2 #3 Model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        # the encoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.num_frames = num_frames

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = VMAEPatchEmbed(img_size, patch_size, in_chans, int(embed_dim/2))

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, int(embed_dim/2)))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, int(embed_dim / 2)), requires_grad=tr_pos)  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        # self.blocks_v = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # self.blocks_v = PretrainVisionTransformerEncoder(embed_dim=int(embed_dim), num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, depth=modality_specific_depth, init_values=0., all_frames=8)
        self.blocks_v = VisionTransformer(embed_dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, depth=modality_specific_depth, init_values=0., all_frames=8)

        self.proj_v = nn.Linear(int(embed_dim/2), embed_dim, bias=True)

        # unified branch
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12-modality_specific_depth)])

        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)

        # the decoder part
        # Project to lower dimension for the decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        video_decoder_embed_dim = 192
        self.decoder_embed_video = nn.Linear(decoder_embed_dim, video_decoder_embed_dim, bias=True)

        # token used for masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, decoder_embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding

        self.decoder_blocks_audio = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(decoder_depth)])
        self.decoder_blocks_video = PretrainVisionTransformerDecoder(embed_dim=192, num_heads=3, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, depth=8, init_values=0.)

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # project channel is different for two modality, use two projection head
        self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch
        # self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans * 2, bias=True)  # decoder to patch 
#         self.decoder_pred_v = nn.ConvTranspose2d( ### TEMP FIX!!!   
#             in_channels=decoder_embed_dim,  # Input feature depth
#             out_channels=in_chans,          # Typically 3 for RGB images
#             kernel_size=patch_size,         # Size of the kernel
#             stride=patch_size,              # Stride: Upsample factor (often set equal to kernel size)
#             padding=0,                      # Padding
#             bias=True                       # Include a bias term
#         )
        # Gets output from "torch.Size([24, 784, 768])" to "torch.Size([24, 784 * 2, 768])""
        # self.decoder_head_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.intermediate_outputs = {}

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)

    def register_hooks(self, blocks, block_type):
        def hook_fn(m, i, o):
            self.intermediate_outputs[block_type + str(m)] = o
        for idx, block in enumerate(blocks):
            block.register_forward_hook(hook_fn)

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding, opt the cls token, add by myself
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        # pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        print("NUM PATCHES: ", self.patch_embed_v.num_patches)
        pos_embed_v = get_sinusoid_encoding_table(self.patch_embed_v.num_patches, self.pos_embed_v.shape[-1])
        # self.pos_embed_v.data.copy_(torch.from_numpy(pos_embed_v).float().unsqueeze(0))
        self.pos_embed_v.data.copy_(pos_embed_v.float())


        decoder_pos_embed_a = get_2d_sincos_pos_embed(self.decoder_pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.decoder_pos_embed_a.data.copy_(torch.from_numpy(decoder_pos_embed_a).float().unsqueeze(0))

        # decoder_pos_embed_v = get_2d_sincos_pos_embed(self.decoder_pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
        # self.decoder_pos_embed_v.data.copy_(torch.from_numpy(decoder_pos_embed_v).float().unsqueeze(0))
        decoder_pos_embed_v = get_sinusoid_encoding_table(self.patch_embed_v.num_patches, self.decoder_pos_embed_v.shape[-1])
        self.decoder_pos_embed_v.data.copy_(decoder_pos_embed_v.float())


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed_a.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.patch_embed_v.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        torch.nn.init.normal_(self.decoder_modality_a, std=.02)
        torch.nn.init.normal_(self.decoder_modality_v, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def patchify(self, imgs, c, h, w, p=16):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     import pdb ; pdb.set_trace()
    #     x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
    #     return x
    
    def patchify(self, imgs, c, t, h, w, p=16, m='a'):
        """
        Adapted for imgs with an extra dimension for time (or frames): (N, C, T, H, W).
        Args:
            imgs: input images of shape (N, C, T, H, W)
            c: number of channels
            t: number of time frames or similar (additional dimension)
            h: number of vertical patches (not pixels)
            w: number of horizontal patches (not pixels)
            p: size of each patch in pixels (assuming square patches for simplicity)

        Returns:
            Patchified images of shape (N, T * H * W, P^2 * C)
        """
        # Reshape to separate the patches, now correctly accounting for the number of patches and patch size
        # Note: imgs originally of shape (N, C, T, H*P, W*P) assuming H, W are in patches now
        # We reshape considering the real height and width in pixels are H*p and W*p
        x = imgs.reshape(shape=(imgs.shape[0], c, t, h, p, w, p))
        # Rearrange the patches to put all patches from all frames into the batch dimension
        x = torch.einsum('ncthpwq->nthwpqc', x)
        # Combine the time, patch height, and patch width dimensions into a single dimension,
        if m == 'a':
            # preserving the batch dimension separate. Now each patch's contents are flattened.
            x = x.reshape(shape=(imgs.shape[0], t * h * w, p ** 2 * c))
        else:
            x = x.reshape(shape=(imgs.shape[0], int(t / 2) * h * w, 2 * p ** 2 * c))
        return x



    def unpatchify(self, x, c, t, h, w, p=16, m='a'):
        """
        Adapted to reassemble images that include a time dimension.
        Args:
            x: Tensor with shape (N, T * H * W, P^2 * C), output from the patchify function
            c: Number of channels
            t: Number of time frames
            h: Number of vertical patches
            w: Number of horizontal patches
            p: Size of each patch in pixels

        Returns:
            Reassembled images of shape (N, C, T, H * P, W * P)
        """
        # Reshape to disentangle time, height, and width of patches
        x = x.reshape(shape=(x.shape[0], t, h, w, p, p, c))
        # Rearrange to restore the original order of dimensions: batch, channel, time, height, width
        x = torch.einsum('nthwpqc->ncthpwq', x)
        # Combine the patch height and patch width back into single height and width dimensions
        if m == 'a':
            # Now the patches are combined into the original image shape
            x = x.reshape(shape=(x.shape[0], c, t, h * p, w * p))
        else:
            x = x.reshape(shape=(x.shape[0], c, t, h * p, w * p))
        return x
    
    def random_masking_unstructured(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        assert L == f * t
        noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
        if mode == 'time':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
        elif mode == 'freq':
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        elif mode == 'tf':
            for i in range(N):
                mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
                for k in mask_t_list:
                    noise[i, :, k] = 1.1  # large value will be removed
            for i in range(N):
                mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
                for k in mask_f_list:
                    noise[i, k, :] = 1.1  # large value will be removed
        noise = noise.reshape(N, L)

        # sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, a, v, mask_ratio_a, mask_ratio_v, mask_mode='unstructured'):
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        # print("Shape of audio and visual: ", a.shape, v.shape)


        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # print("Successfully embedded audio and visual")
        # print("Shape of audio and visual: ", a.shape, v.shape)
        # print("Shape of pos embed: ", self.pos_embed_a.shape, self.pos_embed_v.shape)

        # by default, we always use unstructured masking
        if mask_mode == 'unstructured':
            a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
        # in ablation study, we tried time/freq/tf masking. mode in ['freq', 'time', 'tf']
        else:
            a, mask_a, ids_restore_a = self.random_masking_structured(a, mask_ratio_a, t=64, f=8, mode=mask_mode)

        # visual branch always use unstructured masking
        v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)

        # audio and visual stream, independent blocks
        for blk in self.blocks_a:
            a = blk(a)

        # for blk in self.blocks_v:
        #     v = blk(v)
        v = self.blocks_v(v)

        # print("Shape of audio and visual after blocks: ", a.shape, v.shape)
        # print("Shape of mask a and mask v: ", mask_a.shape, mask_v.shape)
        
        v = self.proj_v(v)

        # concatenate audio and visual tokens
        x = torch.cat((a, v), dim=1)

        # unified stream, shared blocks_u, but independent normalization layers
        for blk in self.blocks_u:
            x = blk(x)
        x = self.norm(x)

        for blk in self.blocks_u: 
            ca = blk(a, 'a')
        ca = self.norm_a(ca)

        for blk in self.blocks_u:
            cv = blk(v, 'v')
        cv = self.norm_v(cv)

        return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv

    def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):
        # print("Shape of x (latent) at decoder: ", x.shape)
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
        mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)
        a_ = torch.cat([x[:, :self.patch_embed_a.num_patches-int(mask_a[0].sum()), :], mask_tokens_a], dim=1)  # no cls token
        a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # similar for the visual modality
        mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
        v_ = torch.cat([x[:, self.patch_embed_a.num_patches-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1)  # no cls token
        v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x_a = a_ + self.decoder_pos_embed_a
        x_v = v_ + self.decoder_pos_embed_v

        # # concatenate audio and visual tokens
        # x = torch.cat([a_, v_], dim=1)

        # decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
        # x = x + decoder_pos_embed

        # # add modality indication tokens
        # x[:, 0:self.patch_embed_a.num_patches, :] = x[:, 0:self.patch_embed_a.num_patches, :] + self.decoder_modality_a
        # x[:, self.patch_embed_a.num_patches:, :] = x[:, self.patch_embed_a.num_patches:, :] + self.decoder_modality_v

        # x_a = x[:, :self.patch_embed_a.num_patches, :]
        # x_v = x[:, self.patch_embed_a.num_patches:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks_audio:
            x_a = blk(x_a)
        x_a = self.decoder_norm(x_a)

        # for blk in self.decoder_blocks_video:
        #     x_v = blk(x_v)
        x_v = self.decoder_embed_video(x_v)
        x_v = self.decoder_blocks_video(x_v, 0)

        # predictor projection
        x_a = self.decoder_pred_a(x_a)
        # print("DEBUG: Shape of x: ", x.shape)
        # print("DEBUG: Number of audio patches: ", self.patch_embed_a.num_patches)
        # x_v_reshaped = x[:, self.patch_embed_a.num_patches:, :].transpose(1, 2).view(x.shape[0], -1, H, W)  # Adjust H and W based on expected output size
        # x_v = self.decoder_pred_v(x_v_reshaped)
        # original_x_v_shape = (x.shape[0], x.shape[1] - self.patch_embed_a.num_patches, x.shape[2])
        # x_v = self.decoder_pred_v(x_v)
        # x_v = x_v.reshape(x_v.shape[0], original_x_v_shape[1] * 2, -1)
        # return audio and video tokens
        return x_a, x_v

    def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
        # calculate nce loss for mean-visual representation and mean-audio representation

        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

        total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

        # by default we use single directional
        if bidirect_contrast == False:
            nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            return nce, c_acc
        else:
            nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
            nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
            c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
            nce = (nce_1 + nce_2) / 2
            c_acc = (c_acc_1 + c_acc_2) / 2
            return nce, c_acc

    def forward_mae_loss(self, input, pred, mask, modality):
        if modality == 'a':
            # For audio, adjust the shape as needed
            input = input.unsqueeze(1)
            input = input.transpose(2, 3)
            target = self.patchify(input, 1, 1, int(input.shape[2]/self.patch_embed_a.patch_size[0]), int(input.shape[3]/self.patch_embed_a.patch_size[1]), 16, 'a')
        elif modality == 'v':
            # print("Input shape: ", input.shape)
            # Now correctly handle the additional time (or frames) dimension for video
            target = self.patchify(input, 3, input.shape[2], int(input.shape[3]/self.patch_embed_v.patch_size[0]), int(input.shape[4]/self.patch_embed_v.patch_size[1]), 16, 'v')
        
        # print("Target shape after patchify: ", target.shape)
        # print("Pred shape: ", pred.shape)

        # patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # print("LAST DEBUG?", loss.shape, mask.shape)
        # if modality == 'v':
        #     mask = mask.repeat(1,2)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured'):
        # print("Shape before forward: ", audio.shape, imgs.shape)
        # latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        # print("Latent shape: ", latent.shape)
        # print("Mask a shape: ", mask_a.shape)
        # print("Mask v shape: ", mask_v.shape)
        # print("Ids restore a shape: ", ids_restore_a.shape)
        # print("Ids restore v shape: ", ids_restore_v.shape)
        # if mae loss is used
        if mae_loss_weight != 0:
            pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)
            # print("Pred a shape (after decoder): ", pred_a.shape)
            # print("Pred v shape (after decoder): ", pred_v.shape)
            # print("audio shape: ", audio.shape)
            # print("imgs shape: ", imgs.shape)
            loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
            loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
            loss_mae = mae_loss_weight * (loss_mae_a + loss_mae_v)
        else:
            loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)
            pred_a = None
            pred_v = None

        # if contrastive loss is used
        if contrast_loss_weight != 0:
            # note this is single directional
            loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1))
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

        loss = loss_mae + loss_c

        recon_a = self.unpatchify(pred_a, 1, 1, 8, 64, 16)
        recon_a = torch.einsum('ntchw->nthwc', recon_a)
        recon_v = self.unpatchify(pred_v, 3, 8, 14, 14, 16)
        recon_v = torch.einsum('ntchw->nthwc', recon_v)

        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, recon_a, recon_v

    # used only for inpainting, ignore if inpainting is not of interest
    def forward_inpaint(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mask_mode='unstructured'):
        latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)
        pred_a, pred_v = self.forward_decoder(latent, mask_a, ids_restore_a, mask_v, ids_restore_v)  # [N, L, p*p*3]
        loss_pixel_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
        loss_pixel_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
        return pred_a, pred_v, mask_a, mask_v, loss_pixel_a, loss_pixel_v

    # used for retrieval, ignore if retrieval is not of interest
    def forward_feat(self, a, v, register_hook=False):
        if register_hook:
            self.register_hooks(self.blocks_a, "blocks_a_")
            self.register_hooks(self.blocks_v.blocks, "blocks_v_")
            self.register_hooks(self.blocks_u, "blocks_u_")
        # embed patches
        a = a.unsqueeze(1)
        a = a.transpose(2, 3)
        a = self.patch_embed_a(a)
        a = a + self.pos_embed_a
        a = a + self.modality_a

        v = self.patch_embed_v(v)
        v = v + self.pos_embed_v
        v = v + self.modality_v

        # the modality-specific stream
        for blk in self.blocks_a:
            a = blk(a)

        for blk in self.blocks_v.blocks:
            v = blk(v)

        # use modality specific normalization,
        for blk in self.blocks_u:
            a = blk(a, 'a')
        a = self.norm_a(a)

        for blk in self.blocks_u:
            v = blk(v, 'v')
        v = self.norm_v(v)
        return a, v

# the finetuned CAV-MAE model
class CAVMAEv2FT(nn.Module):
    def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
                 embed_dim=768, modality_specific_depth=11, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, num_frames=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False):
        super().__init__()
        print('A CAV-MAEv2 #3 Model')
        print('Use norm_pix_loss: ', norm_pix_loss)
        print('Learnable Positional Embedding: ', tr_pos)

        # the encoder part
        # overide the timm package
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        timm.models.vision_transformer.Block = Block

        self.num_frames = num_frames

        self.patch_embed_a = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.patch_embed_v = VMAEPatchEmbed(img_size, patch_size, in_chans, int(embed_dim/2))

        self.patch_embed_a.num_patches = int(audio_length * 128 / 256)
        print('Number of Audio Patches: {:d}, Visual Patches: {:d}'.format(self.patch_embed_a.num_patches, self.patch_embed_v.num_patches))

        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, int(embed_dim/2)))

        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.patch_embed_a.num_patches, embed_dim), requires_grad=tr_pos)  # fixed sin-cos embedding
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.patch_embed_v.num_patches, int(embed_dim / 2)), requires_grad=tr_pos)  # fixed sin-cos embedding

        # audio-branch
        self.blocks_a = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(modality_specific_depth)])
        # visual-branch
        self.blocks_v = VisionTransformer(embed_dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=norm_layer, depth=modality_specific_depth, init_values=0., all_frames=8)

        self.proj_v = nn.Linear(int(embed_dim/2), embed_dim, bias=True)

        # unified branch
        self.blocks_u = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(12-modality_specific_depth)])

        # independent normalization layer for audio, visual, and audio-visual
        self.norm_a, self.norm_v, self.norm = norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)

        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))

        self.initialize_weights()

        print('Audio Positional Embedding Shape:', self.pos_embed_a.shape)
        print('Visual Positional Embedding Shape:', self.pos_embed_v.shape)
        

    def get_patch_num(self, input_shape, stride):
        test_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        test_proj = torch.nn.Conv2d(1, 4, kernel_size=(16, 16), stride=(stride, stride))
        test_output = test_proj(test_input)
        return test_output.shape[2], test_output[3], test_output[2] * test_output[2]

    def initialize_weights(self):
        pos_embed_a = get_2d_sincos_pos_embed(self.pos_embed_a.shape[-1], 8, int(self.patch_embed_a.num_patches/8), cls_token=False)
        self.pos_embed_a.data.copy_(torch.from_numpy(pos_embed_a).float().unsqueeze(0))

        pos_embed_v = get_2d_sincos_pos_embed(self.pos_embed_v.shape[-1], int(self.patch_embed_v.num_patches ** .5), int(self.patch_embed_v.num_patches ** .5), cls_token=False)
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
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, a, v, mode):
        # multi-modal fine-tuning, our default method for fine-tuning
        if mode == 'multimodal':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            # for blk in self.blocks_v:
            #     v = blk(v)
                
            v = self.blocks_v(v)
            v = self.proj_v(v)

            x = torch.cat((a, v), dim=1)

            for blk in self.blocks_u:
                x = blk(x)
            x = self.norm(x)

            x = x.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only audio (and inference with only audio when the model is finetuned with only audio)
        elif mode == 'audioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # note here uses the 'a' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)
            x = a.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # finetune with only image (and inference with only audio when the model is finetuned with only image)
        elif mode == 'videoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            # for blk in self.blocks_v:
            #     v = blk(v)

            v = self.blocks_v(v)
            v = self.proj_v(v)

            # note here uses the 'v' normalization, it is used in both training and inference, so it is fine
            for blk in self.blocks_u:
                v = blk(v, 'v')
            v = self.norm_v(v)
            x = v.mean(dim=1)
            x = self.mlp_head(x)
            return x

        # used in case that the model is finetuned with both modality, but in inference only audio is given
        elif mode == 'missingaudioonly':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = a
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                a = blk(a, 'a') # note here use modality-specific normalization
            a = self.norm_a(a)
            a = a.mean(dim=1)

            # average the output of the two forward passes
            x = (u + a) / 2
            x = self.mlp_head(x)
            return x

        # used in case that the model is fine-tuned with both modality, but in inference only image is given
        elif mode == 'missingvideoonly':
            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            # for blk in self.blocks_v:
            #     v = blk(v)

            v = self.blocks_v(v)
            v = self.proj_v(v)

            # two forward passes to the block_u, one with modality-specific normalization, another with unified normalization
            u = v
            for blk in self.blocks_u:
                u = blk(u) # note here use unified normalization
            u = self.norm(u)
            u = u.mean(dim=1)

            for blk in self.blocks_u:
                v = blk(v, 'v') # note here use modality-specific normalization
            v = self.norm_v(v)
            v = v.mean(dim=1)

            # average the output of the two forward passes
            x = (u + v) / 2
            x = self.mlp_head(x)
            return x

    # for retrieval
    def forward_feat(self, a, v, mode='av', register_hook=False):
        if register_hook:
            self.register_hooks(self.blocks_a, "blocks_a_")
            self.register_hooks(self.blocks_v, "blocks_v_")
            self.register_hooks(self.blocks_u, "blocks_u_")

        # return both audio and visual
        if mode == 'av':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            v = self.patch_embed_v(v)
            v = v + self.pos_embed_v
            v = v + self.modality_v

            for blk in self.blocks_a:
                a = blk(a)

            # for blk in self.blocks_v:
            #     v = blk(v)

            v = self.blocks_v(v)
            v = self.proj_v(v)

            for blk in self.blocks_u:
                a = blk(a, 'a')
            a = self.norm_a(a)

            for blk in self.blocks_u:
                v = blk(v, 'v')

            v = self.norm_v(v)
            return a, v

        # return only audio
        if mode == 'a':
            a = a.unsqueeze(1)
            a = a.transpose(2, 3)
            a = self.patch_embed_a(a)
            a = a + self.pos_embed_a
            a = a + self.modality_a

            for blk in self.blocks_a:
                a = blk(a)

            for blk in self.blocks_u:
                a = blk(a, 'a')

            a = self.norm_a(a)
            return a
