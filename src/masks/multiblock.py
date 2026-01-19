# -*- coding: utf-8 -*-
# @Time    : 1/19/26
# @Author  : Adapted from I-JEPA (Meta)
# @Description: Multiblock masking for audio-visual JEPA training
#
# Key differences from original I-JEPA:
# - Separate masks for audio and visual modalities
# - Audio patches have different aspect ratios (time x frequency)
# - Supports asymmetric masking between modalities

import math
from multiprocessing import Value
from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MultiBlockMaskCollator:
    """
    Multiblock mask generator following I-JEPA style.

    Generates structured block masks instead of random token masking.
    This forces the model to learn semantic-level representations
    rather than local interpolation patterns.

    Args:
        input_size: (height, width) of the patch grid
        patch_size: size of each patch (for reference, not used in mask generation)
        enc_mask_scale: (min, max) scale for context encoder mask (visible patches)
        pred_mask_scale: (min, max) scale for prediction targets
        aspect_ratio: (min, max) aspect ratio for block shapes
        nenc: number of context masks to generate
        npred: number of prediction target masks
        min_keep: minimum number of patches to keep
        allow_overlap: whether to allow overlap between context and prediction masks
    """

    def __init__(
        self,
        input_size=(14, 14),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False
    ):
        super().__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.patch_size = patch_size
        self.height, self.width = input_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

    def step(self):
        """Thread-safe iteration counter"""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        """Sample a random block size given scale and aspect ratio constraints"""
        _rand = torch.rand(1, generator=generator).item()

        # Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        # Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # Compute block height and width
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))

        # Clamp to grid dimensions
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        """
        Sample a block mask of given size.

        Args:
            b_size: (height, width) of the block
            acceptable_regions: list of masks indicating valid placement regions

        Returns:
            mask: indices of patches in the block
            mask_complement: binary mask where block region is 0
        """
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Restrict mask to acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False

        while not valid_mask:
            # Sample block top-left corner
            top = torch.randint(0, max(1, self.height - h), (1,))
            left = torch.randint(0, max(1, self.width - w), (1,))

            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1

            # Constrain to acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)

            mask = torch.nonzero(mask.flatten())

            # Check if mask is large enough
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Valid mask not found, relaxing constraints [{tries}]')

        mask = mask.squeeze()

        # Create complement mask
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0

        return mask, mask_complement

    def __call__(self, batch):
        """
        Generate encoder and predictor masks for a batch.

        Returns:
            collated_batch: the original batch collated
            collated_masks_enc: encoder (context) masks
            collated_masks_pred: predictor (target) masks
        """
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Sample block sizes
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio
        )
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.)  # Context mask uses square-ish blocks
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(B):
            # Generate prediction masks
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            # Set acceptable regions for context mask
            acceptable_regions = masks_C if not self.allow_overlap else None

            # Generate encoder masks
            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Truncate to minimum size for batching
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


class AudioVisualMaskCollator:
    """
    Specialized mask collator for audio-visual data.

    Generates separate masks for audio and visual modalities with
    modality-specific aspect ratios and scales.

    Audio patches: typically (8, 64) for 1024-frame input
    - 8 frequency bins x 64 time steps = 512 patches
    - Aspect ratio favors time-focused blocks

    Visual patches: typically (14, 14) for 224x224 input
    - 14 x 14 = 196 patches
    - Standard spatial blocks

    Args:
        audio_size: (freq_bins, time_steps) for audio patches
        visual_size: (height, width) for visual patches
        enc_mask_scale: (min, max) visible patch ratio for context
        pred_mask_scale: (min, max) masked patch ratio for prediction
        audio_aspect_ratio: (min, max) for audio blocks (time-focused)
        visual_aspect_ratio: (min, max) for visual blocks (square-ish)
        npred: number of prediction blocks per modality
        min_keep: minimum patches to keep in each mask
        allow_overlap: whether enc and pred masks can overlap
    """

    def __init__(
        self,
        audio_size=(8, 64),  # 512 patches: 8 freq x 64 time
        visual_size=(14, 14),  # 196 patches: 14 x 14
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        audio_aspect_ratio=(0.5, 2.0),  # Time-focused for audio
        visual_aspect_ratio=(0.75, 1.5),  # Square-ish for visual
        npred=4,
        min_keep=4,
        allow_overlap=False
    ):
        super().__init__()
        self.audio_size = audio_size
        self.visual_size = visual_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.audio_aspect_ratio = audio_aspect_ratio
        self.visual_aspect_ratio = visual_aspect_ratio
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)

        # Create separate collators for each modality
        self.audio_collator = MultiBlockMaskCollator(
            input_size=audio_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=audio_aspect_ratio,
            nenc=1,
            npred=npred,
            min_keep=min_keep,
            allow_overlap=allow_overlap
        )
        self.visual_collator = MultiBlockMaskCollator(
            input_size=visual_size,
            enc_mask_scale=enc_mask_scale,
            pred_mask_scale=pred_mask_scale,
            aspect_ratio=visual_aspect_ratio,
            nenc=1,
            npred=npred,
            min_keep=min_keep,
            allow_overlap=allow_overlap
        )

    def generate_masks(self, batch_size, device='cpu'):
        """
        Generate multiblock masks for a batch.

        Args:
            batch_size: number of samples
            device: device to place masks on

        Returns:
            audio_mask: binary mask for audio [B, num_audio_patches]
            visual_mask: binary mask for visual [B, num_visual_patches]
            audio_pred_indices: list of prediction target indices for audio
            visual_pred_indices: list of prediction target indices for visual
        """
        # Generate masks using the internal collators
        # We create dummy data to pass through
        dummy_audio = [torch.zeros(1) for _ in range(batch_size)]
        dummy_visual = [torch.zeros(1) for _ in range(batch_size)]

        _, audio_enc_masks, audio_pred_masks = self.audio_collator(dummy_audio)
        _, visual_enc_masks, visual_pred_masks = self.visual_collator(dummy_visual)

        # Convert index-based masks to binary masks
        num_audio = self.audio_size[0] * self.audio_size[1]
        num_visual = self.visual_size[0] * self.visual_size[1]

        audio_binary = self._indices_to_binary(audio_enc_masks, num_audio, batch_size)
        visual_binary = self._indices_to_binary(visual_enc_masks, num_visual, batch_size)

        return {
            'audio_enc_mask': audio_binary.to(device),
            'visual_enc_mask': visual_binary.to(device),
            'audio_pred_masks': audio_pred_masks,
            'visual_pred_masks': visual_pred_masks,
        }

    def _indices_to_binary(self, index_masks, num_patches, batch_size):
        """Convert index masks to binary masks (1 = keep, 0 = mask)"""
        # index_masks shape: [B, nenc, num_keep]
        binary = torch.zeros(batch_size, num_patches)

        for b in range(batch_size):
            for enc_mask in index_masks[b]:
                binary[b, enc_mask] = 1

        return binary

    def get_random_binary_masks(self, batch_size, mask_ratio_a=0.75, mask_ratio_v=0.75, device='cpu'):
        """
        Generate random binary masks (original CAV-JEPA style) for comparison.

        This provides a fallback to unstructured random masking.

        Args:
            batch_size: number of samples
            mask_ratio_a: audio masking ratio (0.75 = mask 75%)
            mask_ratio_v: visual masking ratio
            device: device to place masks on

        Returns:
            audio_mask: [B, num_audio_patches] binary (1 = keep, 0 = mask)
            visual_mask: [B, num_visual_patches] binary
        """
        num_audio = self.audio_size[0] * self.audio_size[1]
        num_visual = self.visual_size[0] * self.visual_size[1]

        keep_a = int(num_audio * (1 - mask_ratio_a))
        keep_v = int(num_visual * (1 - mask_ratio_v))

        # Generate random permutations
        audio_noise = torch.rand(batch_size, num_audio)
        visual_noise = torch.rand(batch_size, num_visual)

        # Sort and keep top-k
        audio_ids = torch.argsort(audio_noise, dim=1)
        visual_ids = torch.argsort(visual_noise, dim=1)

        audio_keep = audio_ids[:, :keep_a]
        visual_keep = visual_ids[:, :keep_v]

        # Convert to binary
        audio_mask = torch.zeros(batch_size, num_audio, device=device)
        visual_mask = torch.zeros(batch_size, num_visual, device=device)

        for b in range(batch_size):
            audio_mask[b, audio_keep[b]] = 1
            visual_mask[b, visual_keep[b]] = 1

        return audio_mask, visual_mask


def create_multiblock_masks_simple(
    batch_size,
    num_patches,
    grid_size,
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5),
    npred=4,
    device='cpu'
):
    """
    Simplified function to create multiblock masks.

    This is a simpler interface for cases where you don't need the full collator.

    Args:
        batch_size: number of samples
        num_patches: total number of patches
        grid_size: (height, width) of the patch grid
        enc_mask_scale: (min, max) for visible patch ratio
        pred_mask_scale: (min, max) for prediction target ratio
        aspect_ratio: (min, max) for block shapes
        npred: number of prediction blocks
        device: device to place masks on

    Returns:
        enc_mask: [B, num_patches] binary mask (1 = visible, 0 = masked)
        pred_masks: list of [B, pred_size] index tensors for prediction targets
    """
    height, width = grid_size

    # Sample scales
    enc_scale = enc_mask_scale[0] + torch.rand(1).item() * (enc_mask_scale[1] - enc_mask_scale[0])
    pred_scale = pred_mask_scale[0] + torch.rand(1).item() * (pred_mask_scale[1] - pred_mask_scale[0])

    # Sample aspect ratios
    ar = aspect_ratio[0] + torch.rand(1).item() * (aspect_ratio[1] - aspect_ratio[0])

    # Calculate block sizes
    enc_size = int(num_patches * enc_scale)
    pred_size = int(num_patches * pred_scale)

    enc_h = int(math.sqrt(enc_size * ar))
    enc_w = int(math.sqrt(enc_size / ar))
    enc_h = min(enc_h, height)
    enc_w = min(enc_w, width)

    pred_h = int(math.sqrt(pred_size * ar))
    pred_w = int(math.sqrt(pred_size / ar))
    pred_h = min(pred_h, height)
    pred_w = min(pred_w, width)

    enc_masks = []
    pred_masks = []

    for b in range(batch_size):
        # Sample encoder block position
        enc_top = torch.randint(0, max(1, height - enc_h), (1,)).item()
        enc_left = torch.randint(0, max(1, width - enc_w), (1,)).item()

        # Create encoder binary mask
        enc_mask = torch.zeros(height, width)
        enc_mask[enc_top:enc_top+enc_h, enc_left:enc_left+enc_w] = 1
        enc_masks.append(enc_mask.flatten())

        # Sample prediction blocks
        batch_pred = []
        for _ in range(npred):
            pred_top = torch.randint(0, max(1, height - pred_h), (1,)).item()
            pred_left = torch.randint(0, max(1, width - pred_w), (1,)).item()

            pred_mask = torch.zeros(height, width, dtype=torch.int32)
            pred_mask[pred_top:pred_top+pred_h, pred_left:pred_left+pred_w] = 1
            pred_indices = torch.nonzero(pred_mask.flatten()).squeeze(-1)
            batch_pred.append(pred_indices)
        pred_masks.append(batch_pred)

    enc_masks = torch.stack(enc_masks).to(device)

    return enc_masks, pred_masks
