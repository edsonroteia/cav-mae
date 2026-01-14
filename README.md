# CAV-JEPA: Contrastive Audio-Visual Joint-Embedding Predictive Architecture

> **Branch**: `cav-mae-jepa`
>
> This branch replaces the MAE objective with JEPA while keeping contrastive learning.

## Overview

Replace the MAE (Masked Autoencoder) objective in CAV-MAE with JEPA (Joint-Embedding Predictive Architecture) while keeping the contrastive learning objective. Similar to how CAV-MAE initializes from ImageMAE checkpoints, CAV-JEPA will initialize from I-JEPA/V-JEPA/A-JEPA checkpoints.

## Background: MAE vs JEPA

| Aspect | MAE (CAV-MAE) | JEPA (CAV-JEPA) |
|--------|---------------|-----------------|
| **Target** | Reconstruct raw pixels | Predict latent representations |
| **Decoder** | Full Transformer decoder | Lightweight predictor |
| **Loss** | MSE in pixel space | MSE/cosine in latent space |
| **Target encoder** | None | EMA of context encoder |
| **Mask tokens** | Learnable, appended to sequence | Not needed |

## Architecture Design

### Current CAV-MAE Architecture
```
Audio Input → Patch Embed → [Mask 75%] → Encoder (11 blocks) ──┐
                                                                ├→ Unified Block → Decoder → Pixel Reconstruction (MAE)
Visual Input → Patch Embed → [Mask 75%] → Encoder (11 blocks) ─┘                    ↓
                                                                              Contrastive Loss
```

### Proposed CAV-JEPA Architecture
```
Audio Input → Patch Embed → [Mask 75%] → Context Encoder (11 blocks) ──┐
                                                                        ├→ Unified Block → Predictor → Latent Prediction
Visual Input → Patch Embed → [Mask 75%] → Context Encoder (11 blocks) ─┘        ↑              ↓
                                                                                 │        JEPA Loss (MSE in latent space)
Audio Input → Patch Embed → Target Encoder (EMA) ───────────────────────────────┘              ↓
Visual Input → Patch Embed → Target Encoder (EMA) ─────────────────────────────────→ Target Representations
                                                                                               ↓
                                                                                      Contrastive Loss
```

## Key Components

### 1. Target Encoder (EMA)
- Copy of context encoder updated via exponential moving average
- `target_params = momentum * target_params + (1 - momentum) * context_params`
- Momentum schedule: 0.996 → 0.999 (increases during training)
- **No gradients** flow through target encoder

### 2. Predictor Network
Replace the heavy decoder with a lightweight predictor:
- Small Transformer (2-4 blocks) or MLP
- Input: Context encoder output + positional embeddings for masked positions
- Output: Predicted representations for masked patches
- Much smaller than MAE decoder (512 dim, 8 blocks → 384 dim, 2-4 blocks)

### 3. JEPA Loss
```python
def forward_jepa_loss(self, pred, target, mask):
    # Normalize representations
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # MSE loss on masked positions only
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

### 4. Combined Training Objective
```
Total Loss = jepa_loss_weight * (jepa_loss_a + jepa_loss_v) + contrast_loss_weight * contrastive_loss
```

## Initialization Strategy

The architecture supports multiple initialization options via command-line arguments:

### Option A: I-JEPA + A-JEPA (Recommended)
- **Visual encoder**: I-JEPA ViT-B/16 checkpoint (image-based)
- **Audio encoder**: A-JEPA checkpoint (audio-specific)
- Best of both worlds: modality-specific pretraining
- Requires A-JEPA weights to be available

### Option B: I-JEPA for Both
- Use I-JEPA ViT-B/16 for both audio and visual encoders
- Similar to how CAV-MAE uses ImageMAE for both modalities
- Adapt audio patch embedding (sum RGB channels → 1 channel)
- Fallback if A-JEPA weights not available

### Option C: V-JEPA for Both
- Use V-JEPA checkpoint (video-based, temporal understanding)
- May be better for video datasets
- Same adaptation process as Option B

### Option D: V-JEPA + A-JEPA (Hybrid)
- V-JEPA for visual stream (designed for video/images)
- A-JEPA for audio stream
- Alternative if V-JEPA performs better on visual tasks

## Implementation Plan

### Phase 1: Model Architecture Changes

#### 1.1 Create `src/models/cav_jepa.py`
New model class `CAVJEPA` with:
- Context encoder (same as CAV-MAE encoder)
- Target encoder (EMA copy)
- Lightweight predictor (replaces decoder)
- JEPA loss function
- Contrastive loss (unchanged)

#### 1.2 Key Methods to Implement
```python
class CAVJEPA(nn.Module):
    def __init__(self, ...):
        # Encoders
        self.context_encoder = ...  # Same as CAV-MAE encoder
        self.target_encoder = ...   # EMA copy (no gradients)

        # Predictor (replaces decoder)
        self.predictor = ...  # Lightweight Transformer/MLP

        # Momentum schedule
        self.momentum = 0.996

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update of target encoder"""
        for param_q, param_k in zip(self.context_encoder.parameters(),
                                     self.target_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    def forward_jepa_loss(self, pred, target, mask):
        """JEPA loss in latent space"""
        ...

    def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, ...):
        # 1. Get target representations (no grad)
        with torch.no_grad():
            target_a, target_v = self.target_encoder(audio, imgs)

        # 2. Get context representations (with masking)
        context, mask_a, mask_v = self.context_encoder(audio, imgs, mask_ratio_a, mask_ratio_v)

        # 3. Predict masked representations
        pred_a, pred_v = self.predictor(context, mask_a, mask_v)

        # 4. Compute JEPA loss
        loss_jepa_a = self.forward_jepa_loss(pred_a, target_a, mask_a)
        loss_jepa_v = self.forward_jepa_loss(pred_v, target_v, mask_v)

        # 5. Compute contrastive loss (unchanged)
        loss_c, c_acc = self.forward_contrastive(...)

        return loss_jepa + loss_c, ...
```

### Phase 2: Weight Adaptation

#### 2.1 Create `src/adapt_jepa_weights.py`
Flexible weight adaptation supporting multiple JEPA checkpoints:

```bash
# Usage examples:
python adapt_jepa_weights.py --visual_ckpt ijepa_vit_b16.pth --audio_ckpt ajepa_vit_b16.pth  # Option A
python adapt_jepa_weights.py --visual_ckpt ijepa_vit_b16.pth --audio_ckpt ijepa_vit_b16.pth  # Option B
python adapt_jepa_weights.py --visual_ckpt vjepa_vit_b16.pth --audio_ckpt vjepa_vit_b16.pth  # Option C
python adapt_jepa_weights.py --visual_ckpt vjepa_vit_b16.pth --audio_ckpt ajepa_vit_b16.pth  # Option D
```

Key operations:
- Load separate checkpoints for audio and visual streams
- Duplicate encoder blocks for modality-specific streams
- Initialize target encoder as copy of context encoder
- Adapt audio patch embedding (sum RGB channels → 1 channel) if using I-JEPA/V-JEPA
- Initialize predictor from JEPA predictor if available, else random

### Phase 3: Training Infrastructure

#### 3.1 Create `src/traintest_cavjepa.py`
Training loop with:
- EMA update after each step
- Momentum schedule (0.996 → 0.999)
- Logging for JEPA loss components

#### 3.2 Create `src/run_cavjepa_pretrain.py`
Entry point with arguments:
- `--jepa_loss_weight` (default: 1.0)
- `--contrast_loss_weight` (default: 0.01)
- `--momentum_start` (default: 0.996)
- `--momentum_end` (default: 0.999)
- `--predictor_depth` (default: 4)

#### 3.3 Create training scripts
- `egs/vggsound/run_cavjepa_pretrain.sh`
- `egs/audioset/run_cavjepa_pretrain.sh`

### Phase 4: Evaluation

Use existing finetuning pipeline (`run_cavmae_ft.py`) with CAV-JEPA checkpoints.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/models/cav_jepa.py` | Create | New CAVJEPA model class |
| `src/adapt_jepa_weights.py` | Create | Weight adaptation from I-JEPA/V-JEPA/A-JEPA |
| `src/traintest_cavjepa.py` | Create | Training loop with EMA |
| `src/run_cavjepa_pretrain.py` | Create | Pretraining entry point |
| `egs/vggsound/run_cavjepa_pretrain.sh` | Create | VGGSound training script |
| `src/models/__init__.py` | Modify | Export CAVJEPA class |

## Hyperparameters

Based on I-JEPA and V-JEPA papers:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Mask ratio | 0.75-0.90 | JEPA often uses higher masking |
| Predictor depth | 4 blocks | Lightweight |
| Predictor dim | 384 | Smaller than encoder |
| Momentum start | 0.996 | EMA momentum |
| Momentum end | 0.999 | Increases during training |
| JEPA loss weight | 1.0 | |
| Contrastive loss weight | 0.01 | Same as CAV-MAE |

## Expected Benefits

1. **Better representations**: JEPA learns more semantic features than pixel reconstruction
2. **Efficiency**: Smaller predictor vs. full decoder
3. **Transfer learning**: I-JEPA/V-JEPA/A-JEPA checkpoints are strong starting points
4. **Multimodal synergy**: Contrastive + JEPA may complement each other

## References

- [I-JEPA (CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA (Meta AI, 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [A-JEPA (Audio JEPA)](https://arxiv.org/abs/2311.15830)
- [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985)
- [CAV-MAE (ICLR 2023)](https://openreview.net/forum?id=QPtMRyk5rb)
- [I-JEPA GitHub](https://github.com/facebookresearch/ijepa)
- [V-JEPA GitHub](https://github.com/facebookresearch/jepa)
