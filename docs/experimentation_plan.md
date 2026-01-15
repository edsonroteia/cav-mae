# CAV-JEPA Experimentation Plan

## Research Question

Can JEPA (Joint-Embedding Predictive Architecture) replace the MAE objective in CAV-MAE while maintaining or improving audio-visual representation quality?

---

## 1. Architecture Comparison: MAE vs JEPA

| Aspect | CAV-MAE | CAV-JEPA |
|--------|---------|----------|
| **Target** | Reconstruct raw pixels | Predict latent representations |
| **Decoder** | Full Transformer decoder (8 blocks, 512 dim) | Lightweight predictor (2-4 blocks, 384 dim) |
| **Loss** | MSE in pixel space | MSE/cosine in latent space |
| **Target encoder** | None | EMA of context encoder |
| **Mask tokens** | Learnable, appended to sequence | Not needed |
| **Efficiency** | Heavy decoder computation | Lighter prediction head |

### Key Architectural Changes
```
CAV-MAE:
Audio/Visual → Encoder → [Mask Tokens] → Decoder → Pixel Reconstruction

CAV-JEPA:
Audio/Visual → Context Encoder ────────────────────→ Predictor → Latent Prediction
                     ↓ (EMA copy)                           ↓
Audio/Visual → Target Encoder ───────────────────→ Target Representations
                                                           ↓
                                                    JEPA Loss (latent MSE)
```

---

## 2. Initialization Strategy

### Available Checkpoints

| Source | Model | Modality | Link |
|--------|-------|----------|------|
| I-JEPA | ViT-B/16, ViT-L/16, ViT-H/14 | Images | [GitHub](https://github.com/facebookresearch/ijepa) |
| V-JEPA | ViT-B/16, ViT-L/16 | Video | [GitHub](https://github.com/facebookresearch/jepa) |
| A-JEPA | ViT-B | Audio | [arXiv](https://arxiv.org/abs/2311.15830) |

### Initialization Options

| Option | Visual | Audio | Pros | Cons |
|--------|--------|-------|------|------|
| **A** | I-JEPA | A-JEPA | Modality-specific pretraining | Requires A-JEPA weights |
| **B** | I-JEPA | I-JEPA | Consistent architecture | Audio uses image model |
| **C** | V-JEPA | V-JEPA | Temporal understanding | May overfit to video patterns |
| **D** | V-JEPA | A-JEPA | Best of both | Complex setup |

### Weight Adaptation Process
1. Load JEPA checkpoint(s)
2. Duplicate encoder blocks for audio/visual streams
3. Adapt audio patch embedding (RGB → 1 channel if using I-JEPA/V-JEPA)
4. Initialize target encoder as copy of context encoder
5. Initialize predictor (from JEPA predictor if available, else random)

---

## 3. Implementation Phases

### Phase 1: Model Architecture
**Files to create:**
- `src/models/cav_jepa.py` - CAVJEPA model class

**Key components:**
```python
class CAVJEPA(nn.Module):
    def __init__(self):
        self.context_encoder = ...  # Audio-visual encoder
        self.target_encoder = ...   # EMA copy (no gradients)
        self.predictor = ...        # Lightweight Transformer
        self.momentum = 0.996       # EMA momentum

    @torch.no_grad()
    def update_target_encoder(self):
        """EMA update after each step"""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(),
                                 self.target_encoder.parameters()):
            p_tgt.data = self.momentum * p_tgt.data + (1 - self.momentum) * p_ctx.data

    def forward_jepa_loss(self, pred, target, mask):
        """JEPA loss in normalized latent space"""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum()
```

### Phase 2: Weight Adaptation
**Files to create:**
- `src/adapt_jepa_weights.py` - Checkpoint conversion

**Usage:**
```bash
python src/adapt_jepa_weights.py \
    --visual_ckpt ijepa_vit_b16.pth \
    --audio_ckpt ajepa_vit_b16.pth \
    --output cav_jepa_init.pth
```

### Phase 3: Training Infrastructure
**Files to create:**
- `src/traintest_cavjepa.py` - Training loop with EMA
- `src/run_cavjepa_pretrain.py` - Entry point

**Key training additions:**
- EMA update after each optimizer step
- Momentum schedule: 0.996 → 0.999 over training
- Logging: JEPA loss (audio/visual), contrastive loss/acc

### Phase 4: Training Scripts
**Files to create:**
- `egs/audioset/run_cavjepa_pretrain.sh`
- `egs/vggsound/run_cavjepa_pretrain.sh`

---

## 4. Hyperparameter Configuration

### Fixed Parameters (from CAV-MAE)
| Parameter | Value |
|-----------|-------|
| Masking ratio | 0.75 |
| Learning rate | 1e-4 |
| Batch size | 120 |
| Epochs | 25 |
| Optimizer | AdamW |
| Contrastive loss weight | 0.01 |

### JEPA-Specific Parameters (to tune)
| Parameter | Default | Sweep Range | Notes |
|-----------|---------|-------------|-------|
| Momentum start | 0.996 | {0.99, 0.996, 0.999} | EMA starting momentum |
| Momentum end | 0.999 | {0.999, 0.9999} | EMA final momentum |
| Predictor depth | 4 | {2, 4, 6} | Number of Transformer blocks |
| Predictor dim | 384 | {256, 384, 512} | Hidden dimension |
| JEPA loss weight | 1.0 | {0.5, 1.0, 2.0} | Relative to contrastive |

### Momentum Schedule
Linear increase from `momentum_start` to `momentum_end`:
```python
momentum = momentum_start + (momentum_end - momentum_start) * (step / total_steps)
```

---

## 5. Evaluation Protocol

### Downstream Tasks

#### A. Classification (Primary)
| Dataset | Metric | Setup |
|---------|--------|-------|
| VGGSound | Top-1/5 Accuracy | Full finetuning |
| AudioSet-20K | mAP, AUC | Full finetuning |

**Modality variants:**
- Multimodal (audio + visual)
- Audio-only
- Visual-only

#### B. Retrieval
| Task | Metrics |
|------|---------|
| Audio → Visual | R@1, R@5, R@10 |
| Visual → Audio | R@1, R@5, R@10 |

#### C. Representation Quality (JEPA-specific)
| Metric | Description |
|--------|-------------|
| Linear probe accuracy | Freeze encoder, train linear head |
| k-NN accuracy | Nearest neighbor classification |
| Representation similarity | CKA between audio/visual embeddings |

### Evaluation Matrix
```
For each model, evaluate:
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Model           │ Multimodal   │ Audio-only   │ Visual-only  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ CAV-MAE (base)  │ ✓ (baseline) │ ✓            │ ✓            │
│ CAV-JEPA Opt A  │ ✓            │ ✓            │ ✓            │
│ CAV-JEPA Opt B  │ ✓            │ ✓            │ ✓            │
│ CAV-JEPA Opt C  │ ✓            │ ✓            │ ✓            │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 6. Experiment Phases

### Phase 1: Baseline & Initial Training
1. [x] Port environment setup from cav-mae-merge
2. [ ] Implement CAV-JEPA model architecture
3. [ ] Adapt I-JEPA weights for audio-visual
4. [ ] Train CAV-JEPA Option B (I-JEPA for both)
5. [ ] Evaluate on VGGSound and AudioSet

### Phase 2: Initialization Comparison
1. [ ] Train Option A (I-JEPA + A-JEPA) if A-JEPA available
2. [ ] Train Option C (V-JEPA)
3. [ ] Compare all initialization strategies

### Phase 3: Ablations
1. [ ] Predictor depth sweep (2, 4, 6 blocks)
2. [ ] Momentum schedule sweep
3. [ ] JEPA loss weight sweep

### Phase 4: Analysis
1. [ ] Compare CAV-JEPA vs CAV-MAE across all metrics
2. [ ] Analyze representation quality differences
3. [ ] Visualize learned attention patterns
4. [ ] Statistical significance tests

---

## 7. Expected Results

| Model | VGGSound Acc | AudioSet mAP | Retrieval R@1 | Notes |
|-------|--------------|--------------|---------------|-------|
| CAV-MAE (baseline) | ~65.8% | ~42.0 | ~50% | Joint MAE + Contrastive |
| CAV-JEPA Option B | ? | ? | ? | I-JEPA init |
| CAV-JEPA Option A | ? | ? | ? | I-JEPA + A-JEPA |
| CAV-JEPA Option C | ? | ? | ? | V-JEPA init |

**Hypotheses:**
1. JEPA should learn more semantic representations than pixel-level MAE
2. Modality-specific initialization (Option A) should outperform single-modal (Option B)
3. Lighter predictor should speed up training without hurting quality

---

## 8. Implementation Checklist

### Model Implementation
- [ ] `src/models/cav_jepa.py` - CAVJEPA class
- [ ] `src/models/__init__.py` - Export CAVJEPA

### Weight Adaptation
- [ ] `src/adapt_jepa_weights.py` - Checkpoint conversion
- [ ] Download I-JEPA checkpoint (ViT-B/16)
- [ ] Download V-JEPA checkpoint (optional)
- [ ] Locate/train A-JEPA checkpoint (optional)

### Training Scripts
- [ ] `src/traintest_cavjepa.py` - Training loop
- [ ] `src/run_cavjepa_pretrain.py` - Entry point
- [ ] `egs/audioset/run_cavjepa_pretrain.sh`
- [ ] `egs/vggsound/run_cavjepa_pretrain.sh`

### Evaluation
- [ ] Adapt finetuning scripts for CAV-JEPA
- [ ] Add linear probe evaluation
- [ ] Add retrieval evaluation

---

## 9. References

- [I-JEPA (Assran et al., CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA (Bardes et al., 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)
- [A-JEPA (Fei et al., 2023)](https://arxiv.org/abs/2311.15830)
- [V-JEPA 2 (2025)](https://arxiv.org/abs/2506.09985)
- [CAV-MAE (Gong et al., ICLR 2023)](https://openreview.net/forum?id=QPtMRyk5rb)
- [I-JEPA GitHub](https://github.com/facebookresearch/ijepa)
- [V-JEPA GitHub](https://github.com/facebookresearch/jepa)
