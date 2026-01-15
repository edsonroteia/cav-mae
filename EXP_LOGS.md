# CAV-JEPA Experiment Logs

This document tracks all experiment runs for the CAV-JEPA project (replacing MAE with JEPA while keeping contrastive learning).

---

## Experiment Overview

**Research Question**: Can JEPA (Joint-Embedding Predictive Architecture) replace the MAE objective in CAV-MAE while maintaining or improving performance?

**Key Differences from CAV-MAE**:
- Predicts in latent space instead of pixel space
- Uses target encoder with EMA updates
- Lightweight predictor instead of full decoder

**Dataset**: AudioSet 2M / VGGSound for pretraining

**Cluster**: Ferranti H100 (4 GPUs per job, 256GB memory)

---

## Initialization Options

| Option | Visual Encoder | Audio Encoder | Notes |
|--------|----------------|---------------|-------|
| **A** | I-JEPA ViT-B/16 | A-JEPA | Recommended: modality-specific |
| **B** | I-JEPA ViT-B/16 | I-JEPA ViT-B/16 | Fallback if A-JEPA unavailable |
| **C** | V-JEPA ViT-B/16 | V-JEPA ViT-B/16 | Video-based, temporal |
| **D** | V-JEPA ViT-B/16 | A-JEPA | Hybrid |

---

## Active Experiments

### Run 1: CAV-JEPA Initial Training (ImageMAE Init)
| Field | Value |
|-------|-------|
| **Job ID** | 312906 |
| **Status** | Pending (Resources) |
| **Submitted** | 2026-01-15 |
| **Script** | `egs/audioset/run_cavjepa_pretrain.sh` |
| **Output Dir** | `egs/audioset/exp/cavjepa-audioset-lr1e-4-epoch25-bs120-mr0.75-mom0.996-0.999-pred4` |
| **Log File** | `egs/audioset/log/312906_cavjepa_pretrain.txt` |
| **Initialization** | ImageMAE-based (same as CAV-MAE for fair comparison) |

**JEPA Hyperparameters**:
- Momentum start: 0.996
- Momentum end: 0.999
- Predictor depth: 4 blocks
- Predictor dim: 384
- JEPA loss weight: 1.0
- Contrastive loss weight: 0.01

**Training Hyperparameters**:
- Learning rate: 1e-4
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- GPUs: 4 x H100

**Notes**: Uses `cav_jepa_from_mae_init.pth` adapted from CAV-MAE's IN-initial.pth to ensure fair comparison. Target encoder initialized as copy of context encoder.

---

## Baseline Models (from CAV-MAE)

| Model | Source | Status | Notes |
|-------|--------|--------|-------|
| CAV-MAE Joint | cav-mae-merge | Available | MAE + Contrastive baseline |
| CAV-MAE MAE-only | cav-mae-merge | Training | Job 312886 |
| CAV-MAE Contrastive-only | cav-mae-merge | Training | Job 312887 |

---

## Pending Experiments

### Phase 1: Implementation ✅
- [x] Implement `src/models/cav_jepa.py`
- [x] Create weight adaptation script `src/adapt_cavmae_to_jepa.py`
- [x] Create training loop `src/traintest_cavjepa.py`
- [x] Create entry point `src/run_cavjepa_pretrain.py`
- [x] Create training script `egs/audioset/run_cavjepa_pretrain.sh`

### Phase 2: Initial Training (In Progress)
- [x] ImageMAE-based initialization training (Job 312906) - PENDING
- [ ] Evaluate on downstream tasks (VGGSound, AudioSet)

### Phase 3: Ablations
- [ ] I-JEPA initialization (if ViT-B checkpoints become available)
- [ ] V-JEPA initialization
- [ ] Predictor depth sweep (2, 4, 6 blocks)
- [ ] Momentum schedule sweep (0.99-0.999, 0.996-0.9999)

---

## Useful Commands

```bash
# Check job status
squeue -u kqr867

# Watch specific job logs
tail -f egs/audioset/log/<jobid>_cavjepa.txt

# Cancel a job
scancel <JOB_ID>

# Check job details
scontrol show job <JOB_ID>

# Activate environment
source activate_env.sh
```

---

## Results Summary

| Model | Init | JEPA Loss | Contrastive Acc | VGGSound Acc | AudioSet mAP | Notes |
|-------|------|-----------|-----------------|--------------|--------------|-------|
| CAV-MAE (baseline) | ImageMAE | N/A | ~70% | ~65.8% | ~42.0 | Joint MAE+Contrastive |
| CAV-JEPA (Job 312906) | ImageMAE | - | - | - | - | Training |

---

## Changelog

- **2026-01-15**: Submitted CAV-JEPA initial training (Job 312906)
- **2026-01-15**: Created `adapt_cavmae_to_jepa.py` for fair initialization from CAV-MAE weights
- **2026-01-15**: Fixed numpy deprecation in `pos_embed.py` (np.float -> np.float64)
- **2026-01-15**: Implemented full CAV-JEPA architecture with reference I-JEPA patterns
  - Target encoder with EMA updates (in-place operations)
  - Predictor network (4 blocks, 384 dim)
  - smooth_l1_loss following I-JEPA reference
- **2026-01-15**: Created experiment tracking infrastructure
- **2026-01-15**: Ported environment setup from cav-mae-merge branch
