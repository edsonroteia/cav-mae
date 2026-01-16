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
| **Job ID** | 312964 |
| **Status** | ✅ Running |
| **Started** | 2026-01-15 20:41 |
| **Node** | mlcbm012 |
| **Script** | `egs/audioset/run_cavjepa_pretrain.sh` |
| **Output Dir** | `egs/audioset/exp/cavjepa-audioset-lr1e-4-epoch25-bs120-mr0.75-mom0.996-0.999-pred4` |
| **Log File** | `egs/audioset/log/312964_cavjepa_pretrain.txt` |
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

**Training Progress** (as of Epoch 13/25):

| Metric | Epoch 1 | Epoch 13 | Change |
|--------|---------|----------|--------|
| Total Loss | 0.440 | 0.321 | -27.0% |
| JEPA Audio Loss | 0.167 | 0.086 | -48.7% |
| JEPA Visual Loss | 0.187 | 0.196 | +5.0% |
| Contrastive Loss | 0.086 | 0.039 | -54.5% |
| Contrastive Acc | 36.9% | 69.3% | +32.4% |
| Momentum | 0.996 | 0.9975 | +0.1% |

**Key Observations**:
- JEPA Audio loss decreased significantly (-49%), suggesting audio representations are well-suited for latent-space prediction
- JEPA Visual loss remained stable (+5%), visual may be harder to predict in latent space
- Contrastive accuracy reached 69%, comparable to CAV-MAE baseline (~70%)
- Training is stable with momentum schedule working as expected

**Notes**: Uses `cav_jepa_from_mae_init.pth` adapted from CAV-MAE's IN-initial.pth to ensure fair comparison. Target encoder initialized as copy of context encoder.

**Training Curves**: See `egs/audioset/training_curves_cavjepa.png`

**Test Run (Job 312962)**: Verified training works - losses stable (~0.11-0.13), JEPA audio ~0.04, JEPA visual ~0.05, contrastive ~0.02, momentum schedule working correctly.

---

## Baseline Models (from CAV-MAE)

| Model | Source | Status | Job ID | Notes |
|-------|--------|--------|--------|-------|
| CAV-MAE Joint | cav-mae-merge | Available | - | MAE + Contrastive baseline |
| CAV-MAE MAE-only (lr=1e-4) | cav-mae-merge | ✅ Completed | 312886 | 25 epochs, loss 3.6→3.4 (only 6% drop) |
| CAV-MAE Contrastive-only | cav-mae-merge | ✅ Completed | 312887 | 25 epochs, acc 5%→68% |
| CAV-MAE MAE-only (lr=2e-4) | cav-mae-merge | 🔄 Running | 313261 | Higher LR to improve MAE convergence |

### Completed Baseline Results

**Contrastive-only (Job 312887)**:
- Train Contrastive Loss: 13.5 → 2.1
- Train Contrastive Acc: 5% → 83%
- Eval Contrastive Acc: 68%

**MAE-only lr=1e-4 (Job 312886)**:
- Train MAE Loss: 3.67 → 3.40 (only ~6% reduction)
- Audio MAE: 2.36 → 2.16
- Visual MAE: 1.31 → 1.24
- **Issue**: Loss barely decreased - LR may be too low without contrastive gradients

**MAE-only lr=2e-4 (Job 313261)** - NEW:
- Started: 2026-01-16
- Hypothesis: Higher LR needed for MAE-only training
- Output: `exp/mae-only-audioset-cav-mae-balNone-lr2e-4-epoch25-bs120-normTrue-mr-unstructured-0.75`

---

## Pending Experiments

### Phase 1: Implementation ✅
- [x] Implement `src/models/cav_jepa.py`
- [x] Create weight adaptation script `src/adapt_cavmae_to_jepa.py`
- [x] Create training loop `src/traintest_cavjepa.py`
- [x] Create entry point `src/run_cavjepa_pretrain.py`
- [x] Create training script `egs/audioset/run_cavjepa_pretrain.sh`

### Phase 2: Initial Training (In Progress)
- [x] ImageMAE-based initialization training (Job 312964) - RUNNING
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
| CAV-JEPA (Job 312964) | ImageMAE | 0.28 (A:0.09, V:0.20) | 69.3% | - | - | Training (13/25 epochs) |

---

## Changelog

- **2026-01-16**: CAV-JEPA training progress analysis (epoch 13/25): JEPA audio -49%, contrastive acc 69%
- **2026-01-16**: Created `egs/audioset/parse_and_plot_cavjepa_logs.py` for JEPA training analysis
- **2026-01-16**: Launched MAE-only lr=2e-4 (Job 313261) to test if higher LR improves MAE convergence
- **2026-01-16**: Analyzed completed baseline runs - MAE-only (1e-4) showed minimal loss reduction (6%), contrastive-only showed strong convergence (68% acc)
- **2026-01-16**: Created training curves plot: `egs/audioset/training_curves_ablation.png`
- **2026-01-15 20:41**: Job 312964 started running on mlcbm012 - initial metrics look healthy
- **2026-01-15**: Launched full CAV-JEPA training (Job 312964) after successful test run
- **2026-01-15**: Test run (Job 312962) verified training loop works correctly
- **2026-01-15**: Fixed dtype mismatch in `forward_predictor` (torch.zeros needs explicit dtype)
- **2026-01-15**: Fixed checkpoint loading order (load weights BEFORE DataParallel wrapping)
- **2026-01-15**: Created `adapt_cavmae_to_jepa.py` for fair initialization from CAV-MAE weights
- **2026-01-15**: Fixed numpy deprecation in `pos_embed.py` (np.float -> np.float64)
- **2026-01-15**: Implemented full CAV-JEPA architecture with reference I-JEPA patterns
  - Target encoder with EMA updates (in-place operations)
  - Predictor network (4 blocks, 384 dim)
  - smooth_l1_loss following I-JEPA reference
- **2026-01-15**: Created experiment tracking infrastructure
- **2026-01-15**: Ported environment setup from cav-mae-merge branch
