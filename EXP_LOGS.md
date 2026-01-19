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

### Run 2: CAV-JEPA v2 Ablation - E1c (Random Init, 100 Epochs)
| Field | Value |
|-------|-------|
| **Job ID** | 314545 |
| **Status** | 🔄 Running |
| **Started** | 2026-01-19 |
| **Script** | `egs/audioset/run_cavjepa_ablations.sh E1c` |
| **Output Dir** | `egs/audioset/exp/ablation-E1c-random_init-100ep-v2sched` |
| **Log File** | `egs/audioset/log/314545_cavjepa_ablation.txt` |

**Key v2 Improvements Applied**:
- ✅ Random initialization (no MAE pretrain bias)
- ✅ Depth-scaled weight rescaling (I-JEPA style)
- ✅ Learnable mask tokens in predictor
- ✅ Target representation normalization
- ✅ Warmup + cosine LR scheduler
- ✅ Weight decay schedule (0.04 → 0.4)

**Training Configuration**:
- Init mode: `random`
- Epochs: 100
- Warmup epochs: 15
- LR schedule: 1e-4 → 1e-3 → 1e-6
- WD schedule: 0.04 → 0.4
- Batch size: 120
- GPUs: 4 x H100

**Hypothesis**: Random init with proper I-JEPA-style training should outperform MAE-init by avoiding pretrain bias and allowing JEPA-specific representation learning.

---

### Run 3: CAV-JEPA v1 Baseline - E1a (MAE Init, 25 Epochs)
| Field | Value |
|-------|-------|
| **Job ID** | 314548 |
| **Status** | 🔄 Running |
| **Started** | 2026-01-19 |
| **Script** | `egs/audioset/run_cavjepa_ablations.sh E1a` |
| **Output Dir** | `egs/audioset/exp/ablation-E1a-mae_init-25ep-v1sched` |
| **Log File** | `egs/audioset/log/314548_cavjepa_ablation.txt` |

**Configuration** (Original v1):
- Init mode: `mae` (from `cav_jepa_from_mae_init.pth`)
- Epochs: 25
- Scheduler: v1 (MultiStepLR)
- No warmup, no WD schedule

**Purpose**: Baseline for comparison with v2 improvements.

---

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

**Training Progress** (as of Epoch 20/25 - 80% complete):

| Metric | Epoch 1 | Epoch 13 | Epoch 20 | Total Change |
|--------|---------|----------|----------|--------------|
| Total Loss | 0.440 | 0.321 | 0.245 | -44.3% |
| JEPA Audio Loss | 0.167 | 0.086 | 0.068 | -59.3% |
| JEPA Visual Loss | 0.187 | 0.196 | 0.149 | -20.3% |
| Contrastive Loss | 0.086 | 0.039 | 0.029 | -66.3% |
| Contrastive Acc | 36.9% | 69.3% | 76-79% | +40% |
| Momentum | 0.996 | 0.9975 | 0.9983 | - |

**Key Observations**:
- JEPA Audio loss continues to decrease strongly (-59% overall)
- JEPA Visual loss now also decreasing (epoch 13→20: 0.196→0.149)
- Contrastive accuracy reached ~77%, approaching CAV-MAE baseline (~70% eval)
- Training is stable with momentum schedule working as expected

**Notes**: Uses `cav_jepa_from_mae_init.pth` adapted from CAV-MAE's IN-initial.pth to ensure fair comparison. Target encoder initialized as copy of context encoder.

**Training Curves**: See `egs/audioset/training_curves_cavjepa.png`

**VGGSound Retrieval Results** (epoch 20 checkpoint):

| Model | Direction | R@1 | R@5 | R@10 | Median Rank |
|-------|-----------|-----|-----|------|-------------|
| **CAV-JEPA** | A→V | 12.5% | 29.9% | 38.7% | 23 |
| **CAV-JEPA** | V→A | 14.1% | 31.9% | 40.5% | 19 |
| Contrastive-only | A→V | 16.0% | 35.7% | 44.9% | 15 |
| Contrastive-only | V→A | 16.2% | 37.3% | 46.1% | 14 |

**Retrieval Analysis**:
- Contrastive-only outperforms CAV-JEPA on retrieval (R@1 +3-4%, MR 6-8 positions better)
- This is expected: contrastive learning directly optimizes audio-visual alignment
- JEPA objective focuses on latent prediction, may benefit different downstream tasks
- V→A direction shows closer performance than A→V

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

### Phase 2: Initial Training ✅
- [x] ImageMAE-based initialization training (Job 312964) - Completed
- [x] VGGSound retrieval evaluation (R@1=12.5%)

### Phase 3: CAV-JEPA v2 Implementation ✅
- [x] Random initialization mode (`init_mode='random'`)
- [x] Depth-scaled weight rescaling (I-JEPA style)
- [x] Learnable mask tokens in predictor
- [x] Target representation normalization
- [x] Warmup + cosine LR scheduler (`src/schedulers.py`)
- [x] Weight decay schedule (cosine 0.04 → 0.4)
- [x] Multiblock masking module (`src/masks/`)
- [x] V2 training scripts (`run_cavjepa_v2_pretrain.sh`, `run_cavjepa_ablations.sh`)

### Phase 4: V2 Ablation Study (In Progress)
| Experiment | Config | Status | Job ID |
|------------|--------|--------|--------|
| E1a | MAE init, 25 ep, v1 sched | 🔄 Running | 314548 |
| E1b | MAE init, 100 ep, v2 sched | Pending | - |
| **E1c** | **Random init, 100 ep, v2 sched** | 🔄 Running | 314545 |
| E1d | Random init, 300 ep, v2 sched | Pending | - |
| E2a | Random init, v1 scheduler | Pending | - |
| E2b | Random init, v2 scheduler | Pending | - |
| E3a | Random init, no target norm | Pending | - |
| E3b | Random init, with target norm | Pending | - |

### Phase 5: Downstream Evaluation (Pending)
- [ ] VGGSound retrieval comparison (v1 vs v2)
- [ ] AudioSet-20K fine-tuning (linear probe)
- [ ] AudioSet-20K fine-tuning (end-to-end)
- [ ] Cross-dataset transfer (ESC-50)

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

| Model | Init | JEPA/MAE Loss | Contrastive Acc | VGGSound R@1 | Notes |
|-------|------|---------------|-----------------|--------------|-------|
| CAV-MAE (baseline) | ImageMAE | N/A | ~70% | - | Joint MAE+Contrastive |
| Contrastive-only | ImageMAE | N/A | 68% eval | **16.0%** A→V | Best retrieval |
| CAV-JEPA (Job 312964) | ImageMAE | 0.22 (A:0.07, V:0.15) | 77% | 12.5% A→V | Training (20/25 epochs) |

---

## Changelog

- **2026-01-19**: Launched v2 ablation experiments: E1c (Job 314545) random init 100ep, E1a (Job 314548) MAE baseline 25ep
- **2026-01-19**: Implemented CAV-JEPA v2 with I-JEPA improvements: random init, learnable mask tokens, target norm, warmup+cosine scheduler, WD schedule
- **2026-01-19**: Created new files: `src/schedulers.py`, `src/masks/multiblock.py`, `src/masks/utils.py`
- **2026-01-19**: Created ablation scripts: `run_cavjepa_v2_pretrain.sh`, `run_cavjepa_ablations.sh`
- **2026-01-16**: VGGSound retrieval evaluation - CAV-JEPA R@1=12.5%, Contrastive-only R@1=16.0%
- **2026-01-16**: CAV-JEPA training progress (epoch 20/25): Total Loss 0.245, Contrastive Acc ~77%
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
