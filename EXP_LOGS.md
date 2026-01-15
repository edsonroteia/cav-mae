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

### Run 1: [Template - Not Yet Started]
| Field | Value |
|-------|-------|
| **Job ID** | - |
| **Status** | Not Started |
| **Submitted** | - |
| **Script** | `egs/audioset/run_cavjepa_pretrain.sh` |
| **Output Dir** | `egs/audioset/exp/cavjepa-...` |
| **Log File** | `egs/audioset/log/<jobid>_cavjepa.txt` |
| **Initialization** | Option B (I-JEPA for both) |

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

---

## Baseline Models (from CAV-MAE)

| Model | Source | Status | Notes |
|-------|--------|--------|-------|
| CAV-MAE Joint | cav-mae-merge | Available | MAE + Contrastive baseline |
| CAV-MAE MAE-only | cav-mae-merge | Training | Job 312886 |
| CAV-MAE Contrastive-only | cav-mae-merge | Training | Job 312887 |

---

## Pending Experiments

### Phase 1: Implementation
- [ ] Implement `src/models/cav_jepa.py`
- [ ] Create weight adaptation script `src/adapt_jepa_weights.py`
- [ ] Create training loop `src/traintest_cavjepa.py`
- [ ] Create training scripts for AudioSet/VGGSound

### Phase 2: Initial Training
- [ ] Option B training (I-JEPA initialization)
- [ ] Evaluate on downstream tasks

### Phase 3: Ablations
- [ ] Option A training (I-JEPA + A-JEPA)
- [ ] Option C training (V-JEPA)
- [ ] Predictor depth sweep (2, 4, 6 blocks)
- [ ] Momentum schedule sweep

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
| CAV-JEPA Option B | I-JEPA | - | - | - | - | Pending |
| CAV-JEPA Option A | I-JEPA+A-JEPA | - | - | - | - | Pending |
| CAV-JEPA Option C | V-JEPA | - | - | - | - | Pending |

---

## Changelog

- **2026-01-15**: Created experiment tracking infrastructure
- **2026-01-15**: Ported environment setup from cav-mae-merge branch
