# CAV-MAE Model Merging Experiment Logs

This document tracks all experiment runs for the model merging baseline project.

---

## Experiment Overview

**Research Question**: Can training MAE and Contrastive objectives separately and merging match or exceed joint multi-task training?

**Dataset**: AudioSet 2M (1,753,193 samples for pretraining, 17,354 for evaluation)

**Cluster**: Ferranti H100 (4 GPUs per job, 256GB memory)

---

## Active Experiments

### Run 1: MAE-Only Pretraining (lr=1e-4)
| Field | Value |
|-------|-------|
| **Job ID** | 312886 |
| **Status** | ✅ Completed |
| **Submitted** | 2026-01-15 |
| **Completed** | 2026-01-16 |
| **Script** | `egs/audioset/run_cavmae_pretrain_mae_only.sh` |
| **Output Dir** | `egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/312886_mae_only.txt` |

**Hyperparameters**:
- Learning rate: 1e-4
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- MAE loss weight: 1.0
- Contrastive loss weight: 0.0 (disabled)

**Results**:
- Train MAE Loss: 3.67 → 3.40 (**only 6% reduction**)
- Audio MAE: 2.36 → 2.16
- Visual MAE: 1.31 → 1.24
- Eval MAE Loss: 3.30
- **Issue**: Loss barely decreased - LR may be too low without contrastive gradients

---

### Run 1b: MAE-Only Pretraining (lr=2e-4) - RETRY WITH HIGHER LR
| Field | Value |
|-------|-------|
| **Job ID** | 313261 |
| **Status** | 🔄 Running |
| **Submitted** | 2026-01-16 |
| **Script** | `egs/audioset/run_mae_only_lr2e-4.sh` |
| **Output Dir** | `egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr2e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/313261_mae_only_lr2e-4.txt` |

**Hyperparameters**:
- Learning rate: **2e-4** (2x previous)
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- MAE loss weight: 1.0
- Contrastive loss weight: 0.0 (disabled)

**Hypothesis**: MAE-only training needs higher LR since it lacks the additional gradient signal from contrastive loss

---

### Run 2: Contrastive-Only Pretraining
| Field | Value |
|-------|-------|
| **Job ID** | 312887 |
| **Status** | ✅ Completed |
| **Submitted** | 2026-01-15 |
| **Completed** | 2026-01-16 |
| **Script** | `egs/audioset/run_cavmae_pretrain_contrastive_only.sh` |
| **Output Dir** | `egs/audioset/exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/312887_contrastive_only.txt` |

**Hyperparameters**:
- Learning rate: 1e-4
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- MAE loss weight: 0.0 (disabled)
- Contrastive loss weight: 1.0

**Results**:
- Train Contrastive Loss: 13.5 → 2.1 (**84% reduction**)
- Train Contrastive Acc: 5% → 83%
- Eval Contrastive Loss: 4.44
- Eval Contrastive Acc: 68%
- **Strong convergence** - contrastive objective trains well in isolation

---

## Existing Models (Baseline)

### Joint Training (Pre-existing)
| Field | Value |
|-------|-------|
| **Status** | Completed (pre-existing) |
| **Output Dir** | TBD - update with actual path |
| **Notes** | Joint MAE + Contrastive training baseline |

---

## Pending Experiments

### Model Merging (after pretraining completes)
- **Script**: `egs/audioset/merge_models.sh`
- **Variants to create**:
  - `merged_simple.pth` - Simple averaging
  - `merged_weighted_0.3.pth` - 30% MAE, 70% Contrastive
  - `merged_weighted_0.5.pth` - 50% MAE, 50% Contrastive
  - `merged_weighted_0.7.pth` - 70% MAE, 30% Contrastive
  - `merged_task_arith_0.5.pth` - Task arithmetic λ=0.5
  - `merged_task_arith_1.0.pth` - Task arithmetic λ=1.0
  - `merged_task_arith_1.5.pth` - Task arithmetic λ=1.5

### Finetuning (after merging)
- **Script**: `egs/audioset/run_cavmae_ft.sh`
- **Dataset**: AudioSet 20k balanced
- **Evaluation**: 527-class classification

---

## Useful Commands

```bash
# Check job status
squeue -u kqr867

# Watch specific job logs
tail -f egs/audioset/log/312868_mae_only.txt
tail -f egs/audioset/log/312869_contrastive_only.txt

# Cancel a job
scancel <JOB_ID>

# Check job details
scontrol show job <JOB_ID>
```

---

## Results Summary

| Model | MAE Loss | Contrastive Acc | Downstream Acc | Notes |
|-------|----------|-----------------|----------------|-------|
| Joint (baseline) | TBD | TBD | TBD | Pre-existing |
| MAE-only (lr=1e-4) | 3.40 | N/A | - | Job 312886 ✅ - Poor convergence |
| MAE-only (lr=2e-4) | - | N/A | - | Job 313261 🔄 - Higher LR retry |
| Contrastive-only | N/A | 68% | - | Job 312887 ✅ - Good convergence |
| Merged (simple) | - | - | - | Pending |
| Merged (weighted 0.5) | - | - | - | Pending |
| Merged (task arith 1.0) | - | - | - | Pending |

**Training Curves**: See `egs/audioset/training_curves_ablation.png`

---

## Changelog

- **2026-01-16**: Launched MAE-only lr=2e-4 (Job 313261) to test higher LR hypothesis
- **2026-01-16**: Created training curves plot `egs/audioset/training_curves_ablation.png`
- **2026-01-16**: MAE-only (312886) completed - only 6% loss reduction, suggesting LR too low
- **2026-01-16**: Contrastive-only (312887) completed - 68% eval acc, strong convergence
- **2026-01-15**: Relaunched MAE-only (312886) and Contrastive-only (312887) with dedicated cav-mae venv
- **2026-01-15**: Created dedicated venv with uv, fixed timm/numpy API compatibility issues
- **2026-01-15**: Previous runs (312874, 312875) failed - missing timm module in avllm-eval env
- **2026-01-15**: Relaunched MAE-only (312874) and Contrastive-only (312875) after fixing tenv alias issue
- **2026-01-15**: Fixed scripts - replaced `tenv` alias with explicit env activation command
- **2026-01-15**: Initial launch failed (312868, 312869) - tenv alias not available in sbatch
- **2026-01-15**: Created experiment infrastructure (scripts, JSON datafiles, directory structure)
