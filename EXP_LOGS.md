# CAV-MAE Model Merging Experiment Logs

This document tracks all experiment runs for the model merging baseline project.

---

## Experiment Overview

**Research Question**: Can training MAE and Contrastive objectives separately and merging match or exceed joint multi-task training?

**Dataset**: AudioSet 2M (1,753,193 samples for pretraining, 17,354 for evaluation)

**Cluster**: Ferranti H100 (4 GPUs per job, 256GB memory)

---

## Active Experiments

### Run 1: MAE-Only Pretraining
| Field | Value |
|-------|-------|
| **Job ID** | 312886 (prev: 312874 failed - missing timm; 312868 failed - tenv alias issue) |
| **Status** | Running |
| **Submitted** | 2026-01-15 |
| **Script** | `egs/audioset/run_cavmae_pretrain_mae_only.sh` |
| **Output Dir** | `egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/312886_mae_only.txt` |
| **Error File** | `egs/audioset/log/312886_mae_only.err` |
| **Partition** | h100-ferranti |
| **GPUs** | 4 |
| **Time Limit** | 2 days |

**Hyperparameters**:
- Learning rate: 1e-4
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- MAE loss weight: 1.0
- Contrastive loss weight: 0.0 (disabled)

---

### Run 2: Contrastive-Only Pretraining
| Field | Value |
|-------|-------|
| **Job ID** | 312887 (prev: 312875 failed - missing timm; 312869 failed - tenv alias issue) |
| **Status** | Running |
| **Submitted** | 2026-01-15 |
| **Script** | `egs/audioset/run_cavmae_pretrain_contrastive_only.sh` |
| **Output Dir** | `egs/audioset/exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/312887_contrastive_only.txt` |
| **Error File** | `egs/audioset/log/312887_contrastive_only.err` |
| **Partition** | h100-ferranti |
| **GPUs** | 4 |
| **Time Limit** | 2 days |

**Hyperparameters**:
- Learning rate: 1e-4
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- MAE loss weight: 0.0 (disabled)
- Contrastive loss weight: 1.0

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

| Model | MAE Loss | Contrastive Loss | Downstream Acc | Notes |
|-------|----------|------------------|----------------|-------|
| Joint (baseline) | TBD | TBD | TBD | Pre-existing |
| MAE-only | - | - | - | Job 312874 |
| Contrastive-only | - | - | - | Job 312875 |
| Merged (simple) | - | - | - | Pending |
| Merged (weighted 0.5) | - | - | - | Pending |
| Merged (task arith 1.0) | - | - | - | Pending |

---

## Changelog

- **2026-01-15**: Relaunched MAE-only (312886) and Contrastive-only (312887) with dedicated cav-mae venv
- **2026-01-15**: Created dedicated venv with uv, fixed timm/numpy API compatibility issues
- **2026-01-15**: Previous runs (312874, 312875) failed - missing timm module in avllm-eval env
- **2026-01-15**: Relaunched MAE-only (312874) and Contrastive-only (312875) after fixing tenv alias issue
- **2026-01-15**: Fixed scripts - replaced `tenv` alias with explicit env activation command
- **2026-01-15**: Initial launch failed (312868, 312869) - tenv alias not available in sbatch
- **2026-01-15**: Created experiment infrastructure (scripts, JSON datafiles, directory structure)
