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
- **CRITICAL ISSUE**: LayerNorm weights (`norm_a`, `norm_v`) collapsed to ALL ZEROS!
  - This causes the model to output all zeros for any input
  - Root cause: MAE-only loss doesn't provide sufficient gradient signal to norm layers
  - The model is unusable for retrieval/downstream tasks
  - Waiting for lr=2e-4 retry (Job 313261) to see if higher LR prevents collapse

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

**VGGSound Retrieval Results**:
| Direction | R@1 | R@5 | R@10 | Median Rank |
|-----------|-----|-----|------|-------------|
| Audio→Visual | **16.0%** | 35.7% | 44.9% | **15** |
| Visual→Audio | **16.2%** | 37.3% | 46.1% | **14** |

---

## Existing Models (Baseline)

### Joint Training (Pre-existing)
| Field | Value |
|-------|-------|
| **Status** | Completed (pre-existing) |
| **Output Dir** | TBD - update with actual path |
| **Notes** | Joint MAE + Contrastive training baseline |

---

## Model Merging - Completed

### Run 3: Model Merging
| Field | Value |
|-------|-------|
| **Status** | ✅ Completed |
| **Completed** | 2026-01-16 |
| **Script** | `egs/audioset/merge_models.sh` |
| **Output Dir** | `egs/audioset/exp/merged-models/` |

**Models Created**:
| Model File | Method | Parameters |
|------------|--------|------------|
| `merged_simple.pth` | Simple Averaging | (MAE + Contrastive) / 2 |
| `merged_weighted_0.3.pth` | Weighted | 30% MAE, 70% Contrastive |
| `merged_weighted_0.5.pth` | Weighted | 50% MAE, 50% Contrastive |
| `merged_weighted_0.7.pth` | Weighted | 70% MAE, 30% Contrastive |
| `merged_task_arith_0.5.pth` | Task Arithmetic | λ=0.5 |
| `merged_task_arith_1.0.pth` | Task Arithmetic | λ=1.0 |
| `merged_task_arith_1.5.pth` | Task Arithmetic | λ=1.5 |

---

### Run 4: VGGSound Retrieval Evaluation (All Models)
| Field | Value |
|-------|-------|
| **Job ID** | 313681 (resubmitted, 313680 failed CUDA error on mlcbm004) |
| **Status** | 🔄 Running |
| **Submitted** | 2026-01-16 |
| **Script** | `egs/audioset/run_retrieval_all.sh` |
| **Output Dir** | `egs/audioset/exp/retrieval_results/` |
| **Log File** | `egs/audioset/log/313681_retrieval_all.txt` |

**Models Being Evaluated**:
1. Base (IN-initial.pth)
2. MAE-only (lr=1e-4)
3. Contrastive-only (lr=1e-4)
4. Merged: Simple average
5. Merged: Weighted α=0.3
6. Merged: Weighted α=0.5
7. Merged: Weighted α=0.7
8. Merged: Task arithmetic λ=0.5
9. Merged: Task arithmetic λ=1.0
10. Merged: Task arithmetic λ=1.5

---

## Pending Experiments

### Finetuning (after retrieval evaluation)
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

### VGGSound Retrieval Results (Verified)

| Model | A→V R@1 | A→V R@5 | A→V R@10 | V→A R@1 | V→A R@5 | V→A R@10 | MR A→V |
|-------|---------|---------|----------|---------|---------|----------|--------|
| Base (IN-initial) | 0.06% | 3.3% | 7.5% | 0.12% | 3.3% | 7.2% | 699 |
| **Contrastive-only** | **16.0%** | **35.7%** | **44.9%** | **16.2%** | **37.4%** | **46.1%** | **15** |
| MAE-only (lr=1e-4) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Key Finding**: Contrastive-only shows **267x improvement** in R@1 over base model (16.0% vs 0.06%)!

### Merged Models Results (Original - Broken Norms)

These models were merged WITHOUT the norm layer fix, so they inherit the broken norm_a/norm_v from MAE-only:

| Model | Method | A→V R@1 | A→V R@5 | V→A R@1 | V→A R@5 | A→V MR |
|-------|--------|---------|---------|---------|---------|--------|
| Weighted α=0.3 | 30% MAE, 70% C | 4.88% | 16.0% | 6.80% | 19.9% | 65 |
| Simple Average | 50/50 | 1.06% | 2.76% | 1.44% | 5.07% | 269 |
| Weighted α=0.5 | 50% each | 1.06% | 2.76% | 1.44% | 5.07% | 269 |
| Task Arith λ=0.5 | - | 1.06% | 2.76% | 1.44% | 5.07% | 269 |
| Weighted α=0.7 | 70% MAE, 30% C | 0.18% | 0.88% | 1.09% | 4.95% | 537 |
| Task Arith λ=1.5 | - | 0.0% | 0.24% | 0.23% | 0.46% | 833 |

**Pattern**: More MAE weight = worse results, confirming the norm layer collapse issue.

### MAE-only Issue Analysis

**LayerNorm Observation**: norm_a/norm_v weights in MAE-only model are ~0 (1e-38), not trained because:
- MAE loss flows through decoder path (forward_mae → decoder)
- Contrastive loss flows through forward_feat → norm_a/norm_v
- With contrastive loss weight = 0, these norms never get gradient updates

**Why norm fix doesn't help**:
- MAE norms ≈ 0, so broken = 0.5 * contrastive (uniform scale across ALL dimensions)
- Fixed = 1.0 * contrastive
- Retrieval uses L2 normalization → uniform scaling cancels out completely
- Broken and fixed produce **identical** cosine similarities

**Real Root Cause**: The issue is the **encoder weights** (transformer blocks), not just the norms:
- MAE training optimizes encoder for reconstruction (low-level features)
- Contrastive training optimizes encoder for alignment (semantic features)
- Averaging these encoder weights produces features that don't work well for either task
- Only heavily weighting contrastive (α=0.3 → 70% contrastive) preserves retrieval ability

### Training Results Summary

| Model | MAE Loss | Contrastive Acc | Notes |
|-------|----------|-----------------|-------|
| Joint (baseline) | TBD | TBD | Pre-existing |
| MAE-only (lr=1e-4) | 3.40 | N/A | ❌ Broken - norm collapse |
| MAE-only (lr=2e-4) | - | N/A | 🔄 Running - Higher LR retry |
| Contrastive-only | N/A | 68% | ✅ Working well |

**Training Curves**: See `egs/audioset/training_curves_ablation.png`

**Retrieval Script**: `src/run_retrieval.py` (fixed for timm, autocast, float32 handling)

---

## Changelog

- **2026-01-16**: Discovered norm fix doesn't help - MAE norms=0, uniform scaling erased by L2 norm in retrieval
- **2026-01-16**: Identified real issue: encoder weights learn different representations (reconstruction vs alignment)
- **2026-01-16**: Merged model evaluation (Job 313690) shows pattern: more MAE weight = worse retrieval results
- **2026-01-16**: VGGSound retrieval verified - Base: R@1=0.06%, Contrastive-only: R@1=16.0% (267x improvement!)
- **2026-01-16**: Discovered MAE-only model has collapsed LayerNorm weights (all zeros) - model unusable
- **2026-01-16**: Fixed run_retrieval.py for autocast/float32 precision issues
- **2026-01-16**: Submitted batch retrieval evaluation (Job 313680) for all 10 models - failed due to MAE issue
- **2026-01-16**: Completed model merging - created 7 merged variants in `exp/merged-models/`
- **2026-01-16**: VGGSound retrieval - Contrastive-only R@1=16.0% A→V, 16.2% V→A
- **2026-01-16**: Created unified retrieval script `src/run_retrieval.py`
- **2026-01-16**: Fixed timm compatibility in `src/models/cav_mae.py` (removed qk_scale from Attention)
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
