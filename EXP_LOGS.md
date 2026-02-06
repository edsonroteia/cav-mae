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
  - Waiting for lr=2e-4 retry (Job 313696) to see if higher LR prevents collapse

---

### Run 1b: MAE-Only Pretraining (lr=2e-4) - RETRY WITH HIGHER LR
| Field | Value |
|-------|-------|
| **Job ID** | 317992 (relaunched from 313762) |
| **Status** | 🔄 Running |
| **Submitted** | 2026-02-04 |
| **Failed Attempt** | 313762 - TIME LIMIT (reached epoch 17/25 before 3-day limit) |
| **Script** | `egs/audioset/run_mae_only_lr2e-4.sh` |
| **Output Dir** | `egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr2e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/317992_mae_only_lr2e-4.txt` |

**Hyperparameters**:
- Learning rate: **2e-4** (2x previous)
- Batch size: 120
- Epochs: 25
- Masking ratio: 0.75
- MAE loss weight: 1.0
- Contrastive loss weight: 0.0 (disabled)

**Hypothesis**: MAE-only training needs higher LR since it lacks the additional gradient signal from contrastive loss

**Fixes Applied**:
- Installed `wandb` package in cav-mae environment: `uv pip install wandb`
- Added `#SBATCH --exclude=mlcbm005,mlcbm012` to avoid CUDA-broken nodes
- **Note**: Previous attempt (Job 313261) failed due to timm compatibility issue (`qk_scale` argument). Fixed in `src/models/cav_mae.py` on 2026-01-16 22:23.

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

## Model Merging - Comprehensive Sweeps (2026-01-16)

### New Implementation: Orthogonal Conflict-Aware Merge

Added to `src/merge_models.py`:

**Algorithm**:
```
For each parameter:
  Δ_c = W_contrastive - W_base  (contrastive task vector)
  Δ_m = W_mae - W_base          (MAE task vector)
  s = cos(Δ_c, Δ_m)             (cosine similarity)

  If s >= τ (aligned objectives):
    W_merged = W_base + Δ_m + β·Δ_c
  If s < τ (conflicting objectives):
    Δ_c_⊥ = Δ_c - proj_{Δ_m}(Δ_c)  (orthogonal component)
    W_merged = W_base + Δ_m + β·Δ_c_⊥
```

**CLI Arguments**: `--ortho_beta`, `--ortho_tau`, `--ortho_group`

---

### Run 5: Comprehensive Model Merging Sweeps
| Field | Value |
|-------|-------|
| **Status** | ✅ Completed |
| **Completed** | 2026-01-16 23:23 |
| **Output Dir** | `egs/audioset/exp/merged-models/` |

**Sweep Scripts Created**:
| Script | Method | Combinations |
|--------|--------|--------------|
| `sweep_orthogonal.sh` | Conflict-aware orthogonal | β×τ = 5×5 = 25 models |
| `sweep_weighted_full.sh` | Weighted averaging | α ∈ [0.0-1.0] = 11 models |
| `sweep_task_arithmetic.sh` | Task arithmetic | λ ∈ {0.3-2.0} = 7 models |
| `sweep_dare_ties.sh` | DARE-TIES | kr×λ = 4×3 = 12 models |

**Total Merged Models Created: 55**

**Directory Structure**:
```
exp/merged-models/
├── orthogonal-sweep/     # 25 models (beta × tau combinations)
│   ├── merged_orthogonal_beta0.25_tau0.0.pth
│   ├── merged_orthogonal_beta0.25_tau0.1.pth
│   ├── ... (25 total)
├── weighted-sweep/       # 11 models (alpha 0.0 to 1.0)
│   ├── merged_weighted_alpha0.0.pth
│   ├── merged_weighted_alpha0.1.pth
│   ├── ... (11 total)
├── task-arithmetic-sweep/ # 7 models (lambda values)
│   ├── merged_task_arith_lambda0.3.pth
│   ├── ... (7 total)
└── dare-ties-sweep/      # 12 models (keep_ratio × lambda)
    ├── merged_dare_ties_kr0.1_lambda0.5.pth
    ├── ... (12 total)
```

---

### Run 6: Parallel Retrieval Evaluation (All 55+ Models)
| Field | Value |
|-------|-------|
| **Job IDs** | 313763-313819 (57 jobs, relaunched) |
| **Failed Attempt** | 313698-313754 - Failed due to relative path issues in script |
| **Status** | ✅ Completed |
| **Submitted** | 2026-01-17 06:41 |
| **Completed** | 2026-01-17 07:00 |
| **Script** | `egs/audioset/launch_all_retrieval_jobs.sh` (fixed with absolute paths) |
| **Output Dir** | `egs/audioset/exp/retrieval_results/` |

**Jobs Submitted**:
- 1 job for MAE-only (lr=1e-4) - Job 313763
- 25 jobs for orthogonal sweep - Jobs 313764-313788
- 11 jobs for weighted sweep - Jobs 313789-313799
- 7 jobs for task arithmetic sweep - Jobs 313800-313806
- 12 jobs for DARE-TIES sweep - Jobs 313807-313818
- 1 job for original merged model - Job 313819

**Fixes Applied**:
- Converted relative paths to absolute paths in `launch_all_retrieval_jobs.sh`:
  - `MERGED_MODELS_BASE`: `./exp/merged-models/` → `/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models/`
  - `RESULTS_BASE`: `./exp/retrieval_results/` → `/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/retrieval_results/`
  - Base model paths: `./IN-initial.pth` and `./exp/mae-only.../` paths updated to absolute

**Results Status**: ✅ ALL COMPLETED
- 25 orthogonal sweep jobs completed
- 11 weighted sweep jobs completed
- 7 task arithmetic sweep jobs completed
- 12 DARE-TIES sweep jobs completed
- Total: 128 evaluation results (55 merged models × 2 directions + baselines × 2 directions)

**Aggregation**:
- Ran `aggregate_retrieval_results.sh` - created `all_results_summary.csv` with all results
- Script location: `egs/audioset/aggregate_retrieval_results.sh`

---

## Existing Models (Baseline)

### Joint Training (Pre-existing)
| Field | Value |
|-------|-------|
| **Status** | Completed (pre-existing) |
| **Output Dir** | TBD - update with actual path |
| **Notes** | Joint MAE + Contrastive training baseline |

---

## Model Merging - Original (Pre-Sweep)

### Run 3: Initial Model Merging
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

## Implemented Merging Methods

**Code Location**: `src/merge_models.py`

| Method | Description | CLI Flag |
|--------|-------------|----------|
| `simple` | (MAE + Contrastive) / 2 | `--method simple` |
| `weighted` | α×MAE + (1-α)×Contrastive | `--method weighted --alpha 0.5` |
| `task_arithmetic` | base + λ×(τ_mae + τ_con) | `--method task_arithmetic --lambda_scale 1.0` |
| `layerwise` | Per-block α based on task-vector magnitudes | `--method layerwise` |
| `dare_ties` | Sparsify + TIES sign consensus | `--method dare_ties --dare_keep_ratio 0.2` |
| `fisher` | Fisher-weighted merge | `--method fisher --fisher_mae <path> --fisher_contrastive <path>` |
| `orthogonal` | Conflict-aware orthogonal | `--method orthogonal --ortho_beta 1.0 --ortho_tau 0.0` |

### Merging Methods Evaluation Status

| Method | Evaluated | Result |
|--------|-----------|--------|
| `simple` | ✅ | Works |
| `weighted` | ✅ | **Best** - α=0.0-0.3 optimal |
| `task_arithmetic` | ✅ | Works, weighted better |
| `layerwise` | ✅ Completed | Per-block α from task-vector magnitudes |
| `dare_ties` | ✅ | ❌ Non-functional (R@1 < 0.015) |
| `fisher` | ✅ **Best for Classification** | **47.41% mAP** 🏆 (Fisher Task-Vec) |
| `orthogonal` | ✅ | ❌ Non-functional (R@1 < 0.001) |

---

## TODO / Next Steps

### Immediate (✅ Retrieval Jobs Complete)
- [x] Run `./aggregate_retrieval_results.sh` to generate summary
- [x] Analyze results: identify top models by R@1
- [x] Compare orthogonal merge vs other methods
- [ ] **Key Insight**: Orthogonal merge and DARE-TIES both perform WORSE than simple weighted averaging
  - Orthogonal sweep (25 models): All R@1 < 0.001 (basically non-functional)
  - DARE-TIES sweep (12 models): All R@1 < 0.015 (non-functional)  - **Conclusion**: Weighted average with α=0.0-0.3 is optimal for merging MAE + Contrastive models
  - Contrastive-weighted models outperform complex merging strategies

### Next Steps (After MAE lr=2e-4 Training Completes - Job 313762)
- [ ] Re-run all sweep scripts with new MAE model:
  - `./sweep_orthogonal.sh`
  - `./sweep_weighted_full.sh`
  - `./sweep_task_arithmetic.sh`
  - `./sweep_dare_ties.sh`
- [ ] Launch retrieval evaluation for new merged models
- [ ] Compare lr=1e-4 vs lr=2e-4 MAE models

### Classification Finetuning (Top Models Only)
- [ ] Select top 5-10 merged models based on retrieval R@1
- [ ] Run classification finetuning on AudioSet-20K
- [ ] Script: `egs/audioset/run_cavmae_ft.sh`

### Fisher-Weighted Merge
- [x] Estimate Fisher diagonal for MAE-only checkpoint - Job 314662 🔄 Running
- [x] Estimate Fisher diagonal for Contrastive-only checkpoint - Job 314663 🔄 Running
- [x] Run Fisher-weighted merge - Job 314664 (pending Fisher estimation)
- [x] Script: `src/estimate_fisher.py`

### Layerwise Merge (NEW)
- [x] Implemented per-block alpha merge based on task-vector magnitudes
- [x] Created 2 models: `merged_layerwise_block.pth`, `merged_layerwise_param.pth`
- [x] Retrieval evaluation pending - Job 314665

---

## Useful Commands

```bash
# Check job status
squeue -u kqr867

# Watch specific job logs
tail -f egs/audioset/log/313696_mae_only_lr2e-4.txt

# Aggregate retrieval results (after jobs complete)
cd egs/audioset && ./aggregate_retrieval_results.sh

# Cancel a job
scancel <JOB_ID>

# Check job details
scontrol show job <JOB_ID>

# Run a single merge
python src/merge_models.py --method orthogonal \
    --model_mae ./exp/mae-only.../models/best_audio_model.pth \
    --model_contrastive ./exp/contrastive-only.../models/best_audio_model.pth \
    --model_base ./IN-initial.pth \
    --ortho_beta 1.0 --ortho_tau 0.0 \
    --use_contrastive_for_norms \
    --output ./exp/merged-models/merged_ortho.pth
```

---

## Results Summary

### VGGSound Retrieval Results (Verified)

| Model | A→V R@1 | A→V R@5 | A→V R@10 | V→A R@1 | V→A R@5 | V→A R@10 | MR A→V |
|-------|---------|---------|----------|---------|---------|----------|--------|
| Base (IN-initial) | 0.06% | 3.3% | 7.5% | 0.12% | 3.3% | 7.2% | 699 |
| **Contrastive-only (Ours)** | **16.0%** | **35.7%** | **44.9%** | **16.2%** | **37.4%** | **46.1%** | **15** |
| **Original Scale++** | 15.29% | 34.74% | 42.62% | 16.71% | 36.95% | 45.36% | 17 |
| Merged α=0.1 (Ours) | 13.70% | 33.04% | 42.97% | 14.99% | 34.52% | 44.21% | 17 |
| Original Scale+ | 11.46% | 27.40% | 36.10% | 14.12% | 31.64% | 40.00% | 30 |
| MAE-only (lr=1e-4) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Ranked Comparison (by Avg R@1)**:
| Rank | Model | A→V R@1 | V→A R@1 | Avg R@1 |
|------|-------|---------|---------|---------|
| 1 | **Contrastive-only (Ours)** | 16.0% | 16.2% | **16.1%** |
| 2 | **Original Scale++** | 15.3% | 16.7% | **16.0%** |
| 3 | Merged α=0.1 (Ours) | 13.7% | 15.0% | 14.4% |
| 4 | Original Scale+ | 11.5% | 14.1% | 12.8% |

**Key Findings**:
- Contrastive-only shows **267x improvement** in R@1 over base model (16.0% vs 0.06%)
- **Our Contrastive-only matches/exceeds Original Scale++** (16.0% vs 15.29% A→V R@1, tied at ~16% avg)
- Our Merged α=0.1 slots between Scale++ and Scale+ (14.4% avg vs 16.0% and 12.8%)
- Scale++ > Scale+ (larger batch size 256 vs 108 helps contrastive learning)
- **Pure contrastive training matches joint MAE+contrastive** - MAE doesn't help retrieval

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

### Sweep Results (✅ Completed)

**Complete Results Available**: `exp/retrieval_results/all_results_summary.csv` (128 rows)

**Top 10 Models by R@1 (Audio→Visual)**:
| Rank | Model | Method | R@1 | R@5 | R@10 | MR |
|------|-------|--------|-----|-----|------|-----|
| 1 | merged_weighted_alpha0.0 | Weighted | 16.0% | 35.7% | 44.9% | 15 |
| 2 | contrastive_only_lr1e-4 | Baseline | 16.0% | 35.7% | 44.9% | 15 |
| 3 | merged_weighted_alpha0.1 | Weighted | 13.7% | 33.0% | 43.0% | 17 |
| 4 | merged_weighted_alpha0.2 | Weighted | 9.5% | 26.2% | 34.6% | 28 |
| 5 | merged_weighted_alpha0.3 | Weighted | 4.9% | 16.0% | 22.6% | 65 |
| 6 | merged_weighted_0.3 | Weighted (old) | 4.9% | 16.0% | 22.6% | 65 |
| 7 | merged_weighted_alpha0.4 | Weighted | 2.2% | 6.7% | 11.4% | 151 |
| 8 | merged_weighted_alpha0.5 | Weighted | 1.1% | 2.8% | 4.8% | 269 |
| 9 | merged_weighted_0.5 | Weighted (old) | 1.1% | 2.8% | 4.8% | 269 |
| 10 | merged_task_arith_0.5 | Task Arith | 1.1% | 2.8% | 4.8% | 269 |

**Key Finding**: Best models are weighted toward contrastive (α=0.0-0.3), confirming hypothesis that MAE-only training learns different features unsuitable for retrieval when merged equally.

---

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
| MAE-only (lr=2e-4) | - | N/A | 🔄 Running (Job 313696) |
| Contrastive-only | N/A | 68% | ✅ Working well |

**Training Curves**: See `egs/audioset/training_curves_ablation.png`

**Retrieval Script**: `src/run_retrieval.py` (fixed for timm, autocast, float32 handling)

---

## Supervised Fine-Tuning (SFT) Experiments

### Run 7: SFT for Four Pretrained Models
| Model | Job ID | Status | Multi-frame mAP | Pretrain Path |
|-------|--------|--------|-----------------|---------------|
| **Fisher Task-Vec** | 314888 | ✅ Completed | **47.41%** 🏆 | `merged-models/fisher-sweep/merged_fisher_taskvec.pth` |
| **CAV-merged (α=0.1)** | 313821 | ✅ Completed | 47.14% | `merged-models/weighted-sweep/merged_weighted_alpha0.1.pth` |
| **MAE-only (lr=1e-4)** | 313822 | ✅ Completed | 44.32% | `mae-only-lr1e-4.../best_audio_model.pth` |
| **CAV-only (Contrastive-only)** | 313820 | ✅ Completed | 43.27% | `contrastive-only.../best_audio_model.pth` |
| **CAV-JEPA** | 313823 | ✅ Completed | 23.93% ❌ | `cavjepa-.../best_audio_model.pth` |

**SFT Results Analysis**:
- **Fisher Task-Vec is NEW BEST**: 47.41% mAP, beating weighted merge (47.14%) by +0.27%
- **MAE helps classification**: Merged models beat pure Contrastive (43.27%) by +4%
- **MAE-only recovers via SFT**: Despite collapsed norms during pretraining, achieves 44.32% mAP
- **CAV-JEPA underperforms**: Only 23.93% suggests bug in fine-tuning implementation - needs investigation

**Retrieval vs Classification Tradeoff**:
| Model | Retrieval R@1 (Avg) | Classification mAP |
|-------|---------------------|-------------------|
| Contrastive-only | **16.1%** | 43.27% |
| Fisher Task-Vec | TBD | **47.41%** |
| Merged α=0.1 | 14.4% | 47.14% |

**Insight**: MAE hurts retrieval but helps classification - different objectives optimize for different downstream tasks

---

### Run 8: Original CAV-MAE Models (Scale++ and Scale+) Evaluation

Downloaded and evaluating original pretrained models from Yuan Gong et al. (ICLR 2023) for fair comparison.

**Models Downloaded**:
| Model | Source URL | File | Size | Batch Size | λ_c |
|-------|-----------|------|------|------------|-----|
| Scale++ | dropbox/l5t5geufdy3qvnv | `cav-mae-scale++.pth` | 729MB | 256 | 0.01 |
| Scale+ | dropbox/xu8bfie6hz86oev | `cav-mae-scale+.pth` | 729MB | 108 | 0.01 |

**Retrieval Evaluation**:
| Model | Job ID | Status | Output |
|-------|--------|--------|--------|
| Scale++ | 313934 | ✅ Completed | `exp/retrieval_results/original_scalepp.csv` |
| Scale+ | 313935 | ✅ Completed | `exp/retrieval_results/original_scalep.csv` |

**Retrieval Results**:
| Model | A→V R@1 | A→V R@5 | A→V R@10 | V→A R@1 | V→A R@5 | V→A R@10 | A→V MR |
|-------|---------|---------|----------|---------|---------|----------|--------|
| **Scale++** | 15.29% | 34.74% | 42.62% | 16.71% | 36.95% | 45.36% | 17 |
| **Scale+** | 11.46% | 27.40% | 36.10% | 14.12% | 31.64% | 40.00% | 30 |

**SFT Evaluation**:
| Model | Job ID | Status | Exp Dir Pattern |
|-------|--------|--------|-----------------|
| Scale++ | 317993 (relaunched from 313936) | ✅ Completed | `sft-scalepp-original-5e-5-bs36-epoch15-20260204_140118/` |
| Scale+ | 317994 (relaunched from 313937) | ✅ Completed | `sft-scalep-original-5e-5-bs36-epoch15-20260204_140051/` |

**SFT Results**:
| Model | mAP | Notes |
|-------|-----|-------|
| **Scale+ (CAV-MAE)** | **49.93%** 🏆 | Best classification overall |
| **Scale++ (CAV-MAE++)** | **49.85%** | Near-best classification |

**Previous Failures (Jobs 313936, 313937)**:
- **Error**: `TypeError: __init__() got an unexpected keyword argument 'qk_scale'`
- **Root Cause**: timm library version incompatibility with Attention class
- **Fix**: Recreated scripts with correct configuration and node exclusions

**Scripts Created**:
- `egs/audioset/run_retrieval_scalepp.sh`
- `egs/audioset/run_retrieval_scalep.sh`
- `egs/audioset/run_sft_scalepp.sh` (recreated 2026-02-04)
- `egs/audioset/run_sft_scalep.sh` (recreated 2026-02-04)
- `egs/audioset/launch_original_eval.sh` (launcher)

**SFT Configuration** (all models):
- Learning rate: 5e-5
- Head LR multiplier: 100x
- Epochs: 15
- Batch size: 36
- Weight averaging: epochs 3-15
- Loss: BCE (multi-label classification)
- Metric: mAP
- Data augmentation: FreqM=48, TimeM=192, Mixup=0.5

**Scripts Created**:
- `egs/audioset/run_sft_cavonly.sh`
- `egs/audioset/run_sft_cavmerged.sh`
- `egs/audioset/run_sft_maeonly.sh`
- `egs/audioset/run_sft_cavjepa.sh`
- `egs/audioset/launch_all_sft_jobs.sh` (launcher)
- `src/run_cavjepa_ft.py` (CAV-JEPA fine-tuning entry point)
- `src/models/cav_jepa.py` (CAVJEPAFT model class)

**Note**: MAE-only model uses lr=1e-4 checkpoint (has collapsed LayerNorm issue). When lr=2e-4 training (Job 313762) completes, re-run with that checkpoint for comparison.

---

## Run 9: Layerwise and Fisher Merge Experiments (2026-01-20)

### Overview
Evaluating two previously implemented but untested merging methods:
1. **Layerwise merge**: Per-block α based on task-vector magnitudes
2. **Fisher-weighted merge**: Uses Fisher information diagonal for importance weighting

### Layerwise Merge
| Field | Value |
|-------|-------|
| **Job ID** | 314655 (merge), 314665 (retrieval eval) |
| **Status** | ✅ Merge completed, 🔄 Retrieval pending |
| **Output Dir** | `egs/audioset/exp/merged-models/layerwise-sweep/` |

**Algorithm**:
```
For each block/parameter group g:
  α_g = ||τ_mae|| / (||τ_mae|| + ||τ_con||)
  merged = base + α_g·τ_mae + (1-α_g)·τ_con
```

**Models Created**:
| Model | Grouping | Description |
|-------|----------|-------------|
| `merged_layerwise_block.pth` | Per-block | α computed per transformer block |
| `merged_layerwise_param.pth` | Per-parameter | α computed per individual parameter |

### Fisher-Weighted Merge
| Field | Value |
|-------|-------|
| **Job IDs** | 314662 (Fisher MAE), 314663 (Fisher Con), 314664 (merge), 314665 (eval) |
| **Status** | ✅ Fisher merge completed |
| **Output Dir** | `egs/audioset/exp/merged-models/fisher-sweep/` |

**Algorithm**:
```
merged = (F_mae·W_mae + F_con·W_con) / (F_mae + F_con)
With base model: merged = base + (F_mae·τ_mae + F_con·τ_con) / (F_mae + F_con)
```

**Fisher Estimation**:
- Using 100 batches from AudioSet eval set
- MAE model: `mae_loss_weight=1.0, contrast_loss_weight=0.0`
- Contrastive model: `mae_loss_weight=0.0, contrast_loss_weight=1.0`
- Output: `exp/fisher-estimates/fisher_mae.pth`, `exp/fisher-estimates/fisher_contrastive.pth`

**Models Created**:
| Model | Method | Description |
|-------|--------|-------------|
| `merged_fisher_direct.pth` | Direct weighting | F-weighted average of model weights |
| `merged_fisher_taskvec.pth` | Task vector weighting | F-weighted average of task vectors from base |

### Fisher Task-Vec SFT Experiment
| Field | Value |
|-------|-------|
| **Job ID** | 314888 (relaunched from failed 314746) |
| **Status** | ✅ **Completed - 47.41% mAP** 🏆 (NEW BEST!) |
| **Submitted** | 2026-01-21 15:44 |
| **Completed** | 2026-01-22 |
| **Node** | mlcbm009 |
| **Script** | `egs/audioset/run_sft_fisher_taskvec.sh` |
| **Output Dir** | `exp/sft-fisher-taskvec-5e-5-bs36-epoch15-20260121_154437/` |
| **Log File** | `egs/audioset/log/314888_sft_fisher_taskvec.txt` |
| **Pretrain Path** | `exp/merged-models/fisher-sweep/merged_fisher_taskvec.pth` |

**Result**: **47.41% mAP** - New best for classification task!

**Failed Attempt (Job 314746)**:
- **Failure Mode**: Ran on CPU instead of GPU
- **Root Cause**: Node mlcbm005 has broken CUDA detection (`torch.cuda.is_available()` returns False)
- **Epoch Time**: ~5000s (vs ~230s expected on GPU)
- **Fix Applied**: Added `#SBATCH --exclude=mlcbm005,mlcbm012` to script

**SFT Configuration**:
- Learning rate: 5e-5
- Head LR multiplier: 100x
- Epochs: 15
- Batch size: 36
- Weight averaging: epochs 3-15
- Loss: BCE (multi-label classification)
- Metric: mAP
- GPUs: 4 x H100

**Comparison with other models**:
| Model | mAP | Notes |
|-------|-----|-------|
| **Fisher Task-Vec** | **47.41%** | 🏆 NEW BEST |
| Merged α=0.1 | 47.14% | Previous best |
| MAE-only | 44.32% | - |
| Contrastive-only | 43.27% | - |

### Job Pipeline
| Phase | Job | ID | Status | Depends On |
|-------|-----|-----|--------|------------|
| **1a** | Layerwise merge | 314655 | ✅ Completed | - |
| **1b** | Fisher MAE estimation | 314662 | ✅ Completed | - |
| **1c** | Fisher Contrastive estimation | 314663 | ✅ Completed | - |
| **2** | Fisher merge | 314664 | ✅ Completed | 314662, 314663 |
| **3** | Retrieval evaluation (all 4 models) | 314665 | ✅ Completed | 314664 |
| **4** | Fisher Task-Vec SFT | 314888 | ✅ Completed (47.41% mAP) | 314664 |

### Scripts Created
- `egs/audioset/sweep_layerwise.sh` - Layerwise merge sweep
- `egs/audioset/run_estimate_fisher_mae.sh` - Fisher estimation for MAE model
- `egs/audioset/run_estimate_fisher_contrastive.sh` - Fisher estimation for Contrastive model
- `egs/audioset/sweep_fisher.sh` - Fisher merge sweep
- `egs/audioset/run_retrieval_layerwise_fisher.sh` - Retrieval evaluation for new models
- `egs/audioset/launch_layerwise_fisher_experiments.sh` - Master launcher

### Fix Applied
- Fixed `src/models/__init__.py`: Made `cav_jepa` import conditional (only exists on cav-mae-jepa branch)

---

## Run 10: CAV++ / MAE++ Pipeline (2026-02-05)

### Overview
Full post-training pipeline for independently trained Contrastive++ and MAE++ models with **larger batch size (256)** and **higher learning rate (2e-4)**.

**Key Insight**: Larger batch size (256 vs 120) significantly improves contrastive learning quality.

### Training Results

| Model | Job ID | Status | Final Metrics | Model Path |
|-------|--------|--------|---------------|------------|
| **Contrastive++** | 318035 | ✅ Completed | Loss=6.39, Acc=57.8% | `exp/contrastive++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/` |
| **MAE++** | 318036 | ✅ Completed | MAE Loss=3.30 | `exp/mae++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/` |

**Hyperparameters (both models)**:
- Learning rate: **2e-4** (higher than previous 1e-4)
- Batch size: **256** (vs 120 in original experiments)
- Epochs: 25
- Masking ratio: 0.75

### Retrieval Results (VGGSound)

#### Base Models
| Model | A→V R@1 | V→A R@1 | Avg R@1 |
|-------|---------|---------|---------|
| **Contrastive++** | **18.81%** | **18.04%** | **18.43%** |
| MAE++ | N/A | N/A | N/A (no contrastive heads) |

**This is the best retrieval result achieved** - 2.3% improvement over previous best (Scale++ at 16.0%).

#### Merged Models (Ranked by Avg R@1)
| Rank | Model | A→V R@1 | V→A R@1 | Avg R@1 |
|------|-------|---------|---------|---------|
| 1 | **merged_weighted_0.0** | 18.81% | 18.04% | **18.43%** |
| 2 | merged_weighted_0.1 | 16.40% | 16.37% | 16.39% |
| 3 | **merged_fisher** | 11.41% | 12.56% | **11.98%** |
| 4 | merged_weighted_0.2 | 10.23% | 10.66% | 10.45% |
| 5 | merged_weighted_0.3 | 4.47% | 4.61% | 4.54% |
| 6 | merged_weighted_0.4 | 1.35% | 2.13% | 1.74% |
| 7 | merged_layerwise | 1.23% | 1.67% | 1.45% |
| 8 | merged_weighted_0.5 / simple / task_arith_0.5 | 0.53% | 0.86% | 0.70% |

**Key Observations**:
1. **Pure Contrastive++ (α=0.0) is best** - 18.43% avg R@1
2. **Fisher merge (11.98%) outperforms equal weighting** at similar ratios
3. **MAE contribution hurts retrieval** - expected since MAE features aren't trained for alignment

### SFT Experiments

**Jobs Submitted**:
| Model | Job ID | Status | mAP |
|-------|--------|--------|-----|
| merged_weighted_0.0 (Contrastive++) | 318178 | ✅ Completed | 44.14% |
| merged_weighted_0.1 | 318179 | ✅ Completed | 44.61% |
| merged_fisher | 318180 | ✅ Completed | 44.61% |

**Note**: Did not beat baseline mAP of 47.41% (Fisher Task-Vec from base experiments) or paper checkpoints (~49.9%).

### Scripts Created
| Script | Purpose |
|--------|---------|
| `run_retrieval_plusplus.sh` | Retrieval eval for base models |
| `run_estimate_fisher_plusplus.sh` | Fisher estimation |
| `run_merge_plusplus.sh` | Create merged models |
| `run_retrieval_merged_plusplus.sh` | Retrieval eval for merged models |
| `run_sft_merged_plusplus.sh` | SFT training template |
| `run_pipeline_plusplus.sh` | Master orchestration script |

### Results Summary

**Retrieval Improvement from Original Experiments**:
| Model | Original (bs=120) | ++ (bs=256) | Improvement |
|-------|-------------------|-------------|-------------|
| Contrastive-only | 16.1% | **18.43%** | **+2.33%** |
| Merged α=0.1 | 14.4% | 16.39% | +1.99% |

---

## Run 11: Comprehensive Pareto Analysis & Fine-Grained Experiments (2026-02-05/06)

### Overview
Systematic evaluation of all models on both **classification (mAP)** and **retrieval (R@1)** to find models that excel at both tasks. Created Pareto frontier plots by training scale.

### Code Review & Norm Fix Rerun

**Issue**: Basic merged models (Jan 16) were created WITHOUT `--use_contrastive_for_norms` flag. MAE norm layers were collapsed.

**Fix Applied**:
- Regenerated all 10 basic merged models via `merge_models.sh` (now includes the fix)
- Re-ran retrieval evaluation (Job 318193) ✅ Completed
- Results unchanged: norm fix is cosmetic for retrieval (L2 normalization cancels uniform scaling)
- Also fixed copy-paste bug in `run_retrieval_merged.sh`: section 6 (λ=1.0) was outputting to `merged_task_arith_1.5.csv`

### Focused SFT Experiments (Job 318226)
| Field | Value |
|-------|-------|
| **Job ID** | 318226 |
| **Status** | ✅ Completed |
| **Submitted** | 2026-02-05 22:10 |
| **Completed** | 2026-02-06 02:06 |
| **Script** | `egs/audioset/run_sft_promising.sh` |

**Results (only models with R@1 ≥ 10%)**:
| Model | Scale | R@1% | mAP% | Status |
|-------|-------|------|------|--------|
| Contrastive++ | ++ | 18.81% | 43.97% | ✅ New |
| Fisher Direct | Base | 12.76% | 46.02% | ✅ New |
| Merged α=0.2 | ++ | 10.23% | 44.27% | ✅ New |

### Fine-Grained Weighted Sweep (Job 318225)
| Field | Value |
|-------|-------|
| **Job ID** | 318225 |
| **Status** | ✅ Completed |
| **Submitted** | 2026-02-05 22:10 |
| **Script** | `egs/audioset/run_retrieval_finegrained.sh` |

**New Models Created & Evaluated** (Scale++, filling gap between α=0.0 and α=0.2):
| Model | R@1 (A→V) | R@1 (V→A) | mAP% | Notes |
|-------|-----------|-----------|------|-------|
| merged_weighted_0.05 | 17.81% | 17.75% | 44.81% | ✅ New sweet spot |
| merged_weighted_0.15 | 13.40% | 13.49% | 44.26% | ✅ Complete |

### Dead Ends Identified (R@1 < 10%)
These methods produce models with poor retrieval and are **not worth further exploration**:
- **Task Arithmetic** (all λ values): R@1 ≈ 0-1%
- **Orthogonal merging** (25 variants): R@1 < 0.1%
- **DARE-TIES** (12 variants): R@1 < 1.5%
- **Layerwise merging**: R@1 ≈ 1%
- **Weighted α ≥ 0.3**: R@1 < 5% (retrieval drops off a cliff)

### Complete Pareto Frontier (R@1 ≥ 10%)

| Model | Scale | R@1% | mAP% | Pareto? |
|-------|-------|------|------|---------|
| Merged α=0.0 | ++ | 18.81% | 44.14% | ✅ |
| **Merged α=0.05** | **++** | **17.81%** | **44.81%** | ✅ |
| Merged α=0.1 | ++ | 16.40% | 44.61% | No |
| Contrastive Only | Base | 15.99% | 43.00% | No |
| **CAV-MAE++ (paper)** | **Orig** | **15.29%** | **49.85%** | ✅ |
| Fisher TaskVec | Base | 15.29% | 45.54% | No |
| Merged α=0.15 | ++ | 13.40% | 44.26% | No |
| Merged α=0.1 | Base | 13.70% | 45.26% | No |
| Fisher Direct | Base | 12.76% | 46.02% | No |
| CAV-MAE (paper) | Orig | 11.46% | 49.93% | ✅ |
| Fisher Merge | ++ | 11.41% | 44.61% | No |
| Merged α=0.2 | ++ | 10.23% | 44.27% | No |

**Key Gap**: 5% mAP gap between our best merged models (~44.8%) and paper checkpoints (~49.9%).

### Plots Generated
- `exp/pareto_plot.png` / `.pdf` - Combined Pareto plot
- `exp/pareto_by_scale.png` / `.pdf` - Three subplots by scale
- `exp/pareto_base.png` / `.pdf` - Base scale individual
- `exp/pareto_plusplus.png` / `.pdf` - Scale++ individual

---

## Run 12: Sequential Training Experiments (2026-02-06)

### Overview
**Hypothesis**: Sequential training (one objective then the other) can combine benefits of both MAE and contrastive objectives better than post-hoc model merging.

### Run 12a: MAE → CAV (Sequential)
| Field | Value |
|-------|-------|
| **Job ID** | 318234 |
| **Status** | 🔄 Running |
| **Submitted** | 2026-02-06 |
| **Script** | `egs/audioset/run_sequential_mae_then_cav.sh` |
| **Start Model** | MAE++ pretrained (bs=256, lr=2e-4) |
| **Training** | 5 epochs contrastive-only (mae_loss=0, contrast_loss=1.0) |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 256 |

**Hypothesis**: MAE gives rich features for classification; adding contrastive training adds cross-modal alignment for retrieval without destroying feature quality.

### Run 12b: CAV → MAE (Sequential)
| Field | Value |
|-------|-------|
| **Job ID** | 318235 |
| **Status** | 🔄 Running |
| **Submitted** | 2026-02-06 |
| **Script** | `egs/audioset/run_sequential_cav_then_mae.sh` |
| **Start Model** | Contrastive++ pretrained (bs=256, lr=2e-4) |
| **Training** | 5 epochs MAE-only (mae_loss=1.0, contrast_loss=0) |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 256 |

**Hypothesis**: Contrastive gives alignment (should preserve retrieval); adding MAE training adds feature richness for classification.

### Expected Outcomes
- **MAE→CAV**: Should gain retrieval (from ~0% → something meaningful). If mAP stays near 44-45%, this bridges the gap.
- **CAV→MAE**: Should gain mAP while losing some retrieval. Key question: how much of each?

### Next Steps After Completion
1. Run retrieval evaluation on both resulting models
2. If R@1 ≥ 10%, run SFT for classification
3. If promising, try more epochs (10, 15) and different LRs (5e-5, 2e-4)

---

## Cluster Notes

### Known Problematic Nodes
| Node | Issue | Status |
|------|-------|--------|
| **mlcbm005** | CUDA detection fails (`torch.cuda.is_available()` returns False), jobs run on CPU | CRITICAL - Always exclude |
| **mlcbm012** | Intermittent GPU issues | Exclude recommended |

**Always add to GPU job scripts:**
```bash
#SBATCH --exclude=mlcbm005,mlcbm012
```

---

## Changelog

- **2026-02-06**: Sequential training experiments & Pareto analysis:
  - **Pareto Analysis**: Comprehensive evaluation of all models on both classification (mAP) and retrieval (R@1)
    - Created Pareto frontier plots by scale (base, original, scale++)
    - Identified dead ends: Task Arithmetic, Orthogonal, DARE-TIES, Layerwise (all R@1 < 10%)
    - Pareto-optimal models: Merged α=0.0 (18.8% R@1), α=0.05 (17.8% R@1, 44.8% mAP), CAV-MAE++ paper (15.3% R@1, 49.9% mAP)
  - **Fine-Grained Weighted Sweep** (Job 318225): Created α=0.05, 0.15 for Scale++
    - α=0.05: R@1=17.81%, mAP=44.81% (new Pareto-optimal point)
    - α=0.15: R@1=13.40%, mAP=44.26%
  - **Focused SFT** (Job 318226): Only models with R@1 ≥ 10%
    - Contrastive++: mAP=43.97%, Fisher Direct: mAP=46.02%, Merged α=0.2 (++): mAP=44.27%
  - **Sequential Training** (Jobs 318234, 318235): New direction
    - MAE++ → 5ep CAV (lr=1e-4): Running
    - CAV++ → 5ep MAE (lr=1e-4): Running
  - Scale+ SFT (Job 317994): ✅ Completed - **49.93% mAP** (best overall)
  - Scale++ SFT (Job 317993): ✅ Completed - **49.85% mAP**
  - Regenerated basic merged models with `--use_contrastive_for_norms` fix (Job 318193)
- **2026-02-05**: CAV++ / MAE++ Post-Training Pipeline:
  - **Phase 1 Completed**: Trained Contrastive++ and MAE++ models with bs=256, lr=2e-4, 25 epochs
    - Contrastive++ (Job 318035): Loss=6.39, Acc=57.8%
    - MAE++ (Job 318036): MAE Loss=3.30
  - **Phase 2 Completed**: Retrieval evaluation on base models (Job 318140)
    - Contrastive++ achieves **18.43% avg R@1** (BEST retrieval result ever!)
    - MAE++ outputs zeros (expected - no contrastive heads)
  - **Phase 3 Completed**: Fisher estimation (Job 318142)
    - Created `fisher_mae++.pt` and `fisher_contrastive++.pt`
  - **Phase 4 Completed**: Model merging (Job 318170)
    - Created 20 merged models: simple, weighted (0.0-1.0), task_arith, fisher, orthogonal, layerwise
  - **Phase 5 Completed**: Retrieval on merged models (Job 318173)
    - Best: `merged_weighted_0.0` (pure Contrastive++) = 18.43% R@1
    - Fisher merge = 11.98% R@1 (better than equal weighting)
  - **Phase 6 Running**: SFT on top 3 models (Jobs 318178, 318179, 318180)
    - `merged_weighted_0.0` (pure Contrastive++)
    - `merged_weighted_0.1` (10% MAE)
    - `merged_fisher` (Fisher-weighted)
- **2026-02-04**: Fixed and relaunched failed experiments:
  - **Fisher Task-Vec SFT (Job 314888)**: ✅ **COMPLETED - 47.41% mAP** (NEW BEST for classification!)
  - **MAE lr=2e-4 Pretraining (Job 317992)**: Relaunched (was Job 313762, failed at epoch 17/25 due to TIME LIMIT)
    - Fixed script: Added `#SBATCH --exclude=mlcbm005,mlcbm012`
  - **Scale++ SFT (Job 317993)**: Relaunched (was Job 313936, failed due to `qk_scale` timm error)
    - Recreated `run_sft_scalepp.sh` with proper configuration
  - **Scale+ SFT (Job 317994)**: Relaunched (was Job 313937, failed due to `qk_scale` timm error)
    - Recreated `run_sft_scalep.sh` with proper configuration
  - Updated experiment log with Fisher Task-Vec results and new job IDs
- **2026-01-21 15:45**: Launched Fisher Task-Vec SFT experiment:
  - Job 314746 failed - ran on CPU due to broken CUDA on mlcbm005
  - Relaunched as Job 314888 with `--exclude=mlcbm005,mlcbm012`
  - Now running on mlcbm009 with CUDA
  - Added cluster notes documenting problematic nodes
  - Created `run_sft_fisher_taskvec.sh` script
- **2026-01-20 12:30**: Launched Layerwise and Fisher merge experiments:
  - Layerwise merge completed (Job 314655): Created 2 models (`merged_layerwise_block.pth`, `merged_layerwise_param.pth`)
  - Fisher estimation running (Jobs 314662, 314663): Estimating Fisher diagonals for MAE and Contrastive models
  - Fisher merge pending (Job 314664): Will create 2 models after estimation completes
  - Retrieval evaluation pending (Job 314665): Will evaluate all 4 new models
  - Fixed `src/models/__init__.py` - cav_jepa import was failing on merge branch
  - Created 6 new scripts for layerwise/Fisher experiments
- **2026-01-17 16:30**: SFT experiments completed (Jobs 313820-313823):
  - **CAV-merged α=0.1: 47.14% mAP** (best) - MAE helps classification!
  - MAE-only: 44.32% mAP - recovers despite norm collapse
  - CAV-only: 43.27% mAP
  - CAV-JEPA: 23.93% mAP ❌ - needs investigation
  - **Key insight**: MAE hurts retrieval but helps classification (+3.87% mAP)
- **2026-01-17 16:15**: Updated comprehensive retrieval comparison table:
  - Added Merged α=0.1 V→A results (R@1=14.99%, already in all_results_summary.csv)
  - Created ranked comparison by average R@1: Contrastive-only (16.1%) ≈ Scale++ (16.0%) > Merged α=0.1 (14.4%) > Scale+ (12.8%)
  - **Insight**: Pure contrastive training matches joint MAE+contrastive for retrieval
- **2026-01-17 16:00**: Added Original CAV-MAE (Scale++ and Scale+) evaluation:
  - Downloaded models from Dropbox (729MB each, verified complete)
  - Retrieval completed: Scale++ R@1=15.29% A→V, Scale+ R@1=11.46% A→V
  - **Key finding**: Our Contrastive-only (16.0%) matches/exceeds Original Scale++ (15.29%)
  - SFT jobs pending (313936, 313937) - waiting for GPU resources
  - Created scripts: `run_retrieval_scalepp.sh`, `run_sft_scalepp.sh`, `launch_original_eval.sh`
- **2026-01-17 08:00**: Launched 4 SFT experiments for pretrained models:
  - CAV-only (Job 313820) - Running
  - CAV-merged α=0.1 (Job 313821) - Pending
  - MAE-only lr=1e-4 (Job 313822) - Pending
  - CAV-JEPA (Job 313823) - Pending
  - Created `src/run_cavjepa_ft.py` for JEPA fine-tuning
  - Added `CAVJEPAFT` model class to src/models/
- **2026-01-17 07:00**: Retrieval evaluation jobs ALL COMPLETED (128 results):
  - Jobs 313763-313819 completed successfully
  - Aggregated results: `all_results_summary.csv` generated
  - **Key Finding**: Weighted merging with α=0.0-0.3 (contrastive-heavy) optimal; orthogonal and DARE-TIES merges non-functional
  - Top model: merged_weighted_alpha0.0 matches contrastive-only baseline (R@1=16.0% A→V, 16.2% V→A)
  - Orthogonal sweep (25 models): All R@1 < 0.001 (worse than weighted averaging)
  - DARE-TIES sweep (12 models): All R@1 < 0.015 (worse than weighted averaging)
- **2026-01-17 06:41**: Fixed and relaunched all jobs:
  - Job 313696 (MAE training): Failed due to missing `wandb` - installed and relaunched as Job 313762
  - Jobs 313698-313754 (retrieval): Failed due to relative paths - fixed with absolute paths and relaunched as Jobs 313763-313819
  - Results already coming in: 13 orthogonal sweep jobs completed
- **2026-01-16 23:26**: Launched 57 parallel retrieval evaluation jobs (313698-313754) for all merged models
- **2026-01-16 23:26**: Re-submitted MAE-only lr=2e-4 training (Job 313696) after timm fix
- **2026-01-16 23:23**: Completed all merge sweeps - 55 total merged models created
- **2026-01-16 23:14**: Created DARE-TIES sweep (12 models)
- **2026-01-16 23:11**: Created orthogonal sweep (25 models), weighted sweep (11 models), task arithmetic sweep (7 models)
- **2026-01-16 22:37**: Implemented orthogonal conflict-aware merge in `src/merge_models.py`
- **2026-01-16 22:23**: Fixed timm compatibility in `src/models/cav_mae.py` (removed qk_scale from Attention)
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
- **2026-01-16**: Launched MAE-only lr=2e-4 (Job 313261) to test higher LR hypothesis - failed timm issue
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
