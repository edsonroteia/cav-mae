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
| **Job ID** | 313762 (relaunched from 313696) |
| **Status** | 🔄 Running (~49 min elapsed) |
| **Submitted** | 2026-01-17 06:41 |
| **Failed Attempt** | 313696 - Failed due to missing `wandb` package |
| **Script** | `egs/audioset/run_mae_only_lr2e-4.sh` |
| **Output Dir** | `egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr2e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/` |
| **Log File** | `egs/audioset/log/313762_mae_only_lr2e-4.txt` |
| **Node** | mlcbm004 |

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
| `fisher` | Fisher-weighted merge | `--method fisher` |
| `orthogonal` | **NEW** Conflict-aware orthogonal | `--method orthogonal --ortho_beta 1.0 --ortho_tau 0.0` |

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
- [ ] Estimate Fisher diagonal for MAE-only checkpoint
- [ ] Estimate Fisher diagonal for Contrastive-only checkpoint
- [ ] Run Fisher-weighted merge
- [ ] Script: `src/estimate_fisher.py`

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

## Changelog

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
