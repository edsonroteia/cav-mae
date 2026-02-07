# CAV-MAE Experiment Plan

## Goal
Find models that excel at **both** classification (mAP) and retrieval (R@1).

## Current Status (Feb 7, 2026)

### Best Models (R@1 >= 10%)
| Model | Scale | R@1% | mAP% | Status |
|-------|-------|------|------|--------|
| merged_weighted_0.0 / contrastive++ | ++ | 18.4% | 44.1% / 44.0% | Complete |
| **merged_weighted_0.05** | **++** | **17.8%** | **44.8%** | **Complete (Pareto optimal)** |
| merged_weighted_0.1 | ++ | 16.4% | 44.6% | Complete |
| contrastive_only | Base | 16.1% | 43.0% | Complete |
| **CAV-MAE++ (paper)** | **Original** | **16.0%** | **49.9%** | **Complete (Pareto optimal)** |
| Fisher TaskVec | Base | 15.4% | 45.6% | Complete |
| merged_weighted_alpha0.1 | Base | 14.3% | 45.5% | Complete |
| merged_weighted_0.15 | ++ | 13.4% | 44.3% | Complete |
| Fisher Direct | Base | 13.4% | 46.1% | Complete |
| **CAV-MAE (paper)** | **Original** | **12.8%** | **50.0%** | **Complete (Pareto optimal)** |
| merged_fisher | ++ | 12.0% | 44.6% | Complete |
| merged_weighted_0.2 | ++ | 10.4% | 44.4% | Complete |
| weighted_alpha0.0 (=contrastive base) | Base | 16.1% | 42.8% | Complete |

### Pareto-Optimal Models
1. **Merged alpha=0.0 (++)**: 18.4% R@1, 44.1% mAP (best retrieval)
2. **Merged alpha=0.05 (++)**: 17.8% R@1, 44.8% mAP (best merged trade-off)
3. **CAV-MAE++ (paper)**: 16.0% R@1, 49.9% mAP (best joint-trained)
4. **CAV-MAE (paper)**: 12.8% R@1, 50.0% mAP (best classification)

**Key Gap**: ~5% mAP between our best merged models (~44.8%) and paper's joint training (~49.9%).

### Dead Ends (R@1 < 10%)
- Task Arithmetic (all lambda values)
- Orthogonal merging
- DARE-TIES
- Layerwise merging
- Weighted alpha >= 0.3

---

## Experiment Directions

### Direction 1: Fine-Grained Weighted Sweep - COMPLETED
**Hypothesis:** Sweet spot exists between alpha=0.1 (16.4%) and alpha=0.2 (10.2%)

**Status:** COMPLETED

**Results:**
| alpha | R@1% | mAP% | Notes |
|-------|------|------|-------|
| 0.00 | 18.4% | 44.1% | Pure contrastive |
| **0.05** | **17.8%** | **44.8%** | **New Pareto-optimal** |
| 0.10 | 16.4% | 44.6% | - |
| 0.15 | 13.4% | 44.3% | - |
| 0.20 | 10.4% | 44.4% | - |

**Conclusion:** alpha=0.05 gives the best retrieval-classification trade-off among merged models. mAP plateaus around 44-45% regardless of alpha -- merging cannot close the 5% gap to joint training.

---

### Direction 2: Sequential Training - FAILED
**Hypothesis:** Sequential training can combine benefits of both objectives

**Status:** FAILED (both directions)

#### 2a: MAE -> CAV (Job 318234)
- **Result:** Loss stuck at 16.6, cancelled
- MAE pretrained model couldn't learn contrastive alignment
- The MAE features are too specialized for reconstruction

#### 2b: CAV -> MAE (Job 318235)
- **Result:** Eval loss flat at 7.578 across all 5 epochs
- Adding MAE training on top of contrastive didn't improve features
- MAE reconstruction is too different an objective to be learned on top of contrastive

**Conclusion:** Sequential training doesn't work -- the objectives create fundamentally different feature spaces that resist sequential combination.

---

### Direction 3: Fisher Variants - COMPLETED
**Hypothesis:** Fisher weighting could be improved

**Results:**
- Fisher TaskVec (Base): 15.4% R@1, 45.6% mAP (best base-scale merge)
- Fisher Direct (Base): 13.4% R@1, 46.1% mAP
- Fisher Merge (++): 12.0% R@1, 44.6% mAP

**Conclusion:** Fisher helps for base-scale merges (best mAP among base-scale at 46.1%), but doesn't close the gap for Scale++ models. Fisher TaskVec on Scale++ not attempted (would likely give similar ceiling as weighted).

---

### Direction 4: MAE lr=2e-4 - NOT WORTH MERGING
**Hypothesis:** Higher LR for MAE-only training produces better features for merging

**Results:**
- MAE lr=2e-4: eval loss = 3.308
- MAE lr=1e-4: eval loss = 3.302
- Nearly identical (0.2% difference)

**Conclusion:** Higher LR doesn't meaningfully change MAE training outcome. The issue isn't MAE quality -- it's the fundamental incompatibility of MAE and contrastive feature spaces.

---

### Direction 5: Layer-Wise Alpha (Per-Block Different alpha) - NEW
**Hypothesis:** Different transformer layers may have diverged differently between MAE and contrastive training. Early layers (processing raw input) may be similar, while later layers (task-specific) may differ more.

**Approach:**
1. Compute CKA similarity or weight distance per transformer block between MAE and contrastive models
2. Use high alpha (more contrastive) for layers with high divergence
3. Use low alpha (more averaged) for layers with low divergence
4. This is more principled than uniform alpha and may preserve retrieval while gaining mAP

**Experiments:**
- [ ] Compute per-block CKA between contrastive++ and MAE++ models
- [ ] Design alpha schedule based on CKA (e.g., alpha_i = 1 - CKA_i)
- [ ] Merge with per-block alpha and evaluate

**Rationale:** The flat layerwise merge (Direction 3) used task-vector magnitudes which didn't work well. CKA measures representational similarity more directly.

---

### Direction 6: Merge at Earlier Training Checkpoints - NEW
**Hypothesis:** The ++ models are trained for 25 epochs. At epoch 25, the MAE and contrastive models have fully specialized. Earlier checkpoints (epoch 5, 10, 15) may not have diverged as much, producing better merges.

**Approach:**
1. Load MAE++ and contrastive++ checkpoints from epochs 5, 10, 15
2. Merge at each epoch with alpha=0.05 (current best) and alpha=0.1
3. Evaluate retrieval and classification

**Experiments:**
- [ ] Check if intermediate checkpoints were saved during training
- [ ] Merge at epoch 5, 10, 15 with alpha=0.05
- [ ] Evaluate retrieval R@1 and SFT mAP for each

**Rationale:** If models are more similar early in training, merging them may better preserve both objectives' features.

---

### Direction 7: Joint Training with Tuned Loss Weights - HIGHEST PRIORITY
**Hypothesis:** Paper uses c=0.01, mae=1.0 (MAE dominates). Increasing contrastive weight could improve retrieval while maintaining classification.

**This is the only approach guaranteed to close the gap** -- it IS joint training, just with different loss weights.

**Experiments:**
- [ ] c=0.05, mae=1.0 (5x more contrastive than paper)
- [ ] c=0.1, mae=1.0 (10x more contrastive than paper)
- [ ] c=0.01, mae=0.5 (weaken MAE to let contrastive shine)
- [ ] c=0.05, mae=0.5 (balanced reweighting)

**Training config:**
- Batch size: 256 (same as ++ models)
- Learning rate: 2e-4
- Epochs: 25
- Masking: 0.75

**Expected outcome:** Higher contrastive weight should improve retrieval R@1 beyond 16% (paper's result) while keeping mAP near ~49%. This directly addresses the 5% gap.

---

### Direction 8: Merge + Continued Joint Training - NEW
**Hypothesis:** Use best merged model as initialization for joint training. The merged model already has both contrastive alignment (for retrieval) and MAE features (for classification), so a few epochs of joint training could refine it.

**Approach:**
1. Start from best merged model (alpha=0.05 ++)
2. Train with both objectives (c=0.01, mae=1.0) for 3-5 epochs
3. Evaluate both retrieval and classification

**Experiments:**
- [ ] Joint training from merged alpha=0.05, 3 epochs
- [ ] Joint training from merged alpha=0.05, 5 epochs
- [ ] Try with higher contrastive weight (c=0.05) for 3 epochs

**Rationale:** Joint training from scratch takes 25 epochs. Starting from a merged model that already has both capabilities should converge faster and potentially find a better optimum.

---

## Summary of Explored vs Remaining

### Exhausted/Failed Approaches
| Approach | Best Result | Why It Failed |
|----------|------------|---------------|
| Weighted merging | alpha=0.05: 17.8% R@1, 44.8% mAP | mAP ceiling at ~45% |
| Fisher merging | 15.4% R@1, 45.6% mAP | Same mAP ceiling |
| Sequential training | Both directions failed | Objectives create incompatible features |
| Task Arithmetic | R@1 < 1% | Non-functional |
| Orthogonal merging | R@1 < 0.1% | Non-functional |
| DARE-TIES | R@1 < 1.5% | Non-functional |
| Layerwise merging | R@1 ~ 1% | Non-functional |
| MAE lr=2e-4 | Same as lr=1e-4 | LR not the bottleneck |

### Remaining Approaches (Priority Order)
1. **Direction 7: Joint training with tuned loss weights** (HIGHEST - guaranteed to work)
2. **Direction 8: Merge + continued joint training** (HIGH - may converge faster)
3. **Direction 5: Per-block CKA-guided alpha** (MEDIUM - more principled merging)
4. **Direction 6: Earlier checkpoint merging** (LOW - depends on checkpoint availability)

---

## Evaluation Protocol

For each model:
1. **Retrieval** (VGGSound): `python src/run_retrieval.py --model_path MODEL.pth ...`
2. **Classification** (AudioSet SFT): `sbatch run_sft_*.sh MODEL_NAME`

Only proceed to SFT if retrieval R@1 >= 10%.

---

## File Structure

```
egs/audioset/
├── EXPERIMENT_PLAN.md              # This file
├── plot_pareto.py                 # Pareto frontier plotting
├── run_retrieval_finegrained.sh   # Fine-grained weighted retrieval
├── run_sft_promising.sh           # SFT for promising models only
├── run_sequential_mae_then_cav.sh # Sequential: MAE -> CAV (FAILED)
├── run_sequential_cav_then_mae.sh # Sequential: CAV -> MAE (FAILED)
├── exp/
│   ├── merged-models-plusplus/    # Scale++ merged models
│   ├── retrieval_results/         # All retrieval CSVs
│   ├── sft-*/                     # SFT experiment outputs
│   ├── pareto_combined.png/pdf    # Combined Pareto plot
│   └── pareto_by_scale.png/pdf    # Per-scale Pareto plots
```
