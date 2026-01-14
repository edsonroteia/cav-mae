# CAV-MAE Model Merging Experimentation Plan

## Research Question
Can training MAE and Contrastive objectives **separately** and then **merging** the models match or exceed joint multi-task training?

---

## 1. Model Merging Methods

Based on recent literature ([NVIDIA Blog](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/), [Model Merging Survey](https://cameronrwolfe.substack.com/p/model-merging), [Awesome Model Merging](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications)):

### 1.1 Implemented Methods (in `src/merge_models.py`)
| Method | Formula | Description |
|--------|---------|-------------|
| Simple Averaging | `(M_mae + M_con) / 2` | Equal weight baseline |
| Weighted Averaging | `α·M_mae + (1-α)·M_con` | Tunable interpolation |
| Task Arithmetic | `M_base + λ·(τ_mae + τ_con)` | Task vector addition |

### 1.2 Future Extensions (Not in Scope)
For reference, these methods could be explored later:
- TIES-Merging, DARE, DARE-TIES, Greedy Soup

---

## 2. Experimental Setup

### 2.1 Base Models to Train

| Model | Training Objective | Script |
|-------|-------------------|--------|
| **M_joint** | MAE + Contrastive (baseline) | `run_cavmae_pretrain.sh` |
| **M_mae** | MAE only | `run_cavmae_pretrain_mae_only.sh` |
| **M_con** | Contrastive only | `run_cavmae_pretrain_contrastive_only.sh` |
| **M_base** | Initial checkpoint (IN-initial.pth) | - |

### 2.2 Datasets
- **Pretraining**: VGGSound (or AudioSet-2M)
- **Finetuning**: VGGSound, AudioSet-20K (balanced)

### 2.3 Training Configuration
Keep consistent across all runs:
- `masking_ratio=0.75`
- `lr=1e-4`
- `epoch=25`
- `batch_size=120`
- Same random seeds where possible

---

## 3. Hyperparameter Sweep

### 3.1 Weighted Averaging (α)
```
α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
```
- α=0.0 → 100% Contrastive
- α=1.0 → 100% MAE

### 3.2 Task Arithmetic (λ)
```
λ ∈ {0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0}
```
Per [Task Arithmetic paper](https://arxiv.org/abs/2212.04089), λ controls the strength of task vectors.

---

## 4. Evaluation Protocol

### 4.1 Downstream Tasks

Based on [VGGSound](https://www.robots.ox.ac.uk/~vgg/publications/2020/Chen20/chen20.pdf) and [VGGSounder](https://vggsounder.github.io/) benchmarks:

#### A. Audio-Visual Classification
| Dataset | Metric | Modality |
|---------|--------|----------|
| VGGSound | Top-1 Accuracy, Top-5 Accuracy | Multimodal, Audio-only, Visual-only |
| AudioSet-20K | mAP, AUC | Multimodal, Audio-only, Visual-only |

#### B. Audio-Visual Retrieval
| Task | Metrics |
|------|---------|
| Audio → Visual | R@1, R@5, R@10 |
| Visual → Audio | R@1, R@5, R@10 |

#### C. Reconstruction Quality (for MAE objective)
| Metric | Description |
|--------|-------------|
| MSE | Reconstruction error on held-out data |
| Qualitative | Visual inspection of inpainted samples |

### 4.2 Evaluation Matrix

For each merged model, evaluate:
```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Model           │ Multimodal   │ Audio-only   │ Visual-only  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ M_joint         │ ✓ (baseline) │ ✓            │ ✓            │
│ M_mae           │ ✓            │ ✓            │ ✓            │
│ M_con           │ ✓            │ ✓            │ ✓            │
│ Merged (each)   │ ✓            │ ✓            │ ✓            │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 5. Experiment Phases

### Phase 1: Baseline Training
1. Train M_joint (joint MAE + Contrastive)
2. Train M_mae (MAE-only)
3. Train M_con (Contrastive-only)
4. Evaluate all three on downstream tasks

### Phase 2: Basic Merging
1. Simple averaging
2. Weighted averaging sweep (5 values of α)
3. Task arithmetic sweep (7 values of λ)
4. Evaluate all merged models

### Phase 3: Analysis
1. Compare all methods
2. Statistical significance tests
3. Ablation studies
4. Visualization of results

---

## 6. Expected Results Table

| Method | VGGSound Acc | AudioSet mAP | Retrieval R@1 | Notes |
|--------|-------------|--------------|---------------|-------|
| Joint Training (baseline) | ~65.8% | ~42.0 | TBD | Current SOTA |
| MAE-only | ? | ? | Low (no contrastive) | |
| Contrastive-only | ? | ? | High | No reconstruction |
| Simple Average | ? | ? | ? | |
| Weighted (best α) | ? | ? | ? | |
| Task Arithmetic (best λ) | ? | ? | ? | |

---

## 7. Implementation Checklist

### 7.1 Already Implemented
- [x] `src/merge_models.py` - Simple, Weighted, Task Arithmetic
- [x] `egs/vggsound/run_cavmae_pretrain_mae_only.sh`
- [x] `egs/vggsound/run_cavmae_pretrain_contrastive_only.sh`
- [x] `egs/vggsound/merge_models.sh`

### 7.2 Scripts to Create
- [ ] `egs/vggsound/run_sweep_weighted.sh` - α sweep
- [ ] `egs/vggsound/run_sweep_task_arith.sh` - λ sweep
- [ ] `egs/vggsound/evaluate_all.sh` - Full evaluation pipeline

### 7.3 Analysis Scripts
- [ ] `src/analyze_merging_results.py` - Aggregate and plot results
- [ ] Generate comparison tables

---

## 8. Key Research Questions to Answer

1. **Does merging match joint training?**
   - Compare best merged model vs. M_joint

2. **Which merging method works best?**
   - Rank methods by downstream performance

3. **What are optimal hyperparameters?**
   - Best α, λ for each method

4. **Does merging preserve both capabilities?**
   - Reconstruction quality (MAE) + Retrieval (Contrastive)

5. **Is there interference between objectives?**
   - Analyze which parameters conflict most

6. **How does modality-specific performance change?**
   - Audio-only vs Visual-only vs Multimodal

---

## 9. References

- [Task Arithmetic (Ilharco et al., 2023)](https://arxiv.org/abs/2212.04089)
- [TIES-Merging (Yadav et al., 2023)](https://arxiv.org/abs/2306.01708)
- [DARE (Yu et al., 2023)](https://arxiv.org/abs/2311.03099)
- [Model Soups (Wortsman et al., 2022)](https://arxiv.org/abs/2203.05482)
- [CAV-MAE (Gong et al., 2023)](https://openreview.net/forum?id=QPtMRyk5rb)
- [VGGSound Dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)
- [MergeKit Toolkit](https://arxiv.org/abs/2403.13257)
