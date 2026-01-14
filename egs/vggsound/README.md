## Pretraining Scripts

### Joint Training (Original CAV-MAE)
- `run_cavmae_pretrain.sh` Pretrain on VGGSound from ImageNet initialization (MAE + Contrastive jointly).
- `run_cavmae_pretrain_as.sh` Pretrain on VGGSound from AS-2M pretraining model (CAV-MAE Scale++).

### Separate Objective Training (Model Merging Baseline)
- `run_cavmae_pretrain_mae_only.sh` Pretrain with **MAE objective only** (contrast_loss_weight=0).
- `run_cavmae_pretrain_contrastive_only.sh` Pretrain with **Contrastive objective only** (mae_loss_weight=0).

## Finetuning Script
- `run_cavmae_ft.sh` Finetune AS-2M pretrained CAV-MAE (Scale++) on VGGSound, should get ~65.8% accuracy.

## Model Merging

The model merging baseline trains MAE and Contrastive objectives **separately**, then merges the resulting models.

### Workflow

1. **Train MAE-only model:**
   ```bash
   sbatch run_cavmae_pretrain_mae_only.sh
   ```

2. **Train Contrastive-only model:**
   ```bash
   sbatch run_cavmae_pretrain_contrastive_only.sh
   ```

3. **Merge models** (after both training runs complete):
   ```bash
   bash merge_models.sh
   ```

### Merging Methods

The `merge_models.sh` script (which calls `src/merge_models.py`) supports three merging strategies:

| Method | Formula | Description |
|--------|---------|-------------|
| Simple Averaging | `(M_mae + M_contrastive) / 2` | Equal weight to both objectives |
| Weighted Averaging | `α * M_mae + (1-α) * M_contrastive` | Tunable weight α ∈ [0,1] |
| Task Arithmetic | `M_base + λ * (τ_mae + τ_contrastive)` | Where τ = M_trained - M_base |

### Usage Example

```bash
# Simple averaging
python ../../src/merge_models.py \
    --method simple \
    --model_mae ./exp/mae-only-.../models/best_audio_model.pth \
    --model_contrastive ./exp/contrastive-only-.../models/best_audio_model.pth \
    --output ./exp/merged-models/merged_simple.pth

# Weighted averaging (70% MAE, 30% Contrastive)
python ../../src/merge_models.py \
    --method weighted \
    --model_mae ./exp/mae-only-.../models/best_audio_model.pth \
    --model_contrastive ./exp/contrastive-only-.../models/best_audio_model.pth \
    --alpha 0.7 \
    --output ./exp/merged-models/merged_weighted_0.7.pth

# Task arithmetic
python ../../src/merge_models.py \
    --method task_arithmetic \
    --model_mae ./exp/mae-only-.../models/best_audio_model.pth \
    --model_contrastive ./exp/contrastive-only-.../models/best_audio_model.pth \
    --model_base ./IN-initial.pth \
    --lambda_scale 1.0 \
    --output ./exp/merged-models/merged_task_arith.pth
```
