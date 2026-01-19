#!/bin/bash
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --job-name="cavjepa-v2"
#SBATCH --output=./log/%j_cavjepa_v2_pretrain.txt
#SBATCH --error=./log/%j_cavjepa_v2_pretrain.err

# CAV-JEPA v2 Pretraining on AudioSet
#
# Key improvements over v1:
# - Random initialization (no MAE pretrain bias)
# - Learnable mask tokens in predictor
# - Target representation normalization
# - Warmup + cosine LR scheduler
# - Weight decay schedule
# - Depth-scaled initialization

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

# Model configuration
model=cav-jepa

# === V2 INITIALIZATION ===
# 'random': Fresh initialization with depth-scaled rescaling (I-JEPA style)
# 'mae': Load from MAE checkpoint (original v1 behavior)
init_mode=random
normalize_targets=True
init_std=0.02

# JEPA-specific parameters
jepa_loss_weight=1.0
contrast_loss_weight=0.01
masking_ratio=0.75
momentum_start=0.996
momentum_end=0.999
predictor_depth=4
predictor_dim=384

# Positional embedding
tr_pos=False

# === V2 SCHEDULER ===
use_v2_scheduler=True
warmup_epochs=15
start_lr=1e-4
ref_lr=1e-3
final_lr=1e-6
start_wd=0.04
final_wd=0.4

# Training parameters
# Note: Longer training for random init (100 epochs vs 25 for MAE init)
bal=None
epoch=100
lr_adapt=False

# Dataset parameters (AudioSet)
dataset=audioset
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=120

# Data paths
cur_dir=$(pwd)
tr_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_2m_pretrain.json
te_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_eval_yuan.json
label_csv=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

# For random init, no pretrain path needed
pretrain_path=None

# Experiment directory
exp_dir=./exp/cavjepa-v2-${dataset}-${init_mode}-epoch${epoch}-bs${batch_size}-ref_lr${ref_lr}-warmup${warmup_epochs}-wd${start_wd}-${final_wd}
mkdir -p $exp_dir
mkdir -p ./log

echo "Starting CAV-JEPA v2 pretraining..."
echo "============================================"
echo "Experiment directory: $exp_dir"
echo ""
echo "V2 Features:"
echo "  Init mode: $init_mode"
echo "  Normalize targets: $normalize_targets"
echo "  V2 scheduler: $use_v2_scheduler"
echo ""
echo "LR Schedule:"
echo "  Warmup epochs: $warmup_epochs"
echo "  LR: $start_lr -> $ref_lr -> $final_lr"
echo ""
echo "Weight Decay Schedule:"
echo "  WD: $start_wd -> $final_wd"
echo ""
echo "EMA Momentum: $momentum_start -> $momentum_end"
echo "Predictor: depth=$predictor_depth, dim=$predictor_dim"
echo "============================================"

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavjepa_pretrain.py \
    --model ${model} \
    --dataset ${dataset} \
    --data-train ${tr_data} \
    --data-val ${te_data} \
    --exp-dir $exp_dir \
    --label-csv ${label_csv} \
    --n_class 527 \
    --lr ${ref_lr} \
    --n-epochs ${epoch} \
    --batch-size $batch_size \
    --save_model True \
    --mixup ${mixup} \
    --bal ${bal} \
    --dataset_mean ${dataset_mean} \
    --dataset_std ${dataset_std} \
    --target_length ${target_length} \
    --noise ${noise} \
    --warmup True \
    --lr_adapt ${lr_adapt} \
    --pretrain_path ${pretrain_path} \
    --jepa_loss_weight ${jepa_loss_weight} \
    --contrast_loss_weight ${contrast_loss_weight} \
    --masking_ratio ${masking_ratio} \
    --momentum_start ${momentum_start} \
    --momentum_end ${momentum_end} \
    --predictor_depth ${predictor_depth} \
    --predictor_dim ${predictor_dim} \
    --tr_pos ${tr_pos} \
    --init_mode ${init_mode} \
    --normalize_targets ${normalize_targets} \
    --init_std ${init_std} \
    --use_v2_scheduler ${use_v2_scheduler} \
    --warmup_epochs ${warmup_epochs} \
    --start_lr ${start_lr} \
    --ref_lr ${ref_lr} \
    --final_lr ${final_lr} \
    --start_wd ${start_wd} \
    --final_wd ${final_wd} \
    --use_wandb True \
    --wandb_project "cav-jepa-v2"

echo "CAV-JEPA v2 pretraining completed!"
