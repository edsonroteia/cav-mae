#!/bin/bash
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --job-name="cavjepa-test"
#SBATCH --output=./log/%j_cavjepa_test.txt
#SBATCH --error=./log/%j_cavjepa_test.err

# CAV-JEPA Quick Test - 1 GPU, 1 hour
# Verify training loop works before full run

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

# Model configuration
model=cav-jepa
masking_ratio=0.75

# JEPA-specific parameters
jepa_loss_weight=1.0
contrast_loss_weight=0.01
momentum_start=0.996
momentum_end=0.999
predictor_depth=4
predictor_dim=384

# Positional embedding
tr_pos=False

# Training parameters - SHORT TEST
bal=None
lr=1e-4
epoch=1
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
lr_adapt=False

# Dataset parameters
dataset=audioset
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=24  # Smaller batch for 1 GPU

# Data paths
cur_dir=$(pwd)
tr_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_2m_pretrain.json
te_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_eval_yuan.json
label_csv=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

# Pretrained weights
pretrain_path=${cur_dir}/cav_jepa_from_mae_init.pth

# Experiment directory
exp_dir=./exp/cavjepa-TEST-1gpu
mkdir -p $exp_dir
mkdir -p ./log

echo "=== CAV-JEPA QUICK TEST ==="
echo "1 GPU, 1 epoch, batch_size=24"
echo "Verifying training loop works..."

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavjepa_pretrain.py \
    --model ${model} \
    --dataset ${dataset} \
    --data-train ${tr_data} \
    --data-val ${te_data} \
    --exp-dir $exp_dir \
    --label-csv ${label_csv} \
    --n_class 527 \
    --lr $lr \
    --n-epochs ${epoch} \
    --batch-size $batch_size \
    --save_model False \
    --mixup ${mixup} \
    --bal ${bal} \
    --lrscheduler_start ${lrscheduler_start} \
    --lrscheduler_decay ${lrscheduler_decay} \
    --lrscheduler_step ${lrscheduler_step} \
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
    --tr_pos ${tr_pos}

echo "=== TEST COMPLETED ==="
