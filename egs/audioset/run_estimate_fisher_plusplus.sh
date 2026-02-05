#!/bin/bash
#SBATCH --job-name=fisher++
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --exclude=mlcbm005,mlcbm012
#SBATCH --output=log/%j_fisher_plusplus.txt
#SBATCH --error=log/%j_fisher_plusplus.err

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
export TORCH_HOME=/weka/kuehne/kqr867/code/cav-mae/pretrained_models

cd /weka/kuehne/kqr867/code/cav-mae/src

# Create output directory
mkdir -p /weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates-plusplus

# Model paths
CONTRASTIVE_PP=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/contrastive++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/models/best_audio_model.pth
MAE_PP=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/mae++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/models/best_audio_model.pth

# Data paths
DATA=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_2m_pretrain.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

# Output paths
FISHER_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates-plusplus

# Common parameters
N_BATCHES=100
BATCH_SIZE=12
MASKING_RATIO=0.75
MASK_MODE=unstructured
DATASET_MEAN=-5.081
DATASET_STD=4.4849

echo "=========================================="
echo "Fisher Information Estimation for ++ Models"
echo "=========================================="

# Estimate Fisher for MAE++ (using MAE loss only)
echo ""
echo "Estimating Fisher for MAE++ model..."
python estimate_fisher.py \
    --model $MAE_PP \
    --output $FISHER_DIR/fisher_mae++.pt \
    --data $DATA \
    --label-csv $LABEL_CSV \
    --dataset audioset \
    --dataset_mean $DATASET_MEAN \
    --dataset_std $DATASET_STD \
    --n-batches $N_BATCHES \
    --batch-size $BATCH_SIZE \
    --num-workers 8 \
    --masking_ratio $MASKING_RATIO \
    --mask_mode $MASK_MODE \
    --mae_loss_weight 1.0 \
    --contrast_loss_weight 0.0 \
    --norm_pix_loss False \
    --tr_pos False \
    --use_amp True

# Estimate Fisher for Contrastive++ (using contrastive loss only)
echo ""
echo "Estimating Fisher for Contrastive++ model..."
python estimate_fisher.py \
    --model $CONTRASTIVE_PP \
    --output $FISHER_DIR/fisher_contrastive++.pt \
    --data $DATA \
    --label-csv $LABEL_CSV \
    --dataset audioset \
    --dataset_mean $DATASET_MEAN \
    --dataset_std $DATASET_STD \
    --n-batches $N_BATCHES \
    --batch-size $BATCH_SIZE \
    --num-workers 8 \
    --masking_ratio $MASKING_RATIO \
    --mask_mode $MASK_MODE \
    --mae_loss_weight 0.0 \
    --contrast_loss_weight 1.0 \
    --norm_pix_loss False \
    --tr_pos False \
    --use_amp True

echo ""
echo "=========================================="
echo "Fisher estimation completed!"
echo "Results saved to $FISHER_DIR"
echo "=========================================="

ls -la $FISHER_DIR
