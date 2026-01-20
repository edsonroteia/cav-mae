#!/bin/bash
#SBATCH --job-name=fisher-mae
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./log/%j_fisher_mae.txt
#SBATCH --error=./log/%j_fisher_mae.err

# =============================================================================
# Estimate Fisher Information Diagonal for MAE-only checkpoint
# =============================================================================
# Uses MAE loss (mae_loss_weight=1.0, contrast_loss_weight=0.0) to compute
# the Fisher information matrix diagonal.
# =============================================================================

set -ex

source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# Configuration
MODEL=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth
DATA=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_eval_yuan.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv
OUTPUT_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates
OUTPUT=${OUTPUT_DIR}/fisher_mae.pth

mkdir -p ${OUTPUT_DIR}
mkdir -p ./log

echo "=============================================="
echo "Estimating Fisher Information for MAE model"
echo "=============================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo ""

python /weka/kuehne/kqr867/code/cav-mae/src/estimate_fisher.py \
    --model ${MODEL} \
    --output ${OUTPUT} \
    --data ${DATA} \
    --label-csv ${LABEL_CSV} \
    --dataset audioset \
    --dataset_mean -5.081 \
    --dataset_std 4.4849 \
    --target_length 1024 \
    --batch-size 12 \
    --n-batches 100 \
    --log-every 25 \
    --mae_loss_weight 1.0 \
    --contrast_loss_weight 0.0 \
    --masking_ratio 0.75 \
    --mask_mode unstructured \
    --norm_pix_loss True \
    --tr_pos False \
    --use_amp True

echo "Fisher estimation complete: $OUTPUT"
ls -la ${OUTPUT}
