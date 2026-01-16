#!/bin/bash
#SBATCH --job-name=cav-retr-working
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=log/%j_retrieval_working.txt
#SBATCH --error=log/%j_retrieval_working.txt
#SBATCH --exclude=mlcbm012,mlcbm005,mlcbm004,mlcbm003

# Evaluate only working models:
# - Base (IN-initial) - for baseline comparison
# - Contrastive-only - should work well for retrieval
#
# NOTE: MAE-only model has collapsed norm layers (all zeros) and is unusable

set -e

mkdir -p log

source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
cd /weka/kuehne/kqr867/code/cav-mae

RESULTS_DIR=./egs/audioset/exp/retrieval_results
mkdir -p ${RESULTS_DIR}

DATA_JSON=./datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=./datafiles/class_labels_indices_vgg.csv

echo "========================================"
echo "VGGSound Retrieval - Working Models Only"
echo "========================================"
echo "Start time: $(date)"

# =============================================================================
# 1. Base Model (IN-initial) - Baseline for comparison
# =============================================================================
echo ""
echo ">>> Evaluating: Base Model (IN-initial)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/IN-initial.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/base_IN_initial.csv

# =============================================================================
# 2. Contrastive-only Model - Expected to work well
# =============================================================================
echo ""
echo ">>> Evaluating: Contrastive-only (lr=1e-4)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/models/best_audio_model.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/contrastive_only_lr1e-4.csv

echo ""
echo "========================================"
echo "Complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
ls -la ${RESULTS_DIR}/*.csv 2>/dev/null || true
