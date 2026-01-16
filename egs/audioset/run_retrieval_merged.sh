#!/bin/bash
#SBATCH --job-name=cav-retr-merged
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=log/%j_retrieval_merged.txt
#SBATCH --error=log/%j_retrieval_merged.txt
#SBATCH --exclude=mlcbm012,mlcbm005,mlcbm004,mlcbm003

# Evaluate all merged models on VGGSound retrieval
# NOTE: MAE-only model has collapsed norm weights - merged models may be affected

set -e

mkdir -p log

source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
cd /weka/kuehne/kqr867/code/cav-mae

RESULTS_DIR=./egs/audioset/exp/retrieval_results
mkdir -p ${RESULTS_DIR}

DATA_JSON=./datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=./datafiles/class_labels_indices_vgg.csv
MERGED_DIR=./egs/audioset/exp/merged-models

echo "========================================"
echo "VGGSound Retrieval - Merged Models"
echo "========================================"
echo "Start time: $(date)"

# =============================================================================
# 1. Simple Average
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Simple Average"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_simple.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_simple.csv

# =============================================================================
# 2. Weighted Average (alpha=0.3) - 30% MAE, 70% Contrastive
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Weighted (alpha=0.3, 30% MAE, 70% Contrastive)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_weighted_0.3.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_weighted_0.3.csv

# =============================================================================
# 3. Weighted Average (alpha=0.5) - 50% MAE, 50% Contrastive
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Weighted (alpha=0.5, 50% MAE, 50% Contrastive)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_weighted_0.5.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_weighted_0.5.csv

# =============================================================================
# 4. Weighted Average (alpha=0.7) - 70% MAE, 30% Contrastive
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Weighted (alpha=0.7, 70% MAE, 30% Contrastive)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_weighted_0.7.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_weighted_0.7.csv

# =============================================================================
# 5. Task Arithmetic (lambda=0.5)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Task Arithmetic (lambda=0.5)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_task_arith_0.5.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_task_arith_0.5.csv

# =============================================================================
# 6. Task Arithmetic (lambda=1.0)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Task Arithmetic (lambda=1.0)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_task_arith_1.0.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_task_arith_1.5.csv

# =============================================================================
# 7. Task Arithmetic (lambda=1.5)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Task Arithmetic (lambda=1.5)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MERGED_DIR}/merged_task_arith_1.5.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_task_arith_1.5.csv

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "All merged model evaluations complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
ls -la ${RESULTS_DIR}/merged*.csv 2>/dev/null || true
echo ""
echo "Quick summary:"
for csv in ${RESULTS_DIR}/merged*.csv; do
    if [ -f "$csv" ]; then
        model_name=$(basename $csv .csv)
        echo "=== $model_name ==="
        cat $csv
        echo ""
    fi
done
