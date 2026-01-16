#!/bin/bash
#SBATCH --job-name=cav-retrieval-all
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=log/%j_retrieval_all.txt
#SBATCH --error=log/%j_retrieval_all.txt
#SBATCH --exclude=mlcbm012,mlcbm005,mlcbm004,mlcbm003

# Evaluate all models (base, MAE-only, contrastive-only, merged variants) on VGGSound retrieval

set -e

# Create log directory
mkdir -p log

# Activate cav-mae environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

cd /weka/kuehne/kqr867/code/cav-mae

# Output results directory
RESULTS_DIR=./egs/audioset/exp/retrieval_results
mkdir -p ${RESULTS_DIR}

# VGGSound retrieval data
DATA_JSON=./datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=./datafiles/class_labels_indices_vgg.csv

echo "========================================"
echo "VGGSound Retrieval Evaluation - All Models"
echo "========================================"
echo "Start time: $(date)"
echo ""

# =============================================================================
# 1. Base Model (IN-initial)
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
# 2. MAE-only Model
# =============================================================================
echo ""
echo ">>> Evaluating: MAE-only (lr=1e-4)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/mae_only_lr1e-4.csv

# =============================================================================
# 3. Contrastive-only Model
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

# =============================================================================
# 4. Merged Models - Simple Average
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Simple Average"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_simple.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_simple.csv

# =============================================================================
# 5. Merged Models - Weighted Average (alpha=0.3)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Weighted (alpha=0.3, 30% MAE, 70% Contrastive)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_weighted_0.3.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_weighted_0.3.csv

# =============================================================================
# 6. Merged Models - Weighted Average (alpha=0.5)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Weighted (alpha=0.5, 50% MAE, 50% Contrastive)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_weighted_0.5.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_weighted_0.5.csv

# =============================================================================
# 7. Merged Models - Weighted Average (alpha=0.7)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Weighted (alpha=0.7, 70% MAE, 30% Contrastive)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_weighted_0.7.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_weighted_0.7.csv

# =============================================================================
# 8. Merged Models - Task Arithmetic (lambda=0.5)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Task Arithmetic (lambda=0.5)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_task_arith_0.5.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_task_arith_0.5.csv

# =============================================================================
# 9. Merged Models - Task Arithmetic (lambda=1.0)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Task Arithmetic (lambda=1.0)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_task_arith_1.0.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_task_arith_1.0.csv

# =============================================================================
# 10. Merged Models - Task Arithmetic (lambda=1.5)
# =============================================================================
echo ""
echo ">>> Evaluating: Merged Task Arithmetic (lambda=1.5)"
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ./egs/audioset/exp/merged-models/merged_task_arith_1.5.pth \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${RESULTS_DIR}/merged_task_arith_1.5.csv

# =============================================================================
# Summary - Aggregate all results
# =============================================================================
echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "Individual CSV files:"
ls -la ${RESULTS_DIR}/*.csv
echo ""

# Create summary table
echo "========================================"
echo "SUMMARY TABLE"
echo "========================================"
echo "Model,Direction,R@1,R@5,R@10,MR"
for csv in ${RESULTS_DIR}/*.csv; do
    model_name=$(basename $csv .csv)
    # Skip header and print each row with model name
    tail -n +2 $csv | while read line; do
        echo "${model_name},${line}"
    done
done

echo ""
echo "Done!"
