#!/bin/bash
#SBATCH --job-name=cav-retrieval-sweep
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=log/%j_retrieval_sweep.txt
#SBATCH --error=log/%j_retrieval_sweep.txt
#SBATCH --exclude=mlcbm012,mlcbm005,mlcbm004,mlcbm003

# =============================================================================
# Batch Retrieval Evaluation for Model Sweep Directories
# =============================================================================
# Usage:
#   sbatch run_retrieval_sweep.sh [MODELS_DIR]
#   sbatch run_retrieval_sweep.sh ./exp/merged-models/orthogonal-sweep
#
# If no argument provided, evaluates all sweep directories:
#   - orthogonal-sweep
#   - weighted-sweep
#   - task-arithmetic-sweep
#   - dare-ties-sweep
# =============================================================================

set -e

# Create log directory
mkdir -p log

# Activate cav-mae environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

cd /weka/kuehne/kqr867/code/cav-mae

# VGGSound retrieval data
DATA_JSON=./datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=./datafiles/class_labels_indices_vgg.csv

# Output results directory
BASE_RESULTS_DIR=./egs/audioset/exp/retrieval_results
mkdir -p ${BASE_RESULTS_DIR}

echo "========================================"
echo "VGGSound Retrieval Evaluation - Sweep Models"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Function to evaluate a single model
evaluate_model() {
    local model_path=$1
    local output_csv=$2
    local model_name=$(basename $model_path .pth)

    if [ ! -f "$model_path" ]; then
        echo "  SKIP: Model not found: $model_path"
        return 1
    fi

    if [ -f "$output_csv" ]; then
        echo "  SKIP: Results already exist: $output_csv"
        return 0
    fi

    echo "  Evaluating: $model_name"
    python src/run_retrieval.py \
        --model_type cavmae \
        --model_path ${model_path} \
        --data_json ${DATA_JSON} \
        --label_csv ${LABEL_CSV} \
        --batch_size 48 \
        --output ${output_csv}

    return 0
}

# Function to evaluate all models in a directory
evaluate_directory() {
    local models_dir=$1
    local results_dir=$2

    if [ ! -d "$models_dir" ]; then
        echo "WARNING: Directory not found: $models_dir"
        return 1
    fi

    mkdir -p ${results_dir}

    echo ""
    echo "========================================"
    echo "Evaluating models in: $models_dir"
    echo "Results will be saved to: $results_dir"
    echo "========================================"

    local count=0
    local total=$(ls -1 ${models_dir}/*.pth 2>/dev/null | wc -l)

    if [ "$total" -eq 0 ]; then
        echo "No .pth files found in $models_dir"
        return 1
    fi

    echo "Found $total models to evaluate"
    echo ""

    for model_path in ${models_dir}/*.pth; do
        count=$((count + 1))
        model_name=$(basename $model_path .pth)
        output_csv="${results_dir}/${model_name}.csv"

        echo "[$count/$total] $model_name"
        evaluate_model "$model_path" "$output_csv"
        echo ""
    done

    # Generate summary CSV for this sweep
    local summary_csv="${results_dir}/summary.csv"
    echo "Generating summary: $summary_csv"
    echo "model,direction,R@1,R@5,R@10,MR" > ${summary_csv}
    for csv in ${results_dir}/*.csv; do
        if [ "$(basename $csv)" != "summary.csv" ]; then
            model_name=$(basename $csv .csv)
            tail -n +2 $csv | while read line; do
                echo "${model_name},${line}"
            done >> ${summary_csv}
        fi
    done

    return 0
}

# =============================================================================
# Main Evaluation Logic
# =============================================================================

MERGED_MODELS_BASE=./egs/audioset/exp/merged-models

if [ $# -ge 1 ]; then
    # Evaluate specific directory provided as argument
    MODELS_DIR=$1
    RESULTS_DIR="${BASE_RESULTS_DIR}/$(basename $MODELS_DIR)"
    evaluate_directory "$MODELS_DIR" "$RESULTS_DIR"
else
    # Evaluate all sweep directories
    echo "No specific directory provided. Evaluating all sweep directories..."

    # Orthogonal sweep
    if [ -d "${MERGED_MODELS_BASE}/orthogonal-sweep" ]; then
        evaluate_directory "${MERGED_MODELS_BASE}/orthogonal-sweep" "${BASE_RESULTS_DIR}/orthogonal-sweep"
    fi

    # Weighted sweep
    if [ -d "${MERGED_MODELS_BASE}/weighted-sweep" ]; then
        evaluate_directory "${MERGED_MODELS_BASE}/weighted-sweep" "${BASE_RESULTS_DIR}/weighted-sweep"
    fi

    # Task arithmetic sweep
    if [ -d "${MERGED_MODELS_BASE}/task-arithmetic-sweep" ]; then
        evaluate_directory "${MERGED_MODELS_BASE}/task-arithmetic-sweep" "${BASE_RESULTS_DIR}/task-arithmetic-sweep"
    fi

    # DARE-TIES sweep
    if [ -d "${MERGED_MODELS_BASE}/dare-ties-sweep" ]; then
        evaluate_directory "${MERGED_MODELS_BASE}/dare-ties-sweep" "${BASE_RESULTS_DIR}/dare-ties-sweep"
    fi
fi

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: ${BASE_RESULTS_DIR}"
ls -la ${BASE_RESULTS_DIR}/

echo ""
echo "To view summary of a sweep:"
echo "  cat ${BASE_RESULTS_DIR}/<sweep-name>/summary.csv"
