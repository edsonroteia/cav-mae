#!/bin/bash
#SBATCH --job-name=retrieval-lw-fisher
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=./log/%j_retrieval_layerwise_fisher.txt
#SBATCH --error=./log/%j_retrieval_layerwise_fisher.err

# =============================================================================
# Retrieval Evaluation for Layerwise and Fisher Merged Models
# =============================================================================
# Evaluates all merged models from:
#   - layerwise-sweep/ (2 models)
#   - fisher-sweep/ (2 models)
# =============================================================================

set -e

source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

cd /weka/kuehne/kqr867/code/cav-mae

# VGGSound retrieval data
DATA_JSON=/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv

# Output results directory
BASE_RESULTS_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/retrieval_results

mkdir -p ${BASE_RESULTS_DIR}/layerwise-sweep
mkdir -p ${BASE_RESULTS_DIR}/fisher-sweep
mkdir -p ./egs/audioset/log

echo "========================================"
echo "VGGSound Retrieval - Layerwise & Fisher Models"
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
        echo "  EXISTS: $output_csv - re-running anyway"
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

# =============================================================================
# Layerwise Models
# =============================================================================
echo ""
echo "========================================"
echo "Evaluating Layerwise Merged Models"
echo "========================================"

LAYERWISE_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models/layerwise-sweep
LAYERWISE_RESULTS=${BASE_RESULTS_DIR}/layerwise-sweep

if [ -d "$LAYERWISE_DIR" ]; then
    for model_path in ${LAYERWISE_DIR}/*.pth; do
        if [ -f "$model_path" ]; then
            model_name=$(basename $model_path .pth)
            output_csv="${LAYERWISE_RESULTS}/${model_name}.csv"
            evaluate_model "$model_path" "$output_csv"
            echo ""
        fi
    done
else
    echo "WARNING: Layerwise models not found at $LAYERWISE_DIR"
    echo "Run sweep_layerwise.sh first!"
fi

# =============================================================================
# Fisher Models
# =============================================================================
echo ""
echo "========================================"
echo "Evaluating Fisher Merged Models"
echo "========================================"

FISHER_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models/fisher-sweep
FISHER_RESULTS=${BASE_RESULTS_DIR}/fisher-sweep

if [ -d "$FISHER_DIR" ]; then
    for model_path in ${FISHER_DIR}/*.pth; do
        if [ -f "$model_path" ]; then
            model_name=$(basename $model_path .pth)
            output_csv="${FISHER_RESULTS}/${model_name}.csv"
            evaluate_model "$model_path" "$output_csv"
            echo ""
        fi
    done
else
    echo "WARNING: Fisher models not found at $FISHER_DIR"
    echo "Run sweep_fisher.sh first (after Fisher estimation)!"
fi

# =============================================================================
# Generate Summary
# =============================================================================
echo ""
echo "========================================"
echo "Generating Summary"
echo "========================================"

# Layerwise summary
if [ -d "$LAYERWISE_RESULTS" ] && [ "$(ls -1 ${LAYERWISE_RESULTS}/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
    summary_csv="${LAYERWISE_RESULTS}/summary.csv"
    echo "model,direction,R@1,R@5,R@10,MR" > ${summary_csv}
    for csv in ${LAYERWISE_RESULTS}/*.csv; do
        if [ "$(basename $csv)" != "summary.csv" ]; then
            model_name=$(basename $csv .csv)
            tail -n +2 $csv | while read line; do
                echo "${model_name},${line}"
            done >> ${summary_csv}
        fi
    done
    echo "Layerwise summary: $summary_csv"
    cat $summary_csv
fi

# Fisher summary
if [ -d "$FISHER_RESULTS" ] && [ "$(ls -1 ${FISHER_RESULTS}/*.csv 2>/dev/null | wc -l)" -gt 0 ]; then
    summary_csv="${FISHER_RESULTS}/summary.csv"
    echo "model,direction,R@1,R@5,R@10,MR" > ${summary_csv}
    for csv in ${FISHER_RESULTS}/*.csv; do
        if [ "$(basename $csv)" != "summary.csv" ]; then
            model_name=$(basename $csv .csv)
            tail -n +2 $csv | while read line; do
                echo "${model_name},${line}"
            done >> ${summary_csv}
        fi
    done
    echo ""
    echo "Fisher summary: $summary_csv"
    cat $summary_csv
fi

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
echo "End time: $(date)"
