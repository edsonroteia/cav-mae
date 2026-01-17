#!/bin/bash
# =============================================================================
# Launch individual retrieval evaluation jobs for all merged models
# =============================================================================
# This script submits separate sbatch jobs for each merged model,
# allowing parallel evaluation across multiple GPUs.
# =============================================================================

set -e

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# Create directories
mkdir -p log
mkdir -p exp/retrieval_results

# Base paths (absolute paths for sbatch jobs)
MERGED_MODELS_BASE=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models
RESULTS_BASE=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/retrieval_results

# VGGSound retrieval data
DATA_JSON=/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv

# Counter for submitted jobs
submitted=0
skipped=0

# Function to submit a single retrieval job
submit_retrieval_job() {
    local model_path=$1
    local output_csv=$2
    local job_name=$3

    # Skip if results already exist
    if [ -f "$output_csv" ]; then
        echo "SKIP: Results exist for $job_name"
        skipped=$((skipped + 1))
        return 0
    fi

    # Skip if model doesn't exist
    if [ ! -f "$model_path" ]; then
        echo "SKIP: Model not found: $model_path"
        skipped=$((skipped + 1))
        return 0
    fi

    # Create output directory if needed
    mkdir -p "$(dirname $output_csv)"

    # Submit the job
    sbatch --job-name="ret-${job_name}" \
           --partition=h100-ferranti \
           --nodes=1 \
           --ntasks=1 \
           --gres=gpu:1 \
           --mem=64G \
           --time=0:30:00 \
           --output="log/%j_retrieval_${job_name}.txt" \
           --error="log/%j_retrieval_${job_name}.txt" \
           --exclude=mlcbm012,mlcbm005,mlcbm004,mlcbm003 \
           --wrap="source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh && \
                   cd /weka/kuehne/kqr867/code/cav-mae && \
                   python src/run_retrieval.py \
                       --model_type cavmae \
                       --model_path ${model_path} \
                       --data_json ${DATA_JSON} \
                       --label_csv ${LABEL_CSV} \
                       --batch_size 48 \
                       --output ${output_csv}"

    submitted=$((submitted + 1))
    echo "Submitted: $job_name"
}

echo "=============================================="
echo "Launching Retrieval Evaluation Jobs"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# =============================================================================
# 1. Base models (for comparison)
# =============================================================================
echo "--- Base Models ---"

# Base model (IN-initial)
submit_retrieval_job \
    "/weka/kuehne/kqr867/code/cav-mae/egs/audioset/IN-initial.pth" \
    "${RESULTS_BASE}/base_IN_initial.csv" \
    "base"

# MAE-only (lr=1e-4)
submit_retrieval_job \
    "/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth" \
    "${RESULTS_BASE}/mae_only_lr1e-4.csv" \
    "mae-1e4"

# Contrastive-only (lr=1e-4)
submit_retrieval_job \
    "/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/models/best_audio_model.pth" \
    "${RESULTS_BASE}/contrastive_only_lr1e-4.csv" \
    "con-1e4"

# =============================================================================
# 2. Orthogonal sweep models
# =============================================================================
echo ""
echo "--- Orthogonal Sweep Models ---"
ORTHO_DIR="${MERGED_MODELS_BASE}/orthogonal-sweep"
ORTHO_RESULTS="${RESULTS_BASE}/orthogonal-sweep"
mkdir -p ${ORTHO_RESULTS}

if [ -d "$ORTHO_DIR" ]; then
    for model in ${ORTHO_DIR}/*.pth; do
        if [ -f "$model" ]; then
            name=$(basename $model .pth)
            submit_retrieval_job "$model" "${ORTHO_RESULTS}/${name}.csv" "ortho-${name##merged_orthogonal_}"
        fi
    done
fi

# =============================================================================
# 3. Weighted averaging sweep models
# =============================================================================
echo ""
echo "--- Weighted Averaging Sweep Models ---"
WEIGHTED_DIR="${MERGED_MODELS_BASE}/weighted-sweep"
WEIGHTED_RESULTS="${RESULTS_BASE}/weighted-sweep"
mkdir -p ${WEIGHTED_RESULTS}

if [ -d "$WEIGHTED_DIR" ]; then
    for model in ${WEIGHTED_DIR}/*.pth; do
        if [ -f "$model" ]; then
            name=$(basename $model .pth)
            submit_retrieval_job "$model" "${WEIGHTED_RESULTS}/${name}.csv" "wgt-${name##merged_weighted_}"
        fi
    done
fi

# =============================================================================
# 4. Task arithmetic sweep models
# =============================================================================
echo ""
echo "--- Task Arithmetic Sweep Models ---"
TASK_DIR="${MERGED_MODELS_BASE}/task-arithmetic-sweep"
TASK_RESULTS="${RESULTS_BASE}/task-arithmetic-sweep"
mkdir -p ${TASK_RESULTS}

if [ -d "$TASK_DIR" ]; then
    for model in ${TASK_DIR}/*.pth; do
        if [ -f "$model" ]; then
            name=$(basename $model .pth)
            submit_retrieval_job "$model" "${TASK_RESULTS}/${name}.csv" "task-${name##merged_task_arith_}"
        fi
    done
fi

# =============================================================================
# 5. DARE-TIES sweep models
# =============================================================================
echo ""
echo "--- DARE-TIES Sweep Models ---"
DARE_DIR="${MERGED_MODELS_BASE}/dare-ties-sweep"
DARE_RESULTS="${RESULTS_BASE}/dare-ties-sweep"
mkdir -p ${DARE_RESULTS}

if [ -d "$DARE_DIR" ]; then
    for model in ${DARE_DIR}/*.pth; do
        if [ -f "$model" ]; then
            name=$(basename $model .pth)
            submit_retrieval_job "$model" "${DARE_RESULTS}/${name}.csv" "dare-${name##merged_dare_ties_}"
        fi
    done
fi

# =============================================================================
# 6. Original merged models (from merge_models.sh)
# =============================================================================
echo ""
echo "--- Original Merged Models ---"
ORIG_DIR="${MERGED_MODELS_BASE}"
ORIG_RESULTS="${RESULTS_BASE}"

for model in ${ORIG_DIR}/merged_*.pth; do
    if [ -f "$model" ]; then
        name=$(basename $model .pth)
        submit_retrieval_job "$model" "${ORIG_RESULTS}/${name}.csv" "${name}"
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Job Submission Complete!"
echo "=============================================="
echo "Submitted: $submitted jobs"
echo "Skipped: $skipped (already have results or model missing)"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Results will be saved to: ${RESULTS_BASE}/"
echo ""
echo "After all jobs complete, generate summary with:"
echo "  ./aggregate_retrieval_results.sh"
