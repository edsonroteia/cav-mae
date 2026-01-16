#!/bin/bash
# =============================================================================
# Extended Sweep script for Task Arithmetic Model Merging
# =============================================================================
# This script runs a comprehensive sweep over lambda values for task arithmetic.
#
# Formula: merged = base + lambda * (tau_mae + tau_contrastive)
#          where tau = trained_model - base_model
#
# Lambda values: [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
# Total: 7 merged models
# =============================================================================

set -e

# Activate environment
pushd /home/kuehne/kqr867/code/avllm-eval/training >/dev/null && source activate_env.sh && popd >/dev/null

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# =============================================================================
# Configuration
# =============================================================================

MAE_MODEL=./exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth
CONTRASTIVE_MODEL=./exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/models/best_audio_model.pth
BASE_MODEL=./IN-initial.pth

OUTPUT_DIR=./exp/merged-models/task-arithmetic-sweep
mkdir -p ${OUTPUT_DIR}

USE_CONTRASTIVE_NORMS="--use_contrastive_for_norms"

# =============================================================================
# Verify models exist
# =============================================================================
if [ ! -f "$MAE_MODEL" ]; then
    echo "ERROR: MAE model not found at $MAE_MODEL"
    exit 1
fi

if [ ! -f "$CONTRASTIVE_MODEL" ]; then
    echo "ERROR: Contrastive model not found at $CONTRASTIVE_MODEL"
    exit 1
fi

if [ ! -f "$BASE_MODEL" ]; then
    echo "ERROR: Base model not found at $BASE_MODEL"
    exit 1
fi

# =============================================================================
# Lambda Sweep
# =============================================================================

LAMBDAS=(0.3 0.5 0.7 1.0 1.2 1.5 2.0)

echo "=============================================="
echo "Starting Task Arithmetic Sweep"
echo "=============================================="
echo "Lambda values: ${LAMBDAS[*]}"
echo "Formula: merged = base + lambda * (tau_mae + tau_contrastive)"
echo ""

count=0
total=${#LAMBDAS[@]}

for lambda in "${LAMBDAS[@]}"; do
    count=$((count + 1))

    output_name="merged_task_arith_lambda${lambda}.pth"
    output_path="${OUTPUT_DIR}/${output_name}"

    echo "[$count/$total] Running task arithmetic: lambda=$lambda"
    echo "  Output: $output_path"

    python ../../src/merge_models.py \
        --method task_arithmetic \
        --model_mae ${MAE_MODEL} \
        --model_contrastive ${CONTRASTIVE_MODEL} \
        --model_base ${BASE_MODEL} \
        --lambda_scale $lambda \
        ${USE_CONTRASTIVE_NORMS} \
        --output ${output_path}

    echo ""
done

echo "=============================================="
echo "Task Arithmetic Sweep Complete!"
echo "=============================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
ls -la ${OUTPUT_DIR}
