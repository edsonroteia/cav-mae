#!/bin/bash
# =============================================================================
# Extended Sweep script for Weighted Averaging Model Merging
# =============================================================================
# This script runs a comprehensive sweep over alpha values for weighted averaging.
#
# Formula: merged = alpha * model_mae + (1-alpha) * model_contrastive
#
# Alpha values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# Total: 11 merged models
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

OUTPUT_DIR=./exp/merged-models/weighted-sweep
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

# =============================================================================
# Alpha Sweep
# =============================================================================

ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

echo "=============================================="
echo "Starting Weighted Averaging Sweep"
echo "=============================================="
echo "Alpha values: ${ALPHAS[*]}"
echo "Formula: merged = alpha * MAE + (1-alpha) * Contrastive"
echo ""

count=0
total=${#ALPHAS[@]}

for alpha in "${ALPHAS[@]}"; do
    count=$((count + 1))

    output_name="merged_weighted_alpha${alpha}.pth"
    output_path="${OUTPUT_DIR}/${output_name}"

    echo "[$count/$total] Running weighted averaging: alpha=$alpha"
    echo "  MAE weight: $alpha, Contrastive weight: $(echo "1 - $alpha" | bc)"
    echo "  Output: $output_path"

    python ../../src/merge_models.py \
        --method weighted \
        --model_mae ${MAE_MODEL} \
        --model_contrastive ${CONTRASTIVE_MODEL} \
        --alpha $alpha \
        ${USE_CONTRASTIVE_NORMS} \
        --output ${output_path}

    echo ""
done

echo "=============================================="
echo "Weighted Averaging Sweep Complete!"
echo "=============================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
ls -la ${OUTPUT_DIR}
