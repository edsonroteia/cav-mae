#!/bin/bash
# =============================================================================
# Sweep script for Layerwise Model Merging
# =============================================================================
# This script runs layerwise merging with different grouping strategies.
#
# Formula: merged = base + alpha_g * tau_mae + (1-alpha_g) * tau_con
#          where alpha_g = ||tau_mae|| / (||tau_mae|| + ||tau_con||)
#
# Group modes: "block" (per transformer block) and "param" (per parameter)
# =============================================================================

set -e

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/IN-initial.pth
MAE_MODEL=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth
CONTRASTIVE_MODEL=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/models/best_audio_model.pth

OUTPUT_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models/layerwise-sweep
mkdir -p ${OUTPUT_DIR}

USE_CONTRASTIVE_NORMS="--use_contrastive_for_norms"

# =============================================================================
# Verify models exist
# =============================================================================
for model in "$BASE_MODEL" "$MAE_MODEL" "$CONTRASTIVE_MODEL"; do
    if [ ! -f "$model" ]; then
        echo "ERROR: Model not found at $model"
        exit 1
    fi
done

echo "=============================================="
echo "Starting Layerwise Merging Sweep"
echo "=============================================="
echo "Formula: merged = base + alpha_g * tau_mae + (1-alpha_g) * tau_con"
echo "         alpha_g = ||tau_mae|| / (||tau_mae|| + ||tau_con||)"
echo ""

# =============================================================================
# Layerwise with block grouping
# =============================================================================
echo "[1/2] Running layerwise merge with block grouping..."
python /weka/kuehne/kqr867/code/cav-mae/src/merge_models.py \
    --method layerwise \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --model_base ${BASE_MODEL} \
    --layerwise_group block \
    ${USE_CONTRASTIVE_NORMS} \
    --output ${OUTPUT_DIR}/merged_layerwise_block.pth

echo ""

# =============================================================================
# Layerwise with param grouping
# =============================================================================
echo "[2/2] Running layerwise merge with param grouping..."
python /weka/kuehne/kqr867/code/cav-mae/src/merge_models.py \
    --method layerwise \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --model_base ${BASE_MODEL} \
    --layerwise_group param \
    ${USE_CONTRASTIVE_NORMS} \
    --output ${OUTPUT_DIR}/merged_layerwise_param.pth

echo ""
echo "=============================================="
echo "Layerwise Merging Sweep Complete!"
echo "=============================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
ls -la ${OUTPUT_DIR}
