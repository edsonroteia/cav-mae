#!/bin/bash
# =============================================================================
# Sweep script for Fisher-Weighted Model Merging
# =============================================================================
# This script runs Fisher-weighted merging after Fisher estimation.
#
# Formula: merged = (F_mae * W_mae + F_con * W_con) / (F_mae + F_con)
# With base: merged = base + (F_mae * tau_mae + F_con * tau_con) / (F_mae + F_con)
#
# Requires: Fisher estimation must be run first (run_estimate_fisher_*.sh)
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

FISHER_MAE=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates/fisher_mae.pth
FISHER_CON=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates/fisher_contrastive.pth

OUTPUT_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models/fisher-sweep
mkdir -p ${OUTPUT_DIR}

USE_CONTRASTIVE_NORMS="--use_contrastive_for_norms"

# =============================================================================
# Verify models and Fisher estimates exist
# =============================================================================
for model in "$BASE_MODEL" "$MAE_MODEL" "$CONTRASTIVE_MODEL"; do
    if [ ! -f "$model" ]; then
        echo "ERROR: Model not found at $model"
        exit 1
    fi
done

for fisher in "$FISHER_MAE" "$FISHER_CON"; do
    if [ ! -f "$fisher" ]; then
        echo "ERROR: Fisher estimate not found at $fisher"
        echo "Run run_estimate_fisher_mae.sh and run_estimate_fisher_contrastive.sh first!"
        exit 1
    fi
done

echo "=============================================="
echo "Starting Fisher-Weighted Merging"
echo "=============================================="
echo "Formula: merged = base + (F_mae * tau_mae + F_con * tau_con) / (F_mae + F_con)"
echo ""

# =============================================================================
# Fisher merge without base model (direct weight averaging)
# =============================================================================
echo "[1/2] Running Fisher merge (direct weighting)..."
python /weka/kuehne/kqr867/code/cav-mae/src/merge_models.py \
    --method fisher \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --fisher_mae ${FISHER_MAE} \
    --fisher_contrastive ${FISHER_CON} \
    ${USE_CONTRASTIVE_NORMS} \
    --output ${OUTPUT_DIR}/merged_fisher_direct.pth

echo ""

# =============================================================================
# Fisher merge with base model (task vector weighting)
# =============================================================================
echo "[2/2] Running Fisher merge (task vector weighting with base)..."
python /weka/kuehne/kqr867/code/cav-mae/src/merge_models.py \
    --method fisher \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --model_base ${BASE_MODEL} \
    --fisher_mae ${FISHER_MAE} \
    --fisher_contrastive ${FISHER_CON} \
    ${USE_CONTRASTIVE_NORMS} \
    --output ${OUTPUT_DIR}/merged_fisher_taskvec.pth

echo ""
echo "=============================================="
echo "Fisher-Weighted Merging Complete!"
echo "=============================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
ls -la ${OUTPUT_DIR}
