#!/bin/bash
# =============================================================================
# Sweep script for DARE-TIES Model Merging
# =============================================================================
# This script runs a sweep over DARE-TIES hyperparameters.
#
# Formula: merged = base + lambda * TIES(DARE(tau_mae), DARE(tau_contrastive))
#
# Hyperparameters:
#   keep_ratio: Fraction of parameters to keep [0.1, 0.2, 0.3, 0.5]
#   lambda_scale: Scaling factor for merged task vectors [0.5, 1.0, 1.5]
#
# Total combinations: 4 x 3 = 12 merged models
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

OUTPUT_DIR=./exp/merged-models/dare-ties-sweep
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
# Hyperparameter Sweep
# =============================================================================

KEEP_RATIOS=(0.1 0.2 0.3 0.5)
LAMBDAS=(0.5 1.0 1.5)

echo "=============================================="
echo "Starting DARE-TIES Sweep"
echo "=============================================="
echo "Keep ratios: ${KEEP_RATIOS[*]}"
echo "Lambda values: ${LAMBDAS[*]}"
echo "Total combinations: $((${#KEEP_RATIOS[@]} * ${#LAMBDAS[@]}))"
echo ""

count=0
total=$((${#KEEP_RATIOS[@]} * ${#LAMBDAS[@]}))

for keep_ratio in "${KEEP_RATIOS[@]}"; do
    for lambda in "${LAMBDAS[@]}"; do
        count=$((count + 1))

        output_name="merged_dare_ties_kr${keep_ratio}_lambda${lambda}.pth"
        output_path="${OUTPUT_DIR}/${output_name}"

        echo "[$count/$total] Running DARE-TIES: keep_ratio=$keep_ratio, lambda=$lambda"
        echo "  Output: $output_path"

        python ../../src/merge_models.py \
            --method dare_ties \
            --model_mae ${MAE_MODEL} \
            --model_contrastive ${CONTRASTIVE_MODEL} \
            --model_base ${BASE_MODEL} \
            --dare_keep_ratio $keep_ratio \
            --lambda_scale $lambda \
            ${USE_CONTRASTIVE_NORMS} \
            --output ${output_path}

        echo ""
    done
done

echo "=============================================="
echo "DARE-TIES Sweep Complete!"
echo "=============================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
ls -la ${OUTPUT_DIR}
