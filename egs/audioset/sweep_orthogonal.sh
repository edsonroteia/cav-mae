#!/bin/bash
# =============================================================================
# Sweep script for Orthogonal Conflict-Aware Model Merging
# =============================================================================
# This script runs a comprehensive sweep over the hyperparameter space for
# the orthogonal (conflict-aware) merging method.
#
# Hyperparameters:
#   beta (ortho_beta): Scaling factor for contrastive task vector [0.25, 0.5, 0.75, 1.0, 1.25]
#   tau (ortho_tau): Conflict threshold for cosine similarity [-0.1, 0.0, 0.1, 0.2, 0.3]
#
# Total combinations: 5 x 5 = 25 merged models
# =============================================================================

set -e

# Activate environment
pushd /home/kuehne/kqr867/code/avllm-eval/training >/dev/null && source activate_env.sh && popd >/dev/null

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# =============================================================================
# Configuration - Update these paths to match your experiment directories
# =============================================================================

# Paths to trained models
MAE_MODEL=./exp/mae-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth
CONTRASTIVE_MODEL=./exp/contrastive-only-audioset-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/models/best_audio_model.pth
BASE_MODEL=./IN-initial.pth

# Output directory for merged models
OUTPUT_DIR=./exp/merged-models/orthogonal-sweep
mkdir -p ${OUTPUT_DIR}

# Whether to use contrastive weights for norm layers (recommended due to MAE norm collapse)
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

# Beta values: scaling factor for contrastive task vector
BETAS=(0.25 0.5 0.75 1.0 1.25)

# Tau values: conflict threshold (cosine similarity)
TAUS=(-0.1 0.0 0.1 0.2 0.3)

echo "=============================================="
echo "Starting Orthogonal Merge Sweep"
echo "=============================================="
echo "Beta values: ${BETAS[*]}"
echo "Tau values: ${TAUS[*]}"
echo "Total combinations: $((${#BETAS[@]} * ${#TAUS[@]}))"
echo ""

count=0
total=$((${#BETAS[@]} * ${#TAUS[@]}))

for beta in "${BETAS[@]}"; do
    for tau in "${TAUS[@]}"; do
        count=$((count + 1))

        # Format tau for filename (replace negative sign with 'n')
        tau_str=$(echo "$tau" | sed 's/-/n/')

        output_name="merged_orthogonal_beta${beta}_tau${tau_str}.pth"
        output_path="${OUTPUT_DIR}/${output_name}"

        echo "[$count/$total] Running orthogonal merge: beta=$beta, tau=$tau"
        echo "  Output: $output_path"

        python ../../src/merge_models.py \
            --method orthogonal \
            --model_mae ${MAE_MODEL} \
            --model_contrastive ${CONTRASTIVE_MODEL} \
            --model_base ${BASE_MODEL} \
            --ortho_beta $beta \
            --ortho_tau $tau \
            ${USE_CONTRASTIVE_NORMS} \
            --output ${output_path}

        echo ""
    done
done

echo "=============================================="
echo "Orthogonal Merge Sweep Complete!"
echo "=============================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
ls -la ${OUTPUT_DIR}
echo ""
echo "Next steps:"
echo "1. Run retrieval evaluation: ./run_retrieval_sweep.sh ${OUTPUT_DIR}"
echo "2. Select top models for classification finetuning"
