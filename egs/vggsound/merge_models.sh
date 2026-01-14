#!/bin/bash
# Script to merge MAE-only and Contrastive-only trained CAV-MAE models
# Run this after completing both training runs

set -x

# =============================================================================
# Configuration - Update these paths to match your experiment directories
# =============================================================================

# Paths to trained models (update these after training completes)
MAE_MODEL=./exp/mae-only-vggsound-cav-mae-balNone-lr1e-4-epoch25-bs120-normTrue-mr-unstructured-0.75/models/best_audio_model.pth
CONTRASTIVE_MODEL=./exp/contrastive-only-vggsound-cav-mae-balNone-lr1e-4-epoch25-bs120-mr-unstructured-0.75/models/best_audio_model.pth

# Base model (the initial checkpoint used to start both training runs)
# This is needed for task arithmetic
BASE_MODEL=./IN-initial.pth

# Output directory for merged models
OUTPUT_DIR=./exp/merged-models
mkdir -p ${OUTPUT_DIR}

# =============================================================================
# Method 1: Simple Averaging
# Formula: merged = (model_mae + model_contrastive) / 2
# =============================================================================
echo "Running simple averaging..."
python ../../src/merge_models.py \
    --method simple \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --output ${OUTPUT_DIR}/merged_simple.pth

# =============================================================================
# Method 2: Weighted Averaging - Various alpha values
# Formula: merged = alpha * model_mae + (1-alpha) * model_contrastive
# =============================================================================

# Equal weighting (same as simple average)
echo "Running weighted averaging with alpha=0.5..."
python ../../src/merge_models.py \
    --method weighted \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --alpha 0.5 \
    --output ${OUTPUT_DIR}/merged_weighted_0.5.pth

# Favor MAE (70% MAE, 30% Contrastive)
echo "Running weighted averaging with alpha=0.7..."
python ../../src/merge_models.py \
    --method weighted \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --alpha 0.7 \
    --output ${OUTPUT_DIR}/merged_weighted_0.7.pth

# Favor Contrastive (30% MAE, 70% Contrastive)
echo "Running weighted averaging with alpha=0.3..."
python ../../src/merge_models.py \
    --method weighted \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --alpha 0.3 \
    --output ${OUTPUT_DIR}/merged_weighted_0.3.pth

# =============================================================================
# Method 3: Task Arithmetic - Various lambda scales
# Formula: merged = base + lambda * (tau_mae + tau_contrastive)
# where tau = trained_model - base_model
# =============================================================================

# Standard task arithmetic (lambda=1.0)
echo "Running task arithmetic with lambda=1.0..."
python ../../src/merge_models.py \
    --method task_arithmetic \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --model_base ${BASE_MODEL} \
    --lambda_scale 1.0 \
    --output ${OUTPUT_DIR}/merged_task_arith_1.0.pth

# Reduced scaling (lambda=0.5)
echo "Running task arithmetic with lambda=0.5..."
python ../../src/merge_models.py \
    --method task_arithmetic \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --model_base ${BASE_MODEL} \
    --lambda_scale 0.5 \
    --output ${OUTPUT_DIR}/merged_task_arith_0.5.pth

# Increased scaling (lambda=1.5)
echo "Running task arithmetic with lambda=1.5..."
python ../../src/merge_models.py \
    --method task_arithmetic \
    --model_mae ${MAE_MODEL} \
    --model_contrastive ${CONTRASTIVE_MODEL} \
    --model_base ${BASE_MODEL} \
    --lambda_scale 1.5 \
    --output ${OUTPUT_DIR}/merged_task_arith_1.5.pth

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Merged models saved to: ${OUTPUT_DIR}"
echo "=========================================="
ls -la ${OUTPUT_DIR}
echo ""
echo "Next steps:"
echo "1. Fine-tune each merged model using run_cavmae_ft.sh"
echo "2. Compare downstream task performance (mAP/accuracy)"
echo "3. Optionally evaluate on retrieval tasks"
