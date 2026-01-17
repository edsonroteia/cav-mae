#!/bin/bash
# =============================================================================
# Launch all evaluation jobs for original CAV-MAE models (Scale++ and Scale+)
# =============================================================================
# This script submits retrieval and SFT jobs for fair comparison against
# our custom pretrained models.
#
# Original models from Yuan Gong et al. (ICLR 2023):
# - Scale++: batch_size=256, lambda_c=0.01, 75% unstructured masking
# - Scale+:  batch_size=108, lambda_c=0.01, 75% unstructured masking
# =============================================================================

set -e

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

# Create directories
mkdir -p log
mkdir -p exp/retrieval_results

echo "=============================================="
echo "Launching Original CAV-MAE Evaluation Jobs"
echo "=============================================="
echo "Start time: $(date)"
echo ""

# Check if models exist
if [ ! -f "cav-mae-scale++.pth" ]; then
    echo "ERROR: cav-mae-scale++.pth not found!"
    echo "Download from: https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1"
    exit 1
fi

if [ ! -f "cav-mae-scale+.pth" ]; then
    echo "ERROR: cav-mae-scale+.pth not found!"
    echo "Download from: https://www.dropbox.com/s/xu8bfie6hz86oev/audio_model.25.pth?dl=1"
    exit 1
fi

echo "Models found:"
echo "  - cav-mae-scale++.pth: $(ls -lh cav-mae-scale++.pth | awk '{print $5}')"
echo "  - cav-mae-scale+.pth:  $(ls -lh cav-mae-scale+.pth | awk '{print $5}')"
echo ""

# =============================================================================
# Retrieval Evaluation Jobs
# =============================================================================
echo "--- Retrieval Jobs ---"

# Scale++ retrieval
echo "Submitting Scale++ retrieval job..."
RET_SCALEPP=$(sbatch run_retrieval_scalepp.sh | awk '{print $4}')
echo "  Scale++ Retrieval Job ID: $RET_SCALEPP"

# Scale+ retrieval
echo "Submitting Scale+ retrieval job..."
RET_SCALEP=$(sbatch run_retrieval_scalep.sh | awk '{print $4}')
echo "  Scale+ Retrieval Job ID: $RET_SCALEP"

echo ""

# =============================================================================
# SFT Evaluation Jobs
# =============================================================================
echo "--- SFT Jobs ---"

# Scale++ SFT
echo "Submitting Scale++ SFT job..."
SFT_SCALEPP=$(sbatch run_sft_scalepp.sh | awk '{print $4}')
echo "  Scale++ SFT Job ID: $SFT_SCALEPP"

# Scale+ SFT
echo "Submitting Scale+ SFT job..."
SFT_SCALEP=$(sbatch run_sft_scalep.sh | awk '{print $4}')
echo "  Scale+ SFT Job ID: $SFT_SCALEP"

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "All Jobs Submitted!"
echo "=============================================="
echo ""
echo "Retrieval Jobs:"
echo "  Scale++ : $RET_SCALEPP"
echo "  Scale+  : $RET_SCALEP"
echo ""
echo "SFT Jobs:"
echo "  Scale++ : $SFT_SCALEPP"
echo "  Scale+  : $SFT_SCALEP"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs will be in: ./log/"
echo ""
echo "Retrieval results will be saved to:"
echo "  - exp/retrieval_results/original_scalepp.csv"
echo "  - exp/retrieval_results/original_scalep.csv"
echo ""
echo "SFT results will be saved to:"
echo "  - exp/sft-scalepp-original-*/"
echo "  - exp/sft-scalep-original-*/"
