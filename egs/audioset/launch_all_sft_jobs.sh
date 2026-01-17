#!/bin/bash
# Launch all SFT jobs for the four pretrained models
# Usage: ./launch_all_sft_jobs.sh

set -x

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

echo "=========================================="
echo "Launching SFT Jobs"
echo "=========================================="

# Launch CAV-only (Contrastive-only) SFT
echo "Submitting CAV-only SFT job..."
JOB1=$(sbatch run_sft_cavonly.sh | awk '{print $4}')
echo "CAV-only Job ID: $JOB1"

# Launch CAV-merged (alpha=0.1) SFT
echo "Submitting CAV-merged SFT job..."
JOB2=$(sbatch run_sft_cavmerged.sh | awk '{print $4}')
echo "CAV-merged Job ID: $JOB2"

# Launch MAE-only SFT
echo "Submitting MAE-only SFT job..."
JOB3=$(sbatch run_sft_maeonly.sh | awk '{print $4}')
echo "MAE-only Job ID: $JOB3"

# Launch CAV-JEPA SFT
echo "Submitting CAV-JEPA SFT job..."
JOB4=$(sbatch run_sft_cavjepa.sh | awk '{print $4}')
echo "CAV-JEPA Job ID: $JOB4"

echo ""
echo "=========================================="
echo "All SFT jobs submitted!"
echo "=========================================="
echo "Job IDs:"
echo "  CAV-only:   $JOB1"
echo "  CAV-merged: $JOB2"
echo "  MAE-only:   $JOB3"
echo "  CAV-JEPA:   $JOB4"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Logs will be in: ./log/"
