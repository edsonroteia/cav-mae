#!/bin/bash
# =============================================================================
# Launch Layerwise and Fisher Merge Experiments
# =============================================================================
# This script launches all jobs needed for layerwise and Fisher merge evaluation:
#
# Phase 1 (parallel):
#   - Layerwise merge sweep (creates 2 models)
#   - Fisher estimation for MAE model
#   - Fisher estimation for Contrastive model
#
# Phase 2 (after Fisher estimation):
#   - Fisher merge sweep (creates 2 models)
#
# Phase 3 (after all merges):
#   - Retrieval evaluation for all new models
# =============================================================================

set -e

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

mkdir -p log

echo "=============================================="
echo "Launching Layerwise & Fisher Merge Experiments"
echo "=============================================="
echo ""

# =============================================================================
# Phase 1: Layerwise merge (local, fast) + Fisher estimation (GPU jobs)
# =============================================================================

echo "Phase 1: Creating merged models and estimating Fisher..."
echo ""

# Run layerwise merge locally (fast, no GPU needed)
echo "[1/3] Running layerwise merge sweep..."
bash sweep_layerwise.sh
echo ""

# Submit Fisher estimation jobs (in parallel)
echo "[2/3] Submitting Fisher estimation for MAE model..."
JOB_FISHER_MAE=$(sbatch --parsable run_estimate_fisher_mae.sh)
echo "  Submitted Job ID: $JOB_FISHER_MAE"

echo "[3/3] Submitting Fisher estimation for Contrastive model..."
JOB_FISHER_CON=$(sbatch --parsable run_estimate_fisher_contrastive.sh)
echo "  Submitted Job ID: $JOB_FISHER_CON"

echo ""
echo "=============================================="
echo "Phase 1 Complete!"
echo "=============================================="
echo ""
echo "Layerwise models created in: exp/merged-models/layerwise-sweep/"
ls -la exp/merged-models/layerwise-sweep/
echo ""
echo "Fisher estimation jobs:"
echo "  MAE:         Job $JOB_FISHER_MAE"
echo "  Contrastive: Job $JOB_FISHER_CON"
echo ""

# =============================================================================
# Phase 2: Fisher merge (after Fisher estimation completes)
# =============================================================================

echo "=============================================="
echo "Phase 2: Submitting Fisher merge (with dependencies)"
echo "=============================================="
echo ""

# Submit Fisher merge with dependency on both Fisher jobs
JOB_FISHER_MERGE=$(sbatch --parsable --dependency=afterok:${JOB_FISHER_MAE}:${JOB_FISHER_CON} --wrap="cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset && bash sweep_fisher.sh" --job-name=fisher-merge --partition=h100-ferranti --gres=gpu:0 --mem=32G --time=00:30:00 --output=./log/%j_fisher_merge.txt --error=./log/%j_fisher_merge.err)
echo "Fisher merge job: $JOB_FISHER_MERGE (depends on $JOB_FISHER_MAE, $JOB_FISHER_CON)"
echo ""

# =============================================================================
# Phase 3: Retrieval evaluation (after all merges complete)
# =============================================================================

echo "=============================================="
echo "Phase 3: Submitting Retrieval evaluation (with dependencies)"
echo "=============================================="
echo ""

# Submit retrieval evaluation with dependency on Fisher merge
JOB_RETRIEVAL=$(sbatch --parsable --dependency=afterok:${JOB_FISHER_MERGE} run_retrieval_layerwise_fisher.sh)
echo "Retrieval evaluation job: $JOB_RETRIEVAL (depends on $JOB_FISHER_MERGE)"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=============================================="
echo "All Jobs Submitted!"
echo "=============================================="
echo ""
echo "Job Pipeline:"
echo "  Phase 1:"
echo "    - Layerwise sweep: DONE (local)"
echo "    - Fisher MAE:      Job $JOB_FISHER_MAE"
echo "    - Fisher Con:      Job $JOB_FISHER_CON"
echo ""
echo "  Phase 2:"
echo "    - Fisher merge:    Job $JOB_FISHER_MERGE (after Fisher estimation)"
echo ""
echo "  Phase 3:"
echo "    - Retrieval eval:  Job $JOB_RETRIEVAL (after Fisher merge)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "Expected outputs:"
echo "  - exp/merged-models/layerwise-sweep/*.pth"
echo "  - exp/merged-models/fisher-sweep/*.pth"
echo "  - exp/fisher-estimates/fisher_*.pth"
echo "  - exp/retrieval_results/layerwise-sweep/*.csv"
echo "  - exp/retrieval_results/fisher-sweep/*.csv"
