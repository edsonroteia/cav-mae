#!/bin/bash
# Master orchestration script for CAV++ / MAE++ post-training pipeline
# This script submits jobs with proper dependencies to ensure correct execution order

set -x

cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

echo "=========================================="
echo "CAV++ / MAE++ Post-Training Pipeline"
echo "=========================================="
echo ""
echo "This script will submit the following jobs:"
echo "  Phase 1 (Parallel):"
echo "    - Retrieval evaluation on base models"
echo "    - Fisher information estimation"
echo "  Phase 2 (After Fisher):"
echo "    - Model merging"
echo "  Phase 3 (After Merging):"
echo "    - Retrieval evaluation on merged models"
echo ""
echo "SFT jobs must be submitted manually after analyzing retrieval results."
echo ""

# Phase 1: Submit retrieval and fisher jobs in parallel
echo "Submitting Phase 1 jobs..."
RETRIEVAL_JOB=$(sbatch --parsable run_retrieval_plusplus.sh)
FISHER_JOB=$(sbatch --parsable run_estimate_fisher_plusplus.sh)

echo "  Retrieval job: $RETRIEVAL_JOB"
echo "  Fisher job: $FISHER_JOB"

# Phase 2: Submit merge job with dependency on Fisher
echo ""
echo "Submitting Phase 2 (merge) with dependency on Fisher job..."
MERGE_JOB=$(sbatch --parsable --dependency=afterok:$FISHER_JOB run_merge_plusplus.sh)
echo "  Merge job: $MERGE_JOB (depends on $FISHER_JOB)"

# Phase 3: Submit merged retrieval with dependency on merge
echo ""
echo "Submitting Phase 3 (merged retrieval) with dependency on merge job..."
MERGED_RETRIEVAL_JOB=$(sbatch --parsable --dependency=afterok:$MERGE_JOB run_retrieval_merged_plusplus.sh)
echo "  Merged retrieval job: $MERGED_RETRIEVAL_JOB (depends on $MERGE_JOB)"

echo ""
echo "=========================================="
echo "Pipeline Submitted!"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  Retrieval (base):    $RETRIEVAL_JOB"
echo "  Fisher estimation:   $FISHER_JOB"
echo "  Model merging:       $MERGE_JOB"
echo "  Retrieval (merged):  $MERGED_RETRIEVAL_JOB"
echo ""
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After Phase 3 completes, analyze results and submit SFT jobs manually:"
echo "  sbatch run_sft_merged_plusplus.sh merged_weighted_0.3"
echo "  sbatch run_sft_merged_plusplus.sh merged_fisher"
echo "  etc."
echo ""
echo "Results will be in:"
echo "  - exp/retrieval_results/plusplus/ (base models)"
echo "  - exp/fisher-estimates-plusplus/ (Fisher info)"
echo "  - exp/merged-models-plusplus/ (merged models)"
echo "  - exp/retrieval_results/merged-plusplus/ (merged model evaluation)"
