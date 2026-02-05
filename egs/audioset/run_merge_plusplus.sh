#!/bin/bash
#SBATCH --job-name=merge++
#SBATCH --partition=cpu-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/log/%j_merge_plusplus.txt
#SBATCH --error=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/log/%j_merge_plusplus.err

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

cd /weka/kuehne/kqr867/code/cav-mae/src

# Model paths
MAE_PP=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/mae++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/models/best_audio_model.pth
CON_PP=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/contrastive++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/models/best_audio_model.pth
BASE_PATH=/weka/kuehne/kqr867/code/cav-mae/IN-initial.pth

# Fisher paths (must run run_estimate_fisher_plusplus.sh first)
FISHER_MAE=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates-plusplus/fisher_mae++.pt
FISHER_CON=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/fisher-estimates-plusplus/fisher_contrastive++.pt

# Output directory
OUT_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models-plusplus
mkdir -p $OUT_DIR

echo "=========================================="
echo "Model Merging for ++ Models"
echo "=========================================="
echo "MAE++ model: $MAE_PP"
echo "Contrastive++ model: $CON_PP"
echo "Base model: $BASE_PATH"
echo "Output directory: $OUT_DIR"
echo "=========================================="

# Simple averaging
echo ""
echo "Creating simple average merge..."
python merge_models.py --method simple \
    --model_mae $MAE_PP \
    --model_contrastive $CON_PP \
    --use_contrastive_for_norms \
    --output $OUT_DIR/merged_simple.pth

# Weighted sweep: alpha from 0.0 to 1.0 (alpha is weight for MAE)
echo ""
echo "Creating weighted merges..."
for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "  alpha = $alpha"
    python merge_models.py --method weighted \
        --model_mae $MAE_PP \
        --model_contrastive $CON_PP \
        --alpha $alpha \
        --use_contrastive_for_norms \
        --output $OUT_DIR/merged_weighted_${alpha}.pth
done

# Check if Fisher files exist before running Fisher merge
if [[ -f "$FISHER_MAE" && -f "$FISHER_CON" ]]; then
    echo ""
    echo "Creating Fisher-weighted merge..."
    python merge_models.py --method fisher \
        --model_mae $MAE_PP \
        --model_contrastive $CON_PP \
        --model_base $BASE_PATH \
        --fisher_mae $FISHER_MAE \
        --fisher_contrastive $FISHER_CON \
        --use_contrastive_for_norms \
        --output $OUT_DIR/merged_fisher.pth
else
    echo ""
    echo "WARNING: Fisher files not found. Skipping Fisher merge."
    echo "Run run_estimate_fisher_plusplus.sh first."
fi

# Task arithmetic sweep
echo ""
echo "Creating task arithmetic merges..."
for lambda in 0.5 0.7 1.0 1.2 1.5; do
    echo "  lambda = $lambda"
    python merge_models.py --method task_arithmetic \
        --model_mae $MAE_PP \
        --model_contrastive $CON_PP \
        --model_base $BASE_PATH \
        --lambda_scale $lambda \
        --use_contrastive_for_norms \
        --output $OUT_DIR/merged_task_arith_${lambda}.pth
done

# Orthogonal merge with best parameters from prior sweep
echo ""
echo "Creating orthogonal merge..."
python merge_models.py --method orthogonal \
    --model_mae $MAE_PP \
    --model_contrastive $CON_PP \
    --model_base $BASE_PATH \
    --ortho_beta 1.0 \
    --ortho_tau 0.0 \
    --use_contrastive_for_norms \
    --output $OUT_DIR/merged_orthogonal.pth

# Layer-wise merge
echo ""
echo "Creating layer-wise merge..."
python merge_models.py --method layerwise \
    --model_mae $MAE_PP \
    --model_contrastive $CON_PP \
    --model_base $BASE_PATH \
    --use_contrastive_for_norms \
    --output $OUT_DIR/merged_layerwise.pth

echo ""
echo "=========================================="
echo "Model merging completed!"
echo "=========================================="
echo ""
echo "Created models:"
ls -la $OUT_DIR/*.pth
