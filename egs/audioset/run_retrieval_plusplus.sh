#!/bin/bash
#SBATCH --job-name=retrieval++
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --exclude=mlcbm005,mlcbm012
#SBATCH --output=log/%j_retrieval_plusplus.txt
#SBATCH --error=log/%j_retrieval_plusplus.err

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
export TORCH_HOME=/weka/kuehne/kqr867/code/cav-mae/pretrained_models

cd /weka/kuehne/kqr867/code/cav-mae

# Create output directory
mkdir -p egs/audioset/exp/retrieval_results/plusplus

# Model paths
CONTRASTIVE_PP=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/contrastive++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/models/best_audio_model.pth
MAE_PP=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/mae++-audioset-cav-mae-balNone-lr2e-4-epoch25-bs256-mr-unstructured-0.75/models/best_audio_model.pth

# Data paths
DATA_JSON=/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv

echo "=========================================="
echo "Retrieval Evaluation for ++ Models"
echo "=========================================="

# Evaluate Contrastive++
echo ""
echo "Evaluating Contrastive++ model..."
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path $CONTRASTIVE_PP \
    --data_json $DATA_JSON \
    --label_csv $LABEL_CSV \
    --batch_size 48 \
    --output egs/audioset/exp/retrieval_results/plusplus/contrastive++.csv

# Evaluate MAE++
echo ""
echo "Evaluating MAE++ model..."
python src/run_retrieval.py \
    --model_type cavmae \
    --model_path $MAE_PP \
    --data_json $DATA_JSON \
    --label_csv $LABEL_CSV \
    --batch_size 48 \
    --output egs/audioset/exp/retrieval_results/plusplus/mae++.csv

echo ""
echo "=========================================="
echo "Retrieval evaluation completed!"
echo "Results saved to egs/audioset/exp/retrieval_results/plusplus/"
echo "=========================================="

# Display results
echo ""
echo "Summary of results:"
cat egs/audioset/exp/retrieval_results/plusplus/*.csv
