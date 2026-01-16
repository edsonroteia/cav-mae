#!/bin/bash
#SBATCH --job-name=retrieval
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=log/%j_retrieval.txt

# Retrieval evaluation on VGGSound for CAV-JEPA and baseline models

set -x

cd /weka/kuehne/kqr867/code/cav-mae
source activate_env.sh

# Paths
DATA_JSON="datafiles/vgg_test_5_per_class_for_retrieval.json"
LABEL_CSV="datafiles/class_labels_indices_vgg.csv"

# CAV-JEPA model (use best or latest checkpoint)
CAVJEPA_MODEL="egs/audioset/exp/cavjepa-audioset-lr1e-4-epoch25-bs120-mr0.75-mom0.996-0.999-pred4/models/best_audio_model.pth"

echo "=============================================="
echo "Evaluating CAV-JEPA on VGGSound Retrieval"
echo "=============================================="

python src/run_retrieval.py \
    --model_type cavjepa \
    --model_path ${CAVJEPA_MODEL} \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output egs/audioset/retrieval_results_cavjepa.csv

echo ""
echo "Done!"
