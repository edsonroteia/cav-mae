#!/bin/bash
#SBATCH --job-name="ret-scalep"
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --exclude=mlcbm012,mlcbm005,mlcbm004,mlcbm003
#SBATCH --output=./log/%j_retrieval_scalep.txt

set -x

# Activate the cav-mae environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
cd /weka/kuehne/kqr867/code/cav-mae

# Model path: Original CAV-MAE Scale+ (from Yuan Gong et al.)
MODEL_PATH=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/cav-mae-scale+.pth

# VGGSound retrieval data
DATA_JSON=/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv

# Output path
OUTPUT_CSV=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/retrieval_results/original_scalep.csv
mkdir -p $(dirname $OUTPUT_CSV)

echo "=========================================="
echo "Retrieval Evaluation: Original CAV-MAE Scale+"
echo "From: Yuan Gong et al. (ICLR 2023)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_CSV}"
echo "=========================================="

python src/run_retrieval.py \
    --model_type cavmae \
    --model_path ${MODEL_PATH} \
    --data_json ${DATA_JSON} \
    --label_csv ${LABEL_CSV} \
    --batch_size 48 \
    --output ${OUTPUT_CSV}

echo "Retrieval Scale+ completed!"
echo "Results saved to: ${OUTPUT_CSV}"
