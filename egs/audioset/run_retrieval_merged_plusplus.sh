#!/bin/bash
#SBATCH --job-name=retrieval-merged++
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --exclude=mlcbm005,mlcbm008,mlcbm012
#SBATCH --output=log/%j_retrieval_merged_plusplus.txt
#SBATCH --error=log/%j_retrieval_merged_plusplus.err

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
export TORCH_HOME=/weka/kuehne/kqr867/code/cav-mae/pretrained_models

cd /weka/kuehne/kqr867/code/cav-mae

# Create output directory
mkdir -p egs/audioset/exp/retrieval_results/merged-plusplus

# Data paths
DATA_JSON=/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv

# Merged models directory
MERGED_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models-plusplus
OUTPUT_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/retrieval_results/merged-plusplus

echo "=========================================="
echo "Retrieval Evaluation for Merged ++ Models"
echo "=========================================="

# Count models
num_models=$(ls -1 $MERGED_DIR/*.pth 2>/dev/null | wc -l)
echo "Found $num_models merged models to evaluate"
echo ""

# Evaluate each merged model
for model_path in $MERGED_DIR/*.pth; do
    if [[ -f "$model_path" ]]; then
        model_name=$(basename "$model_path" .pth)
        output_file="$OUTPUT_DIR/${model_name}.csv"

        echo "Evaluating: $model_name"

        python src/run_retrieval.py \
            --model_type cavmae \
            --model_path "$model_path" \
            --data_json $DATA_JSON \
            --label_csv $LABEL_CSV \
            --batch_size 48 \
            --output "$output_file"

        echo ""
    fi
done

echo "=========================================="
echo "Retrieval evaluation completed!"
echo "Results saved to $OUTPUT_DIR"
echo "=========================================="

# Generate summary table
echo ""
echo "=========================================="
echo "SUMMARY OF RESULTS"
echo "=========================================="
echo ""
echo "Model,A2V_R1,A2V_R5,A2V_R10,A2V_MR,V2A_R1,V2A_R5,V2A_R10,V2A_MR,Avg_R1"

for csv_file in $OUTPUT_DIR/*.csv; do
    if [[ -f "$csv_file" ]]; then
        cat "$csv_file"
    fi
done | sort -t',' -k9 -nr | head -20

echo ""
echo "Top 5 models by Average R@1:"
for csv_file in $OUTPUT_DIR/*.csv; do
    if [[ -f "$csv_file" ]]; then
        cat "$csv_file"
    fi
done | sort -t',' -k9 -nr | head -5
