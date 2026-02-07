#!/bin/bash
#SBATCH --job-name="ret-cavjepa-ckpt"
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --exclude=mlcbm005
#SBATCH --output=./log/%j_retrieval_cavjepa_ckpt.txt
#SBATCH --error=./log/%j_retrieval_cavjepa_ckpt.err

# Evaluate CAV-JEPA checkpoints every N epochs.
# Usage:
#   sbatch run_retrieval_cavjepa_checkpoints.sh [EXP_DIR] [START_EPOCH] [END_EPOCH] [STEP]
#
# Example:
#   sbatch run_retrieval_cavjepa_checkpoints.sh \
#     /weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/cavjepa-audioset-lr1e-4-epoch25-bs120-mr0.75-mom0.996-0.999-pred4 \
#     5 25 5

set -euo pipefail
set -x

source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
cd /weka/kuehne/kqr867/code/cav-mae

EXP_DIR=${1:-/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/cavjepa-audioset-lr1e-4-epoch25-bs120-mr0.75-mom0.996-0.999-pred4}
START_EPOCH=${2:-5}
END_EPOCH=${3:-25}
STEP=${4:-5}

DATA_JSON=/weka/kuehne/kqr867/code/cav-mae/datafiles/vgg_test_5_per_class_for_retrieval.json
LABEL_CSV=/weka/kuehne/kqr867/code/cav-mae/datafiles/class_labels_indices_vgg.csv

RUN_NAME=$(basename "$EXP_DIR")
OUT_DIR=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/retrieval_results/cavjepa_checkpoints
mkdir -p "$OUT_DIR"
mkdir -p /weka/kuehne/kqr867/code/cav-mae/egs/audioset/log

SUMMARY_CSV="${OUT_DIR}/${RUN_NAME}_e${START_EPOCH}-e${END_EPOCH}-s${STEP}_summary.csv"
echo "epoch,a2v_r1,v2a_r1,avg_r1,a2v_mr,v2a_mr,result_csv,stats_csv" > "$SUMMARY_CSV"

echo "============================================================"
echo "CAV-JEPA Checkpoint Retrieval Evaluation"
echo "Run: $RUN_NAME"
echo "Epochs: ${START_EPOCH}..${END_EPOCH} (step=${STEP})"
echo "Summary: $SUMMARY_CSV"
echo "============================================================"

for ((ep=START_EPOCH; ep<=END_EPOCH; ep+=STEP)); do
    ckpt="${EXP_DIR}/models/audio_model.${ep}.pth"
    if [ ! -f "$ckpt" ]; then
        echo "[WARN] Missing checkpoint for epoch ${ep}: $ckpt"
        continue
    fi

    result_csv="${OUT_DIR}/${RUN_NAME}_epoch${ep}.csv"
    stats_csv="${OUT_DIR}/${RUN_NAME}_epoch${ep}_stats.csv"

    echo "------------------------------------------------------------"
    echo "Evaluating epoch ${ep}"
    echo "Checkpoint: $ckpt"
    echo "------------------------------------------------------------"

    python src/run_retrieval.py \
        --model_type cavjepa \
        --model_path "$ckpt" \
        --data_json "$DATA_JSON" \
        --label_csv "$LABEL_CSV" \
        --batch_size 48 \
        --output "$result_csv" \
        --stats_output "$stats_csv"

    a2v_r1=$(awk -F, 'NR==2 {print $2}' "$result_csv")
    v2a_r1=$(awk -F, 'NR==3 {print $2}' "$result_csv")
    a2v_mr=$(awk -F, 'NR==2 {print $5}' "$result_csv")
    v2a_mr=$(awk -F, 'NR==3 {print $5}' "$result_csv")
    avg_r1=$(awk -v a="$a2v_r1" -v b="$v2a_r1" 'BEGIN {printf "%.6f", (a+b)/2}')

    echo "${ep},${a2v_r1},${v2a_r1},${avg_r1},${a2v_mr},${v2a_mr},${result_csv},${stats_csv}" >> "$SUMMARY_CSV"
done

if [ "$(wc -l < "$SUMMARY_CSV")" -le 1 ]; then
    echo "[ERROR] No checkpoints were successfully evaluated."
    exit 1
fi

best_row=$(tail -n +2 "$SUMMARY_CSV" | sort -t, -k4,4gr | head -n 1)
best_epoch=$(echo "$best_row" | cut -d, -f1)
best_avg=$(echo "$best_row" | cut -d, -f4)

echo "============================================================"
echo "Done. Summary: $SUMMARY_CSV"
echo "Best epoch by average R@1: ${best_epoch} (avg_r1=${best_avg})"
echo "============================================================"
