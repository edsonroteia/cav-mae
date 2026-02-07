#!/bin/bash
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=mlcbm005
#SBATCH --job-name="cavjepa-ablation"
#SBATCH --output=./log/%j_cavjepa_ablation.txt
#SBATCH --error=./log/%j_cavjepa_ablation.err

# CAV-JEPA Ablation Study
#
# Run different configurations to understand the impact of each improvement:
# 1. Initialization: MAE vs Random
# 2. Target normalization: On vs Off
# 3. Scheduler: v1 (step decay) vs v2 (warmup + cosine)
# 4. Training length: Short (25 epochs) vs Long (100 epochs)
#
# Usage:
#   sbatch run_cavjepa_ablations.sh <experiment_id>
#
# Experiment IDs:
#   E1a: MAE init, 25 epochs (baseline)
#   E1b: MAE init, 100 epochs
#   E1c: Random init, 100 epochs (expected best)
#   E1d: Random init, 300 epochs (full I-JEPA style)
#   E2a: Random init, v1 scheduler
#   E2b: Random init, v2 scheduler (warmup + cosine)
#   E3a: Random init, no target norm
#   E3b: Random init, with target norm
#
# === NEW SOTA-FOCUSED EXPERIMENTS ===
#   A1:  Higher contrastive weight (0.1) - stronger alignment signal
#   A2a: Contrastive sweep (0.03)
#   A2b: Contrastive sweep (0.05)
#   A2c: Contrastive sweep (0.20)
#   A2d: Contrastive sweep (0.50)
#   B1:  Multiblock masking - semantic-level prediction
#   C1:  Deeper predictor (6 layers) - more prediction capacity
#   C2:  Deeper + wider predictor (6 layers, dim 512)
#   D1:  Higher momentum schedule (0.999 -> 0.9999)
#   M1:  Lower mask ratio (0.60)
#   M2:  Higher mask ratio (0.85)

set -euo pipefail
set -x

# Get experiment ID from argument
EXPERIMENT=${1:-E1a}

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

# Common parameters
model=cav-jepa
dataset=audioset
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=120
bal=None

# JEPA parameters
jepa_loss_weight=1.0
contrast_loss_weight=0.01
masking_ratio=0.75
momentum_start=0.996
momentum_end=0.999
predictor_depth=4
predictor_dim=384
tr_pos=False
init_std=0.02

# Data paths
cur_dir=$(pwd)
tr_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_2m_pretrain.json
te_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_eval_yuan.json
label_csv=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

# Pretrained weights path (only used for MAE init)
pretrain_path_mae=${cur_dir}/cav_jepa_from_mae_init.pth

# Default to random init with v2 scheduler
init_mode=random
normalize_targets=True
use_v2_scheduler=True
use_multiblock_masking=False
warmup_epochs=15
start_lr=1e-4
ref_lr=1e-3
final_lr=1e-6
start_wd=0.04
final_wd=0.4
epoch=100
pretrain_path=None

# Configure based on experiment ID
case $EXPERIMENT in
    # === Experiment 1: Initialization ablation ===
    E1a)
        # Baseline: MAE init, 25 epochs (original v1 behavior)
        init_mode=mae
        epoch=25
        use_v2_scheduler=False
        pretrain_path=$pretrain_path_mae
        exp_name="E1a-mae_init-25ep-v1sched"
        ;;
    E1b)
        # MAE init, longer training
        init_mode=mae
        epoch=100
        use_v2_scheduler=True
        pretrain_path=$pretrain_path_mae
        exp_name="E1b-mae_init-100ep-v2sched"
        ;;
    E1c)
        # Random init, 100 epochs (expected improvement)
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        exp_name="E1c-random_init-100ep-v2sched"
        ;;
    E1d)
        # Random init, full I-JEPA training length
        init_mode=random
        epoch=300
        use_v2_scheduler=True
        exp_name="E1d-random_init-300ep-v2sched"
        ;;

    # === Experiment 2: Scheduler ablation ===
    E2a)
        # Random init with v1 scheduler (step decay)
        init_mode=random
        epoch=100
        use_v2_scheduler=False
        exp_name="E2a-random_init-v1sched"
        ;;
    E2b)
        # Random init with v2 scheduler (warmup + cosine)
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        exp_name="E2b-random_init-v2sched"
        ;;

    # === Experiment 3: Target normalization ablation ===
    E3a)
        # Random init, NO target normalization
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=False
        exp_name="E3a-random_init-no_target_norm"
        ;;
    E3b)
        # Random init, WITH target normalization
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        exp_name="E3b-random_init-with_target_norm"
        ;;

    # === NEW SOTA-FOCUSED EXPERIMENTS ===
    A1)
        # Higher contrastive weight for stronger alignment signal
        # Hypothesis: Direct path to better retrieval performance
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        contrast_loss_weight=0.1  # 10x increase from default 0.01
        exp_name="A1-high_contrast_weight"
        ;;
    A2a)
        # Contrastive weight sweep: light increase
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        contrast_loss_weight=0.03
        exp_name="A2a-contrast_sweep_0.03"
        ;;
    A2b)
        # Contrastive weight sweep: moderate increase
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        contrast_loss_weight=0.05
        exp_name="A2b-contrast_sweep_0.05"
        ;;
    A2c)
        # Contrastive weight sweep: high contrastive dominance
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        contrast_loss_weight=0.2
        exp_name="A2c-contrast_sweep_0.2"
        ;;
    A2d)
        # Contrastive weight sweep: near contrastive-only regime
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        contrast_loss_weight=0.5
        exp_name="A2d-contrast_sweep_0.5"
        ;;
    B1)
        # Multiblock masking for semantic-level prediction
        # Hypothesis: Better semantic representations via structured masking
        # NOTE: Requires integration of src/masks/multiblock.py into training loop
        # Status: PENDING - will raise NotImplementedError until integrated
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        use_multiblock_masking=True
        exp_name="B1-multiblock_masking"
        ;;
    C1)
        # Deeper predictor for more prediction capacity
        # Hypothesis: Better feature transformation with deeper predictor
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        predictor_depth=6  # Increased from default 4
        exp_name="C1-deeper_predictor"
        ;;
    C2)
        # Deeper + wider predictor
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        predictor_depth=6
        predictor_dim=512
        exp_name="C2-deeper_wider_predictor"
        ;;
    D1)
        # Higher momentum schedule for more stable targets
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        momentum_start=0.999
        momentum_end=0.9999
        exp_name="D1-high_momentum"
        ;;
    M1)
        # Lower mask ratio: easier prediction target
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        masking_ratio=0.60
        exp_name="M1-mask_ratio_0.60"
        ;;
    M2)
        # Higher mask ratio: harder prediction target
        init_mode=random
        epoch=100
        use_v2_scheduler=True
        normalize_targets=True
        masking_ratio=0.85
        exp_name="M2-mask_ratio_0.85"
        ;;

    # === Experiment 4: Quick test run ===
    TEST)
        # Quick test with 1 epoch
        init_mode=random
        epoch=1
        use_v2_scheduler=True
        normalize_targets=True
        exp_name="TEST-quick_validation"
        ;;

    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available experiments:"
        echo "  Ablations: E1a, E1b, E1c, E1d, E2a, E2b, E3a, E3b"
        echo "  SOTA-focused: A1, A2a, A2b, A2c, A2d, B1, C1, C2, D1, M1, M2"
        echo "  Debug: TEST"
        exit 1
        ;;
esac

# Experiment directory
exp_dir=./exp/ablation-${exp_name}
mkdir -p $exp_dir
mkdir -p ./log

echo "============================================"
echo "Running CAV-JEPA Ablation: $EXPERIMENT"
echo "============================================"
echo "Experiment: $exp_name"
echo "Directory: $exp_dir"
echo ""
echo "Configuration:"
echo "  Init mode: $init_mode"
echo "  Epochs: $epoch"
echo "  V2 scheduler: $use_v2_scheduler"
echo "  Normalize targets: $normalize_targets"
echo "  Multiblock masking: $use_multiblock_masking"
echo "  Contrastive weight: $contrast_loss_weight"
echo "  Predictor depth: $predictor_depth"
echo "  Pretrain path: $pretrain_path"
echo "============================================"

# Build command arguments
cmd_args=(
    --model ${model}
    --dataset ${dataset}
    --data-train ${tr_data}
    --data-val ${te_data}
    --exp-dir $exp_dir
    --label-csv ${label_csv}
    --n_class 527
    --lr ${ref_lr}
    --n-epochs ${epoch}
    --batch-size $batch_size
    --save_model True
    --mixup ${mixup}
    --bal ${bal}
    --dataset_mean ${dataset_mean}
    --dataset_std ${dataset_std}
    --target_length ${target_length}
    --noise ${noise}
    --warmup True
    --lr_adapt False
    --pretrain_path ${pretrain_path}
    --jepa_loss_weight ${jepa_loss_weight}
    --contrast_loss_weight ${contrast_loss_weight}
    --masking_ratio ${masking_ratio}
    --momentum_start ${momentum_start}
    --momentum_end ${momentum_end}
    --predictor_depth ${predictor_depth}
    --predictor_dim ${predictor_dim}
    --tr_pos ${tr_pos}
    --init_mode ${init_mode}
    --normalize_targets ${normalize_targets}
    --use_multiblock_masking ${use_multiblock_masking}
    --init_std ${init_std}
    --use_v2_scheduler ${use_v2_scheduler}
    --warmup_epochs ${warmup_epochs}
    --start_lr ${start_lr}
    --ref_lr ${ref_lr}
    --final_lr ${final_lr}
    --start_wd ${start_wd}
    --final_wd ${final_wd}
    --use_wandb True
    --wandb_project "cav-jepa-ablations"
    --wandb_run_name "${exp_name}"
)

# V1 scheduler specific args
if [ "$use_v2_scheduler" = "False" ]; then
    cmd_args+=(
        --lrscheduler_start 10
        --lrscheduler_decay 0.5
        --lrscheduler_step 5
    )
fi

if CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavjepa_pretrain.py "${cmd_args[@]}"; then
    echo "============================================"
    echo "Ablation $EXPERIMENT completed!"
    echo "Results saved to: $exp_dir"
    echo "============================================"
else
    rc=$?
    echo "============================================"
    echo "Ablation $EXPERIMENT FAILED (exit code: $rc)"
    echo "Check logs in: ./log"
    echo "============================================"
    exit $rc
fi
