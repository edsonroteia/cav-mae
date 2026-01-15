#!/bin/bash
#SBATCH --job-name="cavmae-as-ft"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=./log/%j_ft.txt
#SBATCH --error=./log/%j_ft.err

# Finetune CAV-MAE pretrained model on AudioSet 20k balanced
# Usage: sbatch run_cavmae_ft.sh [pretrain_path]
# If pretrain_path not provided, downloads default CAV-MAE-Scale++

set -x

# Activate environment (expanded from tenv alias)
pushd /home/kuehne/kqr867/code/avllm-eval/training >/dev/null && source activate_env.sh && popd >/dev/null

export TORCH_HOME=../../pretrained_models
cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

model=cav-mae-ft
ftmode=multimodal

# Get pretrain path from argument or use default
if [ -n "$1" ]; then
    pretrain_path=$1
    # Extract model name for experiment directory
    model_name=$(basename $pretrain_path .pth)
else
    # Download default CAV-MAE-Scale++ if not provided
    cur_dir=$(pwd)
    wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
    pretrain_path=${cur_dir}/cav-mae-scale++.pth
    model_name="scale++"
fi

echo "Using pretrained model: ${pretrain_path}"

freeze_base=False
head_lr=10 # newly initialized ft layers uses 10 times larger than the base lr

bal=bal
lr=1e-4
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=3
wa_end=10
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=48
label_smooth=0.1

dataset=audioset
tr_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_20k_cleaned_auto_train_newval.json
te_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_20k_cleaned_auto_val_newval.json
label_csv=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

exp_dir=./exp/ft-${model_name}-${dataset}-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-${ftmode}-fz${freeze_base}-h${head_lr}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32
