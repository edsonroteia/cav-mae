#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --job-name="as-bal-ft"
#SBATCH --output=/scratch/ssml/araujo/logs/initial_tests/%j_as_ft.txt
#SBATCH --mail-type=FAIL

source /scratch/ssml/araujo/envs/anaconda3/etc/profile.d/conda.sh
conda activate cavmae_walid_sofian
export WANDB_API_KEY='0c0fe5c36ab12210d1646e9966cd22a92b43783c'
wandb login

export TORCH_HOME=../../pretrained_models

model=cav-mae-ft
ftmode=multimodal

pretrain_dir=/local/$SLURM_JOB_ID/models/
pretrain_path=${pretrain_dir}/best_audio_model.pth

freeze_base=False
head_lr=100 # newly initialized ft layers uses 100 times larger than the base lr

bal=None
lr=5e-5
epoch=15
lrscheduler_start=5
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=3
wa_end=15
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=96
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=72
label_smooth=0.1

dataset=audioset
tr_data=datafilles/audioset_20k_cleaned.json
te_data=datafilles/audioset_eval_cleaned.json
label_csv=datafilles/class_labels_indices.csv

exp_dir=/scratch/ssml/araujo/code/aug24/cav-mae/exp/testmae06-bal-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-$(date +%Y%m%d_%H%M%S)
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore src/run_cavmae_ft_sync.py \
    --model ${model} \
    --dataset ${dataset} \
    --data-train ${tr_data} \
    --data-val ${te_data} \
    --exp-dir $exp_dir \
    --label-csv ${label_csv} \
    --n_class 527 \
    --lr $lr \
    --n-epochs ${epoch} \
    --batch-size $batch_size \
    --save_model True \
    --freqm $freqm \
    --timem $timem \
    --mixup ${mixup} \
    --bal ${bal} \
    --label_smooth ${label_smooth} \
    --lrscheduler_start ${lrscheduler_start} \
    --lrscheduler_decay ${lrscheduler_decay} \
    --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} \
    --dataset_std ${dataset_std} \
    --target_length ${target_length} \
    --noise ${noise} \
    --loss BCE \
    --metrics mAP \
    --warmup True \
    --wa ${wa} \
    --wa_start ${wa_start} \
    --wa_end ${wa_end} \
    --lr_adapt ${lr_adapt} \
    --pretrain_path ${pretrain_path} \
    --ftmode ${ftmode} \
    --freeze_base ${freeze_base} \
    --head_lr ${head_lr} \
    --num-workers 48 \
    --wandb_name original_finetune_balanced