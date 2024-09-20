#!/bin/bash
#SBATCH -p a5
#SBATCH --qos regular
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="vgg-ft"
#SBATCH --output=../log/%j_vgg_ft.txt

# finetune cav-mae pretrained on AS-2M with VGGSound dataset
# you can change pretrain_path to other cav-mae models

export TORCH_HOME=../../pretrained_models

model=cav-mae-ft

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
# pretrain_dir=/local/$SLURM_JOB_ID/models/
pretrain_dir=/scratch/ssml/araujo/exp/sync-audioset-cav-mae-balNone-lr2e-4-epoch25-bs512-normTrue-c0.1-p1.0-tpFalse-mr-unstructured-0.75-20240912_021700/models/
pretrain_path=${pretrain_dir}/best_audio_model.pth

freeze_base=False
head_lr=10 # newly initialized ft layers uses 10 times larger than the base lr

bal=bal
lr=${1:-1e-4}  # Use the first argument as lr, default to 1e-4 if not provided
batch_size=${2:-48}  # Use the second argument as batch_size, default to 24 if not provided
ftmode=${3:-multimodal}
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
target_length=96
noise=True
freqm=48
timem=192
mixup=0.5
label_smooth=0.1
lr_scheduler=cosine

dataset=vggsound
tr_data=datafilles/vggsound/cluster_nodes/vgg_train_cleaned.json
te_data=datafilles/vggsound/cluster_nodes/vgg_test_cleaned.json
label_csv=datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv

exp_dir=./exp/testmae02-${dataset}-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5-$(date +%Y%m%d_%H%M%S)
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore src/run_cavmae_ft_sync.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 309 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32 --aggregate self_attention_cls --lr_scheduler ${lr_scheduler}
