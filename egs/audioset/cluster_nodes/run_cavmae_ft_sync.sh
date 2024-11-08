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

# print help if -h is called
if [ "$1" = "-h" ]; then
    echo "Usage: $0 [lr] [batch_size] [ftmode] [cuda_devices] [aggregate] [num_workers] [freeze_base] [num_samples] [epoch] [neptune_tag] [pretrain_path] [cls_token] [n_register_tokens] [total_frame]"
    exit 0
fi

export TORCH_HOME=../../pretrained_models

model=cav-mae-ft

bal=None
lr=${1}
batch_size=${2}
ftmode=${3}
cuda_devices=${4}
aggregate=${5}
total_frame=${6}
freeze_base=${7}
num_samples=${8}
epoch=${9}
neptune_tag=${10}
pretrain_model=${11}
cls_token=${12}
n_register_tokens=${13}
total_frames=${14}
tr_data=${15}
te_data=${16}
label_csv=${17}
lr_scheduler=${18}
target_length=${19}
contrastive_head=${20}
joint_layers=${21}
num_workers=${22}
keep_register_tokens=${23}

# Get pretrain path from models.csv
pretrain_path=$(awk -F, -v model="$pretrain_model" '$1 == model {print $2}' models.csv)

# Set other default parameters that aren't passed
head_lr=1  # since freeze_base is True
lrscheduler_start=5
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=13
wa_end=25
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
noise=True
freqm=48
timem=192
mixup=0.5
label_smooth=0.1
dataset=audioset

# tr_data, te_data, and label_csv are now passed from the calling script

exp_dir=./exp/testmae02-${dataset}-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5-$(date +%Y%m%d_%H%M%S)
mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=${cuda_devices} CUDA_CACHE_DISABLE=1 python -W ignore src/run_cavmae_ft_sync.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss BCE --metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
-w 16 --aggregate ${aggregate} --lr_scheduler ${lr_scheduler} \
--num_samples ${num_samples} --neptune_tag ${neptune_tag} --cls_token ${cls_token} \
--n_register_tokens ${n_register_tokens} --total_frame ${total_frame} --model_id ${pretrain_model} \
--contrastive_head ${contrastive_head} --joint_layers ${joint_layers} --keep_register_tokens ${keep_register_tokens}
