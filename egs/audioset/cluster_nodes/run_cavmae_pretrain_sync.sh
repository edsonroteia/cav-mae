#!/bin/bash
##SBATCH -p a5
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-[1,2,3],sls-sm-[5,6,7,12]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="as-pretrain"
#SBATCH --output=../log/%j_as_pretrain.txt

# run cav-mae pretraining, use smaller lr and batch size, fits smaller GPUs (4*12GB GPUs)

export TORCH_HOME=../../pretrained_models

model=cav-mae
masking_ratio=0.75
mask_mode=unstructured # or time, or freq, or tf
contrast_loss_weight=0.01
mae_loss_weight=1.0
tr_pos=False
norm_pix_loss=True

cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/9nlz523a5q52w86/ori_mae_11.pth?dl=1 -O IN-initial.pth
pretrain_path=${cur_dir}/IN-initial.pth

bal=None
lr=2e-4
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=96
noise=True
mixup=0.0
batch_size=256
lr_adapt=False

dataset=audioset
tr_data=datafilles/audioset_2m/cluster_nodes/audioset_2m_cleaned_aug24.json
te_data=datafilles/audioset_2m/cluster_nodes/audioset_eval_cleaned_aug24.json
label_csv=datafilles/class_labels_indices.csv

sed -i "s|<SLURM_JOB_ID>|$SLURM_JOB_ID|g" ${tr_data}
sed -i "s|<SLURM_JOB_ID>|$SLURM_JOB_ID|g" ${te_data}

exp_dir=/scratch/ssml/araujo/exp/sync-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}-$(date +%Y%m%d_%H%M%S)
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore src/run_cavmae_pretrain_sync.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 527 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--mixup ${mixup} --bal ${bal} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --mask_mode ${mask_mode} \
# --wandb-name sync_pt_as2m_$(hostname)_lr${lr}_epoch${epoch}

