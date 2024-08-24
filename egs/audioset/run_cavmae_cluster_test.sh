#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --job-name="cavmaev2_as20k-pretrain"
#SBATCH --output=/scratch/ssml/araujo/logs/initial_tests/%j_as_pretrain.txt
#SBATCH --mail-type=FAIL

# run cav-mae pretraining, use smaller lr and batch size, fits smaller GPUs (4*12GB GPUs)

source /scratch/ssml/araujo/envs/anaconda3/etc/profile.d/conda.sh
conda activate cavmae_walid_sofian
export WANDB_API_KEY='0c0fe5c36ab12210d1646e9966cd22a92b43783c'
wandb login

export TORCH_HOME=../../pretrained_models

model=cav-mae
masking_ratio=0.75
mask_mode=unstructured
contrast_loss_weight=0.01
mae_loss_weight=1.0
tr_pos=False
norm_pix_loss=True

pretrain_dir=/scratch/ssml/araujo/assets
wget -nc https://www.dropbox.com/s/9nlz523a5q52w86/ori_mae_11.pth?dl=1 -O ${pretrain_dir}/IN-initial.pth
pretrain_path=${pretrain_dir}/IN-initial.pth

bal=None
lr=5e-5
epoch=15
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=96
noise=False
mixup=0.0
batch_size=660
lr_adapt=True

dataset=audioset
tr_data=/scratch/ssml/araujo/code/aug24/cav-mae/datafilles/audioset_20k_cleaned_auto_train.json
te_data=/scratch/ssml/araujo/code/aug24/cav-mae/datafilles/audioset_20k_cleaned_auto_val.json
label_csv=/scratch/ssml/araujo/code/aug24/cav-mae/datafilles/class_labels_indices.csv

exp_dir=/scratch/ssml/araujo/code/aug24/cav-mae/exp/testmae02-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore src/run_cavmae_pretrain_sync.py \
	--model ${model} \
	--dataset ${dataset} \
	--data-train ${tr_data} \
	--data-val ${te_data} \
	--exp-dir ${exp_dir} \
	--label-csv ${label_csv} \
	--n_class 527 \
	--lr ${lr} \
	--n-epochs ${epoch} \
	--batch-size ${batch_size} \
	--save_model True \
	--mixup ${mixup} \
	--bal ${bal} \
	--lrscheduler_start ${lrscheduler_start} \
	--lrscheduler_decay ${lrscheduler_decay} \
	--lrscheduler_step ${lrscheduler_step} \
	--dataset_mean ${dataset_mean} \
	--dataset_std ${dataset_std} \
	--target_length ${target_length} \
	--noise ${noise} \
	--warmup True \
	--lr_adapt ${lr_adapt} \
	--norm_pix_loss ${norm_pix_loss} \
	--pretrain_path ${pretrain_path} \
	--mae_loss_weight ${mae_loss_weight} \
	--contrast_loss_weight ${contrast_loss_weight} \
	--tr_pos ${tr_pos} \
	--masking_ratio ${masking_ratio} \
	--mask_mode ${mask_mode} \
	--wandb_name cavmae_pretrain_sync \
	--num-workers 48 \
	--num_samples 1320
