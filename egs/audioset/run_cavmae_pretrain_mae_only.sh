#!/bin/bash
#SBATCH --job-name="cavmae-as2m-mae"
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=./log/%j_mae_only.txt
#SBATCH --error=./log/%j_mae_only.err

# CAV-MAE pretraining with MAE OBJECTIVE ONLY (no contrastive loss)
# For model merging experiments

set -x

# Activate CAV-MAE environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh

export TORCH_HOME=../../pretrained_models
cd /weka/kuehne/kqr867/code/cav-mae/egs/audioset

model=cav-mae
masking_ratio=0.75
mask_mode=unstructured
contrast_loss_weight=0.0  # DISABLED: No contrastive loss
mae_loss_weight=1.0       # ENABLED: MAE reconstruction loss only
tr_pos=False
norm_pix_loss=True

# Download pretrained vision-MAE checkpoint if not exists
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/9nlz523a5q52w86/ori_mae_11.pth?dl=1 -O IN-initial.pth
pretrain_path=${cur_dir}/IN-initial.pth

bal=None
lr=1e-4
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=120
lr_adapt=False

dataset=audioset
tr_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_2m_pretrain.json
te_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_eval_yuan.json
label_csv=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

exp_dir=./exp/mae-only-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-mr-${mask_mode}-${masking_ratio}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_cavmae_pretrain.py --model ${model} --dataset ${dataset} \
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
--tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --mask_mode ${mask_mode}
