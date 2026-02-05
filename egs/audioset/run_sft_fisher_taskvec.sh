#!/bin/bash
#SBATCH --job-name=sft-fisher-taskvec
#SBATCH --partition=h100-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --exclude=mlcbm005,mlcbm012
#SBATCH --output=log/%j_sft_fisher_taskvec.txt
#SBATCH --error=log/%j_sft_fisher_taskvec.err

set -x

# Activate environment
source /weka/kuehne/kqr867/code/cav-mae/activate_env.sh
export TORCH_HOME=/weka/kuehne/kqr867/code/cav-mae/pretrained_models

model=cav-mae-ft
ftmode=multimodal

# Fisher task-vec merged model
pretrain_path=/weka/kuehne/kqr867/code/cav-mae/egs/audioset/exp/merged-models/fisher-sweep/merged_fisher_taskvec.pth

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
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=36
label_smooth=0.1

dataset=audioset
tr_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_20k_cleaned_auto_train_newval.json
te_data=/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_20k_cleaned_auto_val_newval.json
label_csv=/weka/kuehne/kqr867/code/cav-mae/src/preprocess/sample_datafiles/class_labels_indices_as.csv

exp_dir=./exp/sft-fisher-taskvec-5e-5-bs36-epoch15-$(date +%Y%m%d_%H%M%S)
mkdir -p $exp_dir

echo "=========================================="
echo "SFT for Fisher Task-Vec Merged Model"
echo "Pretrain path: ${pretrain_path}"
echo "Exp dir: ${exp_dir}"
echo "=========================================="

CUDA_CACHE_DISABLE=1 python -W ignore /weka/kuehne/kqr867/code/cav-mae/src/run_cavmae_ft.py --model ${model} --dataset ${dataset} \
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
--num-workers 32 --skip_frame_agg False

echo "SFT Fisher Task-Vec completed!"
