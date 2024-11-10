#!/bin/bash
#SBATCH -p a5
#SBATCH --qos regular
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=120000
#SBATCH --job-name="extract-feat"
#SBATCH --output=../log/%j_extract_feat.txt

export TORCH_HOME=../../pretrained_models

# Get pretrain path from models.csv
pretrain_path=$(awk -F, -v model="$1" '$1 == model {print $2}' models.csv)

# Required params with defaults
batch_size=64
cuda_devices=${2:-0,1,2,3}
# Get values from models.csv
# name,path,num_register_tokens,total_frame,contrastive_head,target_length,joint_layers,keep_register_tokens
n_register_tokens=$(awk -F, -v model="$1" '$1 == model {print $3}' models.csv)
total_frame=$(awk -F, -v model="$1" '$1 == model {print $4}' models.csv)
contrastive_head=$(awk -F, -v model="$1" '$1 == model {print $5}' models.csv)
target_length=$(awk -F, -v model="$1" '$1 == model {print $6}' models.csv)
joint_layers=$(awk -F, -v model="$1" '$1 == model {print $7}' models.csv)
keep_register_tokens=$(awk -F, -v model="$1" '$1 == model {print $8}' models.csv)
aggregate="None"
tr_data=${3:-"datafilles/audioset_20k/cluster_nodes/audioset_20k_cleaned.json"}
te_data=${4:-"datafilles/audioset_20k/cluster_nodes/audioset_eval_cleaned_aug24.json"}
label_csv=${5:-"datafilles/audioset_20k/cluster_nodes/class_labels_indices.csv"}
num_workers=64
cls_token=True

# print all params
echo "Pretrain path: ${pretrain_path}"
echo "Batch size: ${batch_size}"
echo "Cuda devices: ${cuda_devices}"
echo "Number of register tokens: ${n_register_tokens}"
echo "Total frame: ${total_frame}"
echo "Contrastive head: ${contrastive_head}"
echo "Target length: ${target_length}"
echo "Joint layers: ${joint_layers}"
echo "Keep register tokens: ${keep_register_tokens}"

# Set other default parameters
dataset=audioset
dataset_mean=-5.081
dataset_std=4.4849
noise=True

exp_dir=./exp/feature_extraction-$(date +%Y%m%d_%H%M%S)
mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=${cuda_devices} CUDA_CACHE_DISABLE=1 python -W ignore src/run_extraction_sync.py \
--dataset ${dataset} \
--data-train ${tr_data} \
--data-val ${te_data} \
--exp-dir ${exp_dir} \
--label-csv ${label_csv} \
--n_class 527 \
--batch-size ${batch_size} \
--dataset_mean ${dataset_mean} \
--dataset_std ${dataset_std} \
--target_length ${target_length} \
--noise ${noise} \
--pretrain_path ${pretrain_path} \
--aggregate ${aggregate} \
-w ${num_workers} \
--cls_token ${cls_token} \
--n_register_tokens ${n_register_tokens} \
--total_frame ${total_frame} \
--model_id ${1} \
--contrastive_head ${contrastive_head} \
--joint_layers ${joint_layers} \
--keep_register_tokens ${keep_register_tokens}
