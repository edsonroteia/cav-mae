#!/bin/bash

num_classes=${1:-20}

# 1. Create subsampled dataset with 30 classes
echo "Creating subsampled dataset with $num_classes classes..."
python datafilles/vggsound/vggsound_subsample.py --num_classes $num_classes

# 2. Clean json files
echo "Cleaning json files..."
python datafilles/clean_json_files.py datafilles/vggsound/cluster_nodes

# 3. Create new windows in current tmux session
SESSION=$(tmux display-message -p '#S')

# Create windows for monitoring
tmux new-window -n "htop" "htop"
tmux new-window -n "gpu" "watch -n 1 bash ~/brocm-smi.sh"

tr_data=datafilles/vggsound/cluster_nodes/vgg_train_${num_classes}.json
te_data=datafilles/vggsound/cluster_nodes/vgg_test_${num_classes}.json
label_csv=datafilles/vggsound/cluster_nodes/class_labels_indices_vgg_${num_classes}.csv

# Single learning rate, multiple model IDs
lr=5e-3
model_id1=${2:-2776}
model_id2=${3:-2777}
model_id3=${4:-2778}
model_id4=${5:-2779}
lr_scheduler=${6:-cosine}

# Function to get model parameters
get_model_params() {
    local model=$1
    local n_register_tokens=$(awk -F, -v model="$model" '$1 == model {print $3}' models.csv)
    local total_frames=$(awk -F, -v model="$model" '$1 == model {print $4}' models.csv)
    local target_length=$(awk -F, -v model="$model" '$1 == model {print $6}' models.csv)
    local contrastive_head=$(awk -F, -v model="$model" '$1 == model {print $5}' models.csv)
    local joint_layers=$(awk -F, -v model="$model" '$1 == model {print $7}' models.csv)
    echo "$n_register_tokens $total_frames $target_length $contrastive_head $joint_layers"
}

# Create windows for each model ID
for i in 1 2 3 4; do
    model_var="model_id$i"
    model_id=${!model_var}
    read n_register_tokens total_frames target_length contrastive_head joint_layers <<< $(get_model_params $model_id)
    
    gpu_ids=$(( (i-1)*2 )),$(( (i-1)*2+1 ))
    
    tmux new-window -n "model_${model_id}" "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr 48 multimodal $gpu_ids self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
done

# Command that will be run in each window if default values are used
# bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh 1e-2 48 multimodal 4,5,6,7 self_attention_cls 32 True 99999 10 finetuning_vggsound 2918 True 8 16 datafilles/vggsound/cluster_nodes/vgg_train_20.json datafilles/vggsound/cluster_nodes/vgg_test_20.json datafilles/vggsound/cluster_nodes/class_labels_indices_vgg_20.csv;
