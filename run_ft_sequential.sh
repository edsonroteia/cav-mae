#!/bin/bash

num_classes=${1:-20}
model_id=${2:-2776}
lr1=${3:-1e-3}
lr2=${4:-1e-4}
lr_scheduler=${5:-cosine}

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

# Function to get model parameters
get_model_params() {
    local model=$1
    local n_register_tokens=$(awk -F, -v model="$model" '$1 == model {print $3}' models.csv)
    local total_frames=$(awk -F, -v model="$model" '$1 == model {print $4}' models.csv)
    local target_length=$(awk -F, -v model="$model" '$1 == model {print $6}' models.csv)
    local contrastive_head=$(awk -F, -v model="$model" '$1 == model {print $5}' models.csv)
    local joint_layers=$(awk -F, -v model="$model" '$1 == model {print $7}' models.csv)
    local keep_register_tokens=$(awk -F, -v model="$model" '$1 == model {print $8}' models.csv)
    echo "$n_register_tokens $total_frames $target_length $contrastive_head $joint_layers $keep_register_tokens"
}

# Get model parameters once since we're using the same model
read n_register_tokens total_frames target_length contrastive_head joint_layers keep_register_tokens <<< $(get_model_params $model_id)

# Define configurations for the 4 runs
configs=(
    "audioonly $lr1"    # GPU 0,1: audioonly with lr1
    "videoonly $lr1"    # GPU 2,3: videoonly with lr1
    "audioonly $lr2"    # GPU 4,5: audioonly with lr2
    "videoonly $lr2"    # GPU 6,7: videoonly with lr2
)

# Create windows for each configuration
for i in {0..3}; do
    modality=$(echo ${configs[$i]} | cut -d' ' -f1)
    lr=$(echo ${configs[$i]} | cut -d' ' -f2)
    gpu_ids=$(( i*2 )),$(( i*2+1 ))
    
    tmux new-window -n "${modality}_${lr}" "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr 48 $modality $gpu_ids self_attention_cls $total_frames True 9999999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers $keep_register_tokens"
done

# Command that will be run in each window if default values are used
# bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh 1e-2 48 multimodal 4,5,6,7 self_attention_cls 32 True 99999 10 finetuning_vggsound 2918 True 8 16 datafilles/vggsound/cluster_nodes/vgg_train_20.json datafilles/vggsound/cluster_nodes/vgg_test_20.json datafilles/vggsound/cluster_nodes/class_labels_indices_vgg_20.csv;
