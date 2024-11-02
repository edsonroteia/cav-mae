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

model_id=${2:-2776}
n_register_tokens=$(awk -F, -v model="$model_id" '$1 == model {print $3}' models.csv)
total_frames=$(awk -F, -v model="$model_id" '$1 == model {print $4}' models.csv)
target_length=$(awk -F, -v model="$model_id" '$1 == model {print $6}' models.csv)
contrastive_head=$(awk -F, -v model="$model_id" '$1 == model {print $5}' models.csv)
joint_layers=$(awk -F, -v model="$model_id" '$1 == model {print $7}' models.csv)
lr_scheduler=${3:-cosine}

lr1=1e-3
lr2=3e-3
lr3=5e-3
lr4=1e-2

# Print all variables
echo "num_classes: $num_classes"
echo "model_id: $model_id"
echo "n_register_tokens: $n_register_tokens"
echo "total_frames: $total_frames"
echo "target_length: $target_length"
echo "contrastive_head: $contrastive_head"
echo "joint_layers: $joint_layers"

# Print the commands that will be run
echo "Running the following commands:"
echo "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr1 48 multimodal 0,1 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
echo "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr2 48 multimodal 2,3 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
echo "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr3 48 multimodal 4,5 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
echo "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr4 48 multimodal 6,7 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"

# Create windows for training runs
tmux new-window -n "${lr1} ${model_id}" "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr1 48 multimodal 0,1 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
tmux new-window -n "${lr2} ${model_id}" "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr2 48 multimodal 2,3 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
tmux new-window -n "${lr3} ${model_id}" "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr3 48 multimodal 4,5 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"
tmux new-window -n "${lr4} ${model_id}" "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr4 48 multimodal 6,7 self_attention_cls $total_frames True 99999 10 finetuning_vggsound $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $num_classes $lr_scheduler $target_length $contrastive_head $joint_layers"

# Command that will be run in each window if default values are used
# bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh 1e-2 48 multimodal 4,5,6,7 self_attention_cls 32 True 99999 10 finetuning_vggsound 2918 True 8 16 datafilles/vggsound/cluster_nodes/vgg_train_20.json datafilles/vggsound/cluster_nodes/vgg_test_20.json datafilles/vggsound/cluster_nodes/class_labels_indices_vgg_20.csv;
