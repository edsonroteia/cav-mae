#!/bin/bash

num_classes=${1:-20}

# 1. Create subsampled dataset with 30 classes
echo "Creating subsampled dataset with $num_classes classes..."
python datafiles/vggsound/vggsound_subsample.py --num_classes $num_classes

# 2. Clean json files
echo "Cleaning json files..."
python datafilles/clean_json_files.py datafilles/vggsound/cluster_nodes

# 3. Create new window in current tmux session
# Get current session name
SESSION=$(tmux display-message -p '#S')
tmux new-window

# Split window into 6 panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v
tmux split-window -v
tmux split-window -v

# Launch htop in top-right pane
tmux send-keys -t 0.1 'htop' C-m

# Launch nvidia-smi watch in bottom-right pane
tmux send-keys -t 0.3 'watch -n 1 bash ~/.brocm-smi.sh' C-m

# let's have as parameters, the learning rate, and the model ID
# Comma-separated learning rates, default "1e-2,1e-3,1e-4,1e-5"
lr1=1e-3
lr2=8e-4
lr3=5e-4
lr4=1e-4
model_id=${2:-2776}  # Use second argument as model_id, default to 2776 if not provided

# In run_cavmae_ft_sync.sh, the 15th argument is the training data, and the 16th argument is the test data, and the 17th argument is the label csv file
# tr_data=${15:-datafilles/vggsound/cluster_nodes/vgg_train_cleaned.json}
# te_data=${16:-datafilles/vggsound/cluster_nodes/vgg_test_cleaned.json}
# label_csv=${17:-datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv}
# We should pass the data and label csv file as arguments to the script according to the number of classes

tr_data=datafilles/vggsound/cluster_nodes/vgg_train_${num_classes}.json
te_data=datafilles/vggsound/cluster_nodes/vgg_test_${num_classes}.json
label_csv=datafilles/vggsound/cluster_nodes/class_labels_indices_vgg_${num_classes}.csv

# Launch training runs in remaining panes with different model IDs
tmux send-keys -t 0.0 "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr1 48 multimodal 0,1 self_attention_cls 16 True 99999 10 finetuning_vggsound $model_id True 4 16 $tr_data $te_data $label_csv" C-m
tmux send-keys -t 0.2 "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr2 48 multimodal 2,3 self_attention_cls 16 True 99999 10 finetuning_vggsound $model_id True 4 16 $tr_data $te_data $label_csv" C-m
tmux send-keys -t 0.4 "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr3 48 multimodal 4,5 self_attention_cls 16 True 99999 10 finetuning_vggsound $model_id True 4 16 $tr_data $te_data $label_csv" C-m
tmux send-keys -t 0.5 "bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh $lr4 96 multimodal 6,7 self_attention_cls 16 True 99999 10 finetuning_vggsound $model_id True 4 16 $tr_data $te_data $label_csv" C-m