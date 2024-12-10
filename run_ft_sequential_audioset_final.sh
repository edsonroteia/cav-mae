#!/bin/bash

   # Clean json files for AudioSet
   echo "Cleaning json files..."
   python datafilles/clean_json_files.py datafilles/audioset/cluster_nodes

   # Create new windows in current tmux session
   SESSION=$(tmux display-message -p '#S')

   # Create windows for monitoring
   tmux new-window -n "htop" "htop"
   tmux new-window -n "gpu" "watch -n 1 bash ~/brocm-smi.sh"

   # Update data paths for AudioSet
   tr_data=datafilles/audioset_20k/cluster_nodes/audioset_20k_cleaned.json
   te_data=datafilles/audioset_20k/cluster_nodes/audioset_eval_cleaned_aug24.json
   label_csv=datafilles/audioset_20k/cluster_nodes/class_labels_indices.csv

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

#    # Option 1: 3-3-2 split with 3 learning rates
#    learning_rates=(2e-3 3e-3 4e-3)
#    gpu_assignments=("0,1,2" "3,4,5" "6,7")

   # Option 2: 4-4 split with 2 learning rates
   learning_rates=(1e-5 1e-3)
   gpu_assignments=("0,1,2,3" "4,5,6,7")

   model_id=${1:-2776}
   lr_scheduler=${2:-cosine}

   # Loop over learning rates
   for i in "${!learning_rates[@]}"; do
       lr=${learning_rates[$i]}
       gpu_ids=${gpu_assignments[$i]}
       read n_register_tokens total_frames target_length contrastive_head joint_layers keep_register_tokens <<< $(get_model_params $model_id)

       num_workers=16

       echo "Running command: bash egs/audioset/cluster_nodes/run_cavmae_ft_sync.sh $lr 48 multimodal $gpu_ids self_attention_cls $total_frames False 99999 10 finetuning_audioset $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $lr_scheduler $target_length $contrastive_head $joint_layers $num_workers $keep_register_tokens"
       tmux new-window -n "model_${model_id}_lr_${lr}" "bash egs/audioset/cluster_nodes/run_cavmae_ft_sync.sh $lr 48 multimodal $gpu_ids self_attention_cls $total_frames False 99999 10 finetuning_audioset $model_id True $n_register_tokens $total_frames $tr_data $te_data $label_csv $lr_scheduler $target_length $contrastive_head $joint_layers $num_workers $keep_register_tokens"
   done

   # Updated example command
   # bash egs/audioset/cluster_nodes/run_cavmae_ft_sync.sh 3e-3 48 multimodal 0,1 self_attention_cls 16 True 99999 10 finetuning_audioset 2626 True 0 16 datafilles/audioset_20k/cluster_nodes/audioset_20k_cleaned.json datafilles/audioset_20k/cluster_nodes/audioset_eval_cleaned_aug24.json datafilles/audioset_20k/cluster_nodes/class_labels_indices.csv cosine 512 False 16
