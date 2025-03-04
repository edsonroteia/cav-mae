#!/bin/bash

# if help is called, print the usage with the default values
if [ "$1" = "-h" ]; then
  echo "Usage: $0 [model_name] [cls_token] [n_register_tokens] [total_frame] [layer_idx]"
  echo "Default values: model_name=model_2145, cls_token=False, n_register_tokens=4, total_frame=16, layer_idx=0"
  exit 0
fi

# Parse command line arguments
model_name=${1:-model_2145}
cls_token=${2:-False}
n_register_tokens=${3:-4}
total_frame=${4:-16}
layer_idx=${5:-0}

# Get the pretrain_path from models.csv
pretrain_path=$(awk -F ',' -v model="$model_name" '$1 == model {print $2}' models.csv)
if [ -z "$pretrain_path" ]; then
  echo "Error: Model $model_name not found in models.csv"
  exit 1
fi

# Get the number of register tokens from models.csv if not provided
if [ "$n_register_tokens" = "4" ]; then
  n_register_tokens=$(awk -F ',' -v model="$model_name" '$1 == model {print $3}' models.csv)
  if [ -z "$n_register_tokens" ]; then
    n_register_tokens=4  # Default if not found in models.csv
  fi
fi

# Get the total_frame from models.csv if not provided
if [ "$total_frame" = "16" ]; then
  total_frame=$(awk -F ',' -v model="$model_name" '$1 == model {print $4}' models.csv)
  if [ -z "$total_frame" ]; then
    total_frame=16  # Default if not found in models.csv
  fi
fi

# Get target_length from models.csv
target_length=$(awk -F ',' -v model="$model_name" '$1 == model {print $6}' models.csv)
if [ -z "$target_length" ]; then
  target_length=1024  # Default if not found in models.csv
fi

# Print the parameters
echo "Using model name: $model_name"
echo "Using pretrain_path: $pretrain_path"
echo "Using cls_token: $cls_token"
echo "Using n_register_tokens: $n_register_tokens"
echo "Using total_frame: $total_frame"
echo "Using target_length: $target_length"
echo "Using layer_idx: $layer_idx"

# Set paths for data
data_val="datafilles/audioset_20k/cluster_nodes/audioset_eval_cleaned_aug24.json"
label_csv="datafilles/audioset_20k/cluster_nodes/class_labels_indices.csv"

# Create save directory
save_dir="attention_maps/${model_name}_layer${layer_idx}"
mkdir -p $save_dir

# Run the visualization script
python src/visualize_attention_maps.py \
  --pretrain_path $pretrain_path \
  --data_val $data_val \
  --label_csv $label_csv \
  --n_register_tokens $n_register_tokens \
  --cls_token $cls_token \
  --total_frame $total_frame \
  --batch_size 1 \
  --num_workers 4 \
  --num_samples 100 \
  --save_dir $save_dir \
  --layer_idx $layer_idx \
  --target_length $target_length

echo "Attention maps saved to $save_dir" 