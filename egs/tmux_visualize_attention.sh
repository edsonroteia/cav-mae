#!/bin/bash

# if help is called, print the usage with the default values
if [ "$1" = "-h" ]; then
  echo "Usage: $0 [model_name] [cls_token] [n_register_tokens] [total_frame]"
  echo "Default values: model_name=model_2145, cls_token=False, n_register_tokens=4, total_frame=16"
  exit 0
fi

# Parse command line arguments
model_name=${1:-model_2145}
cls_token=${2:-False}
n_register_tokens=${3:-4}
total_frame=${4:-16}

# Create a new tmux window named 'attention_vis'
tmux new-window -n 'attention_vis'

# Split the window into 4 panes
tmux select-layout tiled
for i in {1..3}; do
  tmux split-window -h
done
tmux select-layout tiled

# Define the layers to visualize
layers=(0 1 2 3)
cuda_devices=('0' '1' '2' '3')  # Each run uses one GPU

# Start running commands in each pane
for pane in {0..3}; do
    layer_idx=${layers[$pane]}
    cuda_device=${cuda_devices[$pane]}
    
    # Print the process information
    echo "Launching visualization for layer $layer_idx on GPU $cuda_device"
    
    # Run the command in the corresponding pane
    tmux send-keys -t $pane "echo 'Visualizing attention maps for layer $layer_idx on GPU $cuda_device' && CUDA_VISIBLE_DEVICES=$cuda_device bash egs/visualize_attention.sh $model_name $cls_token $n_register_tokens $total_frame $layer_idx" C-m
done

# Attach to the tmux session
tmux select-pane -t 0  # Move back to the first pane
tmux attach-session 