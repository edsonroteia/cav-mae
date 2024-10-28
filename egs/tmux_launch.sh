#!/bin/bash

# if help is called, print the usage with the default values
if [ "$1" = "-h" ]; then
  echo "Usage: $0 [aggregate] [freeze_base] [debug] [model_name] [cls_token] [dataset]"
  echo "Default values: aggregate=self_attention_cls, freeze_base=True, debug=False, model_name=model_2145, cls_token=False, dataset=audioset"
  exit 0
fi

# Create a new tmux window named 'job_run'
tmux new-window -n 'job_run'

# Split the window into 7 panes
tmux select-layout tiled
for i in {1..3}; do
  tmux split-window -h
done
tmux select-layout tiled

# Declare arrays for the parameter variations
lrs=(1e-4)
ftmodes=(multimodal audioonly videoonly)
cuda_devices=('0,1,2' '3,4,5' '6,7')  # Each run uses three GPUs
# Parse command line arguments
aggregate=${1:-self_attention_cls}
freeze_base=${2:-True}
debug=${3:-False}

# Set default values for num_samples and num_epochs
num_samples=9999999
num_epochs=25

# Override defaults if debug mode is enabled
if [ "$debug" = True ]; then
  num_samples=48
  num_epochs=1
fi

# Print the parsed arguments for verification
echo "Aggregate: $aggregate"
echo "Freeze base: $freeze_base"
echo "Debug mode: $debug"
echo "Number of samples: $num_samples"
echo "Number of epochs: $num_epochs"
# Command to run in each pane
dataset=${6:-audioset}
if [ "$dataset" = "audioset" ]; then
    cmd_prefix="bash egs/audioset/cluster_nodes/run_cavmae_ft_bal_sync.sh"
elif [ "$dataset" = "vggsound" ]; then
    cmd_prefix="bash egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh"
fi

neptune_tag1=aggr_${aggregate}_freeze_${freeze_base}
# get pretrain_path from models.csv
# Read the model name from command line argument
model_name=${4:-model_2145}  # Default to model_2145 if not provided

# Function to get the path for a given model name
get_model_path() {
    local model=$1
    awk -F ',' -v model="$model" '$1 == model {print $2}' models.csv
}

get_num_register_tokens() {
    local model=$1
    awk -F ',' -v model="$model" '$1 == model {print $3}' models.csv
}

get_total_frame() {
    local model=$1
    awk -F ',' -v model="$model" '$1 == model {print $4}' models.csv
}

# Get the pretrain_path
pretrain_path=$(get_model_path "$model_name")
cls_token=${5:-False}
if [ -z "$pretrain_path" ]; then
    echo "Error: Model $model_name not found in models.csv"
    exit 1
fi
# Get the number of register tokens
num_register_tokens=$(get_num_register_tokens "$model_name")
total_frame=$(get_total_frame "$model_name")

echo "Using model name: $model_name"
echo "Using pretrain_path: $pretrain_path"
echo "Using num_register_tokens: $num_register_tokens"
echo "Using total_frame: $total_frame"
# Arguments that remain constant across all runs
batch_size=48
num_workers=8

# Start running commands in each pane
pane=0

# Loop through feature modes (only one learning rate)
for ftmode in "${ftmodes[@]}"; do
    # Assign GPU based on pane number
    cuda_device=${cuda_devices[$pane]}
    
    # Print the process information
    echo "Launching process: lr=${lrs[0]}, ftmode=$ftmode on GPU(s) $cuda_device"
    # Print the exact command that will be run
    echo "Running command:"
    echo "$cmd_prefix ${lrs[0]} $batch_size $ftmode $cuda_device ${aggregate} $num_workers $freeze_base $num_samples $num_epochs $neptune_tag1 $pretrain_path $cls_token $num_register_tokens $total_frame"
    
    tmux send-keys -t $pane "echo 'Launching process: lr=${lrs[0]}, ftmode=$ftmode on GPU(s) $cuda_device' && dev_init && $cmd_prefix ${lrs[0]} $batch_size $ftmode $cuda_device ${aggregate} $num_workers $freeze_base $num_samples $num_epochs $neptune_tag1 $pretrain_path $cls_token $num_register_tokens $total_frame; echo 'Run completed with parameters: lr=${lrs[0]}, batch_size=$batch_size, ftmode=$ftmode, cuda_device=$cuda_device, aggregate=$aggregate, num_workers=$num_workers, freeze_base=$freeze_base, num_samples=$num_samples, num_epochs=$num_epochs, num_register_tokens=$num_register_tokens, total_frame=$total_frame'" C-m
    
    ((pane++))
done

# Add the 4th pane with the brocm-smi.sh command
tmux send-keys -t 3 "watch -n 1 bash ~/brocm-smi.sh" C-m

# Attach to the tmux session (optional)
tmux select-pane -t 0  # Move back to the first pane
tmux attach-session
