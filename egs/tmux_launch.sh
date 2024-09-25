#!/bin/bash

# Create a new tmux window named 'job_run'
tmux new-window -n 'job_run'

# Split the window into 9 panes
tmux select-layout tiled
for i in {1..8}; do
  tmux split-window -h
done
tmux select-layout tiled

# Declare arrays for the parameter variations
lrs=(1e-2 1e-3 1e-4)
ftmodes=(multimodal audioonly videoonly)
cuda_devices=(0 1 2 3 4 5 6 7)  # Each run uses one GPU
aggregate=${1:-self_attention_cls}
freeze_base=${2:-True}
debug=${3:-False}
if [ "$debug" = True ]; then
  num_samples=48
  num_epochs=1
else
  num_samples=9999999
  num_epochs=25
fi
# Command to run in each pane
cmd_prefix="bash egs/audioset/cluster_nodes/run_cavmae_ft_bal_sync.sh"

# Arguments that remain constant across all runs
batch_size=48
num_workers=8

# Start running commands in each pane
pane=0
last_command=""

# Loop through learning rates and feature modes
for lr in "${lrs[@]}"; do
  for ftmode in "${ftmodes[@]}"; do
    if [[ $pane -lt 7 ]]; then
      # Assign one job per GPU
      tmux send-keys -t $pane "dev_init && $cmd_prefix $lr $batch_size $ftmode ${cuda_devices[$pane]} ${aggregate} $num_workers $freeze_base $num_samples $num_epochs; echo 'Run completed with parameters: lr=$lr, batch_size=$batch_size, ftmode=$ftmode, cuda_device=${cuda_devices[$pane]}, aggregate=$aggregate, num_workers=$num_workers, freeze_base=$freeze_base, num_samples=$num_samples, num_epochs=$num_epochs'" C-m
    else
      # Store the remaining command for the last GPU
      last_command="$cmd_prefix $lr $batch_size $ftmode ${cuda_devices[7]} ${aggregate} $num_workers $freeze_base $num_samples; echo 'Run completed with parameters: lr=$lr, batch_size=$batch_size, ftmode=$ftmode, cuda_device=${cuda_devices[7]}, aggregate=$aggregate, num_workers=$num_workers, freeze_base=$freeze_base, num_samples=$num_samples'"
    fi
    ((pane++))
  done
done

# For the last pane (pane 7), run two jobs on the same GPU (using CUDA device 7)
if [[ -n "$last_command" ]]; then
  # Dynamically use the last elements from the arrays
  last_lr=${lrs[-1]}           # Last element of the lrs array
  last_ftmode=${ftmodes[-1]}   # Last element of the ftmodes array

  tmux send-keys -t 7 "dev_init && $last_command; $cmd_prefix $last_lr $batch_size $last_ftmode ${cuda_devices[7]} ${aggregate} $num_workers $freeze_base $num_samples; echo 'Run completed with parameters: lr=$last_lr, batch_size=$batch_size, ftmode=$last_ftmode, cuda_device=${cuda_devices[7]}, aggregate=$aggregate, num_workers=$num_workers, freeze_base=$freeze_base, num_samples=$num_samples'" C-m
fi

# Add the 9th pane with the brocm-smi.sh command
tmux send-keys -t 8 "watch -n 1 bash ~/brocm-smi.sh" C-m

# Attach to the tmux session (optional)
tmux select-pane -t 0  # Move back to the first pane
tmux attach-session