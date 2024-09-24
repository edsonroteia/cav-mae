#!/bin/bash

# Create a new tmux window named 'job_run'
tmux new-window -n 'job_run'

# Split the window into 8 panes
tmux select-layout tiled  # Ensures panes are tiled (you can adjust layout as per your preference)
for i in {1..7}; do
  tmux split-window -h
done
tmux select-layout tiled  # Make sure panes are arranged correctly

# Declare arrays for the parameter variations
lrs=(1e-2 1e-3 1e-4)
ftmodes=(multimodal audioonly videoonly)
cuda_devices=(0 1 2 3 4 5 6 7)  # Each run uses one GPU

# Command to run in each pane
cmd_prefix="bash egs/audioset/cluster_nodes/run_cavmae_ft_bal_sync.sh"

# Arguments that remain constant across all runs
batch_size=48
num_workers=8

# Start running commands in each pane
pane=0
last_command=""

for lr in "${lrs[@]}"; do
  for ftmode in "${ftmodes[@]}"; do
    if [[ $pane -lt 7 ]]; then
      # Assign one job per GPU
      tmux send-keys -t $pane "dev_init && $cmd_prefix $lr $batch_size $ftmode ${cuda_devices[$pane]} None $num_workers" C-m
    else
      # Store the remaining command for the last GPU
      last_command="$cmd_prefix $lr $batch_size $ftmode ${cuda_devices[7]} None $num_workers"
    fi
    ((pane++))
  done
done

# For the last pane (pane 7), run two jobs on the same GPU (using CUDA device 7)
if [[ -n "$last_command" ]]; then
  tmux send-keys -t 7 "dev_init && $last_command; $cmd_prefix 1e-4 $batch_size multimodal ${cuda_devices[7]} None $num_workers" C-m
fi

# Attach to the tmux session (optional)
tmux select-pane -t 0  # Move back to the first pane
tmux attach-session