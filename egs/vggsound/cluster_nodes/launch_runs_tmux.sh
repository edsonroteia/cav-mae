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
lrs=(5e-2 1e-2 1e-3)
ftmodes=(multimodal audioonly videoonly)
cuda_devices=(0 1 2 3 4 5 6 7)  # Each run uses one GPU

# Command to run in each pane
cmd_prefix="egs/vggsound/cluster_nodes/run_cavmae_ft_sync.sh"

# Arguments that remain constant across all runs
batch_size=48
num_workers=8

# Start running commands in each pane
pane=0
for lr in "${lrs[@]}"; do
  for ftmode in "${ftmodes[@]}"; do
    if [[ $pane -lt 7 ]]; then
      # Assign one job per GPU
      tmux send-keys -t $pane "dev_init && $cmd_prefix $lr $batch_size $ftmode ${cuda_devices[$pane]} None $num_workers" C-m
    else
      # For the last pane, run two jobs on the same GPU (using CUDA device 7)
      tmux send-keys -t $pane "dev_init && $cmd_prefix $lr $batch_size $ftmode ${cuda_devices[7]} None $num_workers; $cmd_prefix $lr $batch_size multimodal ${cuda_devices[7]} self_attention_cls $num_workers" C-m
    fi
    ((pane++))
  done
done

# Attach to the tmux session (optional)
tmux select-pane -t 0  # Move back to the first pane
tmux attach-session