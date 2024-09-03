#!/bin/bash

# Check if at least one learning rate is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <learning_rate1> [learning_rate2] [learning_rate3] ..."
    echo "Example: $0 1e-5 5e-5 1e-4"
    exit 1
fi

# Use all command-line arguments as learning rates
learning_rates=("$@")

# Base command (use the updated script name)
base_command="./run_cavmae_ft_bal_cluster_test_240822_check.sh"

# Loop through each learning rate
for lr in "${learning_rates[@]}"
do
    # Get current timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Construct the experiment name
    exp_name="cavmae_ft_audioset_bal_lr${lr}"
    
    # Run the experiment with the current learning rate
    LR=$lr EXP_NAME=$exp_name TIMESTAMP=$timestamp $base_command
    
    # Optional: add a delay between job submissions if needed
    sleep 5
done