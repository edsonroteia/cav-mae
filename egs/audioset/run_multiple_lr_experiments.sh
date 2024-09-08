#!/bin/bash

# Function to run experiment and monitor output
run_and_monitor() {
    local lr=$1
    local exp_name=$2
    local timestamp=$3
    local base_command=$4
    local output_file="output_${exp_name}_${timestamp}.log"
    local pid_file="pid_${exp_name}_${timestamp}.txt"

    echo "Starting experiment: $exp_name (Learning Rate: $lr)"
    echo "Output file: $output_file"

    # Run the experiment in the background and capture its PID
    LR=$lr EXP_NAME=$exp_name TIMESTAMP=$timestamp $base_command > "$output_file" 2>&1 &
    echo $! > "$pid_file"

    # Monitor the output file
    while true; do
        if ! ps -p $(cat "$pid_file") > /dev/null; then
            echo "Process for $exp_name has finished."
            break
        fi

        if [[ $(find "$output_file" -mmin +60) ]]; then
            echo "No output for more than an hour. Restarting $exp_name..."
            kill -9 $(cat "$pid_file")
            LR=$lr EXP_NAME=$exp_name TIMESTAMP=$timestamp $base_command > "$output_file" 2>&1 &
            echo $! > "$pid_file"
        fi

        sleep 300  # Check every 5 minutes
    done

    # Extract the final result (multi-frame mAP)
    local final_result=$(grep "multi-frame mAP is" "$output_file" | tail -n 1 | awk '{print $4}')
    if [ -z "$final_result" ]; then
        final_result="N/A"
        echo "Warning: Could not extract multi-frame mAP for $exp_name"
    fi
    echo "$lr,$final_result" >> results.csv
    echo "Experiment $exp_name completed. Result: $final_result"
}

# Function to print usage
print_usage() {
    echo "Usage: $0 <base_command> <learning_rate1> [learning_rate2] [learning_rate3] ..."
    echo "Example: $0 'bash egs/audioset/run_cavmae_ft_bal_cluster_test_240822_check.sh' 1e-5 5e-5 1e-4"
    exit 1
}

# Check if at least two arguments are provided (base command and at least one learning rate)
if [ $# -lt 2 ]; then
    print_usage
fi

# Extract the base command (first argument)
base_command="$1"
shift

# Use remaining command-line arguments as learning rates
learning_rates=("$@")

# Initialize results file
echo "Learning Rate,Final Result (multi-frame mAP)" > results.csv

# Display total number of experiments
total_experiments=${#learning_rates[@]}
echo "Total experiments to run: $total_experiments"

# Loop through each learning rate
for index in "${!learning_rates[@]}"
do
    lr=${learning_rates[$index]}
    # Get current timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    # Construct the experiment name
    exp_name="experiment_lr${lr}"

    echo "------------------------------"
    echo "Running experiment $((index+1)) of $total_experiments"
    echo "Learning rate: $lr"

    # Run the experiment with monitoring
    run_and_monitor $lr $exp_name $timestamp "$base_command"

    echo "Experiment $((index+1)) of $total_experiments completed"
    echo "------------------------------"

    # Optional: add a delay between job submissions if needed
    sleep 5
done

# Generate a formatted table of results
echo "Results Table:"
column -t -s ',' results.csv | sed 's/^/| /' | sed 's/$/ |/' | sed '2s/[^|]/-/g'