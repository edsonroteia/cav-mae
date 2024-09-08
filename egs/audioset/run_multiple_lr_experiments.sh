#!/bin/bash

# Function to run experiment and monitor output
run_and_monitor() {
    local lr=$1
    local batch_size=$2
    local exp_name=$3
    local timestamp=$4
    local base_command=$5
    local output_file="output_${exp_name}_${timestamp}.log"
    local pid_file="pid_${exp_name}_${timestamp}.txt"

    echo "Starting experiment: $exp_name (Learning Rate: $lr, Batch Size: $batch_size)"
    echo "Output file: $output_file"

    # Run the experiment in the background and capture its PID
    $base_command $lr $batch_size > "$output_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"

    # Monitor the process
    local start_time=$(date +%s)
    local last_output_time=$start_time
    local crashed=false

    while true; do
        if ! ps -p $pid > /dev/null; then
            echo "Process for $exp_name has ended."
            # Check if the process ended too quickly (potential crash)
            if [ $(($(date +%s) - start_time)) -lt 60 ]; then
                echo "WARNING: Process ended very quickly. It may have crashed."
                crashed=true
            fi
            break
        fi

        # Check if there's been any recent output
        if [ -f "$output_file" ]; then
            local last_modified=$(stat -c %Y "$output_file")
            if [ $((last_modified - last_output_time)) -gt 3600 ]; then
                echo "No output for more than an hour. Terminating $exp_name..."
                kill -9 $pid
                crashed=true
                break
            fi
            last_output_time=$last_modified
        fi

        sleep 60  # Check every minute
    done

    # Extract the final result (multi-frame mAP)
    local final_result="N/A"
    if [ "$crashed" = false ]; then
        final_result=$(grep "multi-frame mAP is" "$output_file" | tail -n 1 | awk '{print $4}')
        if [ -z "$final_result" ]; then
            final_result="N/A"
            echo "Warning: Could not extract multi-frame mAP for $exp_name"
        fi
    else
        echo "Experiment $exp_name crashed or was terminated due to inactivity."
    fi

    echo "$lr,$batch_size,$final_result" >> results.csv
    echo "Experiment $exp_name completed. Result: $final_result"

    return $crashed
}

# Function to print usage
print_usage() {
    echo "Usage: $0 <base_command> <batch_size> <learning_rate1> [learning_rate2] [learning_rate3] ..."
    echo "Example: $0 './run_cavmae_ft_sync.sh' 24 1e-5 5e-5 1e-4"
    exit 1
}

# Check if at least three arguments are provided (base command, batch size, and at least one learning rate)
if [ $# -lt 3 ]; then
    print_usage
fi

# Extract the base command (first argument) and batch size (second argument)
base_command="$1"
batch_size="$2"
shift 2

# Use remaining command-line arguments as learning rates
learning_rates=("$@")

# Initialize results file
echo "Learning Rate,Batch Size,Final Result (multi-frame mAP)" > results.csv

# Display total number of experiments
total_experiments=${#learning_rates[@]}
echo "Total experiments to run: $total_experiments"
echo "Batch size for all experiments: $batch_size"

# Loop through each learning rate
for index in "${!learning_rates[@]}"
do
    lr=${learning_rates[$index]}
    # Get current timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    # Construct the experiment name
    exp_name="experiment_lr${lr}_bs${batch_size}"

    echo "------------------------------"
    echo "Running experiment $((index+1)) of $total_experiments"
    echo "Learning rate: $lr"

    # Run the experiment with monitoring
    run_and_monitor $lr $batch_size $exp_name $timestamp "$base_command"
    if [ $? -eq 1 ]; then
        echo "WARNING: Experiment $exp_name may have crashed or terminated unexpectedly."
    fi

    echo "Experiment $((index+1)) of $total_experiments completed"
    echo "------------------------------"

    # Optional: add a delay between job submissions if needed
    sleep 5
done

# Generate a formatted table of results
echo "Results Table:"
column -t -s ',' results.csv | sed 's/^/| /' | sed 's/$/ |/' | sed '2s/[^|]/-/g'