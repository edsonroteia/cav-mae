#!/bin/bash

# Function to run experiment and monitor output
run_and_monitor() {
    local lr=$1
    local batch_size=$2
    local ftmode=$3
    local exp_name=$4
    local timestamp=$5
    local base_command=$6
    local output_file="output_${exp_name}_${timestamp}.log"
    local pid_file="pid_${exp_name}_${timestamp}.txt"

    echo "Starting experiment: $exp_name (Learning Rate: $lr, Batch Size: $batch_size, FT Mode: $ftmode)"
    echo "Output file: $output_file"
    echo "Running command: $base_command $lr $batch_size $ftmode"

    # Run the experiment in the background and capture its PID
    $base_command $lr $batch_size $ftmode > "$output_file" 2>&1 &
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

    echo "$lr,$batch_size,$ftmode,$final_result" >> results.csv
    echo "Experiment $exp_name completed. Result: $final_result"

    return $crashed
}

# Function to print usage
print_usage() {
    echo "Usage: $0 <base_command> <batch_size> <ftmode> <learning_rate1> [learning_rate2] [learning_rate3] ..."
    echo "       If <ftmode> is 'all', it will run experiments for all ftmodes: multimodal, audioonly, videoonly"
    echo "Example: $0 './run_cavmae_ft_sync.sh' 48 multimodal 1e-5 5e-5 1e-4"
    echo "         $0 './run_cavmae_ft_sync.sh' 48 all 1e-5 5e-5 1e-4"
    exit 1
}

# Check if at least four arguments are provided (base command, batch size, ftmode, and at least one learning rate)
if [ $# -lt 4 ]; then
    print_usage
fi

# Extract the base command (first argument), batch size (second argument), and ftmode (third argument)
base_command="$1"
batch_size="$2"
ftmode="$3"
shift 3

# Use remaining command-line arguments as learning rates
learning_rates=("$@")

# Initialize results file
echo "Learning Rate,Batch Size,FT Mode,Final Result (multi-frame mAP)" > results.csv

# If ftmode is 'all', set up an array of all ftmodes
if [ "$ftmode" = "all" ]; then
    ftmodes=("multimodal" "audioonly" "videoonly")
else
    ftmodes=("$ftmode")
fi

# Calculate total number of experiments
total_experiments=$((${#learning_rates[@]} * ${#ftmodes[@]}))
echo "Total experiments to run: $total_experiments"
echo "Batch size for all experiments: $batch_size"

# Loop through each ftmode
for mode in "${ftmodes[@]}"
do
    # Loop through each learning rate
    for lr in "${learning_rates[@]}"
    do
        # Get current timestamp
        timestamp=$(date +"%Y%m%d_%H%M%S")
        # Construct the experiment name
        exp_name="experiment_lr${lr}_bs${batch_size}_${mode}"

        echo "------------------------------"
        echo "Running experiment: $exp_name"
        echo "Learning rate: $lr"
        echo "FT Mode: $mode"

        # Run the experiment with monitoring
        run_and_monitor $lr $batch_size $mode $exp_name $timestamp "$base_command"
        if [ $? -eq 1 ]; then
            echo "WARNING: Experiment $exp_name may have crashed or terminated unexpectedly."
        fi

        echo "Experiment $exp_name completed"
        echo "------------------------------"

        # Optional: add a delay between job submissions if needed
        sleep 5
    done
done

# Generate a formatted table of results
echo "Results Table:"
column -t -s ',' results.csv | sed 's/^/| /' | sed 's/$/ |/' | sed '2s/[^|]/-/g'