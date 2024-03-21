#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_csv_file output_csv_file"
    exit 1
fi

# Assign command-line arguments to variables
input_csv=$1
output_csv=$2
audio_directory="/data1/edson/datasets/audioset-20k/balanced/audio"

# Clear the output file if it already exists
> "$output_csv"

# Read each line from the CSV
while IFS= read -r line; do
    # Extract the ID from the filename (e.g., 91AmOKytRUM from /path/to/91AmOKytRUM.mkv)
    id=$(basename "$line" .mkv)

    # Check if a file with this ID already exists in the target directory
    if [ ! -f "$audio_directory/$id.wav" ]; then
        # If the file does not exist, write the line to the output CSV
        echo "$line" >> "$output_csv"
    fi
done < "$input_csv"

echo "Filtering complete. Output saved to $output_csv"
