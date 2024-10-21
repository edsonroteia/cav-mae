import json
import os
from pathlib import Path

# Path to the JSON file
# json_file_path = 'datafilles/audioset_20k/cluster_nodes/audioset_20k_cleaned.json'
json_file_path = 'datafilles/audioset_2m/cluster_nodes/audioset_2m_cleaned_aug24.json'

# Path to the directory containing WAV files
wav_directory = '/local/1323407/datasets/audioset_2M_yuan/AS2M-audios/audio/'

# Initialize counters
existing_count = 0
missing_count = 0

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Iterate through each item in the data
for item in data['data']:
    video_id = item['video_id']
    wav_file = f"{video_id}.wav"
    wav_path = Path(wav_directory) / wav_file
    if wav_path.exists():
        existing_count += 1
    else:
        missing_count += 1

# Print results
total_count = existing_count + missing_count
print(f"Total video IDs processed: {total_count}")
print(f"WAV files found: {existing_count}")
print(f"WAV files missing: {missing_count}")
print(f"Percentage of existing WAV files: {(existing_count / total_count) * 100:.2f}%")