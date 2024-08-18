import json
import random
import csv
import sys

# Load JSON data from file
with open('/home/edson/code/cav-mae/datafilles/audioset_eval_cleaned_original.json', 'r') as file:
    data = json.load(file)['data']

# Sample n random videos
no_videos = int(sys.argv[1])
no_videos = min(no_videos, len(data))
sampled_videos = random.sample(data, no_videos)

# Create CSV file and write the formatted lines
with open(f'sample_videos_{no_videos}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for video in sampled_videos:
        formatted_line = f"/mnt/CVAI/data/edson/datasets/audioset-2M/unbalanced/unbalanced/{video['video_id']}.mkv"
        writer.writerow([formatted_line])

print(f"CSV file has been created with {no_videos} sampled videos.")