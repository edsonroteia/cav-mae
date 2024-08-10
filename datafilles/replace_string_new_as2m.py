import json

# Path to your original JSON file
input_file_path = '/home/edson/code/cav-mae/datafilles/audioset_2m_cleaned.json'
# Path where the modified JSON file will be saved
output_file_path = '/home/edson/code/cav-mae/datafilles/audioset_2m_yuan_cleaned.json'

# Read the original JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Iterate over each item in the "data" list
for item in data['data']:
    # Replace the old audio path with the new one
    item['wav'] = item['wav'].replace(
        "/data/sls/audioset/dave_version/audio/",
        "/mnt/CVAI/data/edson/datasets/audioset-2M-yuan-preprocess/unbalanced/audio/"
    )
    item['wav'] = item['wav'].replace(".flac", ".wav")
    # Replace the old video path with the new one
    item['video_path'] = item['video_path'].replace(
        "/data/sls/audioset/dave_version/image_mulframe/",
        "/mnt/CVAI/data/edson/datasets/audioset-2M-yuan-preprocess/unbalanced/frames/"
    )

# Save the modified data to a new JSON file
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

print("File has been modified and saved.")