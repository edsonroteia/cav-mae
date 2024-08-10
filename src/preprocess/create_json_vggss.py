import os
import json
import csv

# Define the paths to the input data and files
audio_dir = '/data1/edson/datasets/VGGSS/VGGSS_audios'
video_dir = '/data1/edson/datasets/VGGSS/VGGSS_frames'
label_csv = '/data1/edson/datasets/VGGSS/class_labels_indices_vgg.csv'
output_json = '/data1/edson/datasets/VGGSS/sample_data.json'

# Load label information from the CSV file
label_map = {}
with open(label_csv, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        label_map[row['display_name']] = row['mid']

# Create a list to store the sample data
samples = []

# Iterate over the entries in vggss.json
with open('/data1/edson/datasets/VGGSS/vggss.json', 'r') as json_file:
    vggss_data = json.load(json_file)
    for entry in vggss_data:
        file_name = entry['file']
        class_label = entry['class']
        bbox = entry['bbox']

        # Construct the paths for audio and video files
        audio_path = os.path.join(audio_dir, file_name + '.wav')
        video_id = os.path.splitext(file_name)[0]

        # Map the class label to the corresponding display name
        label = label_map.get(class_label.replace(",", "_"))

        # Create the sample dictionary
        sample = {
            'wav': audio_path,
            'video_id': video_id,
            'video_path': video_dir,
            'labels': label,
            'bbox': bbox
        }

        # Append the sample to the list
        samples.append(sample)

data = {'data': samples}

# Write the sample data to the JSON file
with open(output_json, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("JSON file created successfully!")
