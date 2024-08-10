import json

def process_file(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

    for entry in data['data']:
        # Modify the video_path
        video_path = entry['video_path'].replace('.mkv', '.mp4')
        entry['video_path'] = f"/mnt/CVAI/data/edson/datasets/audioset-2M-mp4/{video_path}"
        
        # Remove the tar_file key
        if 'tar_file' in entry:
            del entry['tar_file']

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
input_path = 'audioset_eval_cleaned.json'
output_path = 'audioset_eval_cleaned_mp4.json'
process_file(input_path, output_path)