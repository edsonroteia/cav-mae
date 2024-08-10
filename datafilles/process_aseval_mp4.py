import json

def transform_file(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

    for entry in data['data']:
        # Modify the video_path
        video_id = entry['video_id']
        entry['video_path'] = f"/mnt/CVAI/data/edson/datasets/audioset-2M-mp4/{video_id}.mp4"
        
        # Remove the wav key
        if 'wav' in entry:
            del entry['wav']

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
input_path = 'audioset_eval_cleaned.json'
output_path = 'audioset_eval_cleaned_mp4.json'
transform_file(input_path, output_path)