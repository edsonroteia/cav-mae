import json

def merge_labels(source_path, target_path, output_path):
    # Read source data
    with open(source_path, 'r') as source_file:
        source_data = json.load(source_file)

    # Read target data
    with open(target_path, 'r') as target_file:
        target_data = json.load(target_file)

    # Create a map of video_id to labels from the source data
    label_map = {entry['video_id']: entry['labels'] for entry in source_data['data']}

    # Print the number of samples in the source data
    print(f"Number of samples in source (audioset_2m_cleaned.json): {len(source_data['data'])}")

    # Print the number of samples in the target data
    print(f"Number of samples in target (as2m_web_nolabels_mp4.json): {len(target_data['data'])}")

    # Add labels to the target data
    for entry in target_data['data']:
        video_id = entry['video_id']
        if video_id in label_map:
            entry['labels'] = label_map[video_id]

    # Print the number of samples in the output data
    print(f"Number of samples in output (merged_output.json): {len(target_data['data'])}")

    # Write the merged data to the output file
    with open(output_path, 'w') as output_file:
        json.dump(target_data, output_file, indent=4)

# Example usage
source_path = 'audioset_2m_cleaned.json'
target_path = 'as2m_web_nolabels_mp4.json'
output_path = 'as2m_web_mp4.json'
merge_labels(source_path, target_path, output_path)