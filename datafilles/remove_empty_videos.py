import json
import concurrent.futures
from tqdm import tqdm

def read_video_ids_from_txt(file_path):
    """ Reads video IDs from a text file, skipping the first two lines. """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Skip the first two lines and extract the rest
        video_ids = [line.strip() for line in lines[2:]]
    return video_ids

def filter_entry(entry, video_ids):
    """ Check if the entry should be included in the filtered data. """
    if entry['video_id'] not in video_ids:
        return entry
    return None

def filter_json_data(json_path, video_ids):
    """ Filters out entries from the JSON file that are present in the video_ids list using multiprocessing. """
    with open(json_path, 'r') as file:
        data = json.load(file)
        data_list = data['data']

    # Set up the ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        # Use a chunksize to optimize performance
        chunksize = int(len(data_list) / (48 * 5))
        results = list(tqdm(executor.map(filter_entry, data_list, [video_ids] * len(data_list), chunksize=chunksize), total=len(data_list), desc="Filtering entries"))

    # Remove None entries
    filtered_data = [item for item in results if item is not None]
    return filtered_data

def save_filtered_data(filtered_data, output_path):
    """ Saves the filtered data into a new JSON file. """
    with open(output_path, 'w') as file:
        json.dump({"data": filtered_data}, file, indent=4)

# Define file paths
txt_file_path = '/home/edson/code/random/test.txt'
json_file_path = '/home/edson/code/cav-mae/datafilles/as2m_web_mp4_cleaned.json'
output_json_path = '/home/edson/code/cav-mae/datafilles/as2m_web_mp4_cleaned_cleaned.json'

# Perform the filtering
video_ids = read_video_ids_from_txt(txt_file_path)
filtered_data = filter_json_data(json_file_path, video_ids)
save_filtered_data(filtered_data, output_json_path)

print("Filtering complete. The filtered JSON has been saved to", output_json_path)