import json
import random
import copy

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def split_dataset(data, split_ratio=0.99):
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    return data[:split_index], data[split_index:]

def create_id_to_labels_map(old_data):
    return {item['video_id']: item['labels'] for item in old_data}

def copy_labels(new_data, id_to_labels_map):
    for item in new_data:
        if item['video_id'] in id_to_labels_map:
            item['labels'] = id_to_labels_map[item['video_id']]
    return new_data

def main():
    # Load the data
    new_json_path = 'datafilles/as2m_web_nolabels.json'
    old_json_path = 'datafilles/audioset_2m_cleaned.json'
    
    new_data = load_json(new_json_path)['data']
    old_data = load_json(old_json_path)['data']

    # Create a map of video_id to labels from the old data
    id_to_labels_map = create_id_to_labels_map(old_data)

    # Split the new data
    train_data, val_data = split_dataset(new_data)

    # Copy labels to the validation set
    val_data = copy_labels(val_data, id_to_labels_map)

    # Prepare the output data
    train_output = {'data': train_data}
    val_output = {'data': val_data}

    # Save the new JSON files
    save_json(train_output, 'datafilles/as2m_web_nolabels_train.json')
    save_json(val_output, 'datafilles/as2m_web_labels_val.json')

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print("Files 'datafilles/as2m_web_nolabels_train.json' and 'datafilles/as2m_web_labels_val.json' have been created.")

if __name__ == "__main__":
    main()