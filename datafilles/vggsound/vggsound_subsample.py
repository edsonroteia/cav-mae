import json
import random
import os
from pathlib import Path

def create_subsampled_dataset(n_classes, input_path='datafilles/vggsound/cluster_nodes/vgg_train_cleaned.json', output_dir='datafilles/vggsound'):
    # Load the original dataset
    with open(input_path, 'r') as f:
        try:
            json_content = json.load(f)
            # Get the data array from the JSON
            original_data = json_content['data']
            print("First item in data:", original_data[0] if original_data else "Empty data")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print("First 100 characters of file:", f.read()[:100])
            return
    
    # Get unique classes (using 'labels' instead of 'label')
    all_classes = set(item['labels'] for item in original_data)
    
    # Randomly select n_classes
    selected_classes = random.sample(list(all_classes), n_classes)
    
    # Filter the dataset to only include selected classes
    subsampled_data = [
        item for item in original_data 
        if item['labels'] in selected_classes
    ]
    
    # Create output filename
    output_path = os.path.join(output_dir, f'vgg_train_{n_classes}.json')
    
    # Save the subsampled dataset
    with open(output_path, 'w') as f:
        # Maintain the same structure as input
        json.dump({"data": subsampled_data}, f, indent=2)
    
    print(f"Created subsampled dataset with {n_classes} classes at {output_path}")
    print(f"Total samples in subsampled dataset: {len(subsampled_data)}")

if __name__ == "__main__":
    # You can modify this number to get different class counts
    N_CLASSES = 30
    create_subsampled_dataset(N_CLASSES)
