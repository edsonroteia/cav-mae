import json
import random
import os
from pathlib import Path
from collections import Counter
import numpy as np
import argparse
def create_subsampled_dataset(n_classes, train_path='datafilles/vggsound/cluster_nodes/vgg_train_cleaned.json', 
                            eval_path='datafilles/vggsound/cluster_nodes/vgg_test_cleaned.json', 
                            output_dir='datafilles/vggsound/cluster_nodes/',
                            random_seed=42):
    # Set all seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    
    # Load the training dataset
    with open(train_path, 'r') as f:
        try:
            train_json = json.load(f)
            train_data = train_json['data']
        except json.JSONDecodeError as e:
            print(f"Error decoding training JSON: {e}")
            return

    # Load the eval dataset
    with open(eval_path, 'r') as f:
        try:
            eval_json = json.load(f)
            eval_data = eval_json['data']
        except json.JSONDecodeError as e:
            print(f"Error decoding eval JSON: {e}")
            return
    
    # Get unique classes from training data
    all_classes = set(item['labels'] for item in train_data)
    
    # Randomly select n_classes
    selected_classes = random.sample(list(all_classes), n_classes)
    
    # Filter both datasets to only include selected classes
    subsampled_train = [item for item in train_data if item['labels'] in selected_classes]
    subsampled_eval = [item for item in eval_data if item['labels'] in selected_classes]
    
    # Create output filenames
    train_output = os.path.join(output_dir, f'vgg_train_{n_classes}.json')
    eval_output = os.path.join(output_dir, f'vgg_test_{n_classes}.json')
    train_weight_output = os.path.join(output_dir, f'vgg_train_{n_classes}_weight.csv')
    
    # Save the subsampled datasets
    with open(train_output, 'w') as f:
        json.dump({"data": subsampled_train}, f, indent=2)
    
    with open(eval_output, 'w') as f:
        json.dump({"data": subsampled_eval}, f, indent=2)
    
    # Generate and save weights for training dataset only
    class_counts = Counter(item['labels'] for item in subsampled_train)
    
    # Calculate initial weights for each class
    class_weights = {
        label: 1.0 / count 
        for label, count in class_counts.items()
    }
    
    # Normalize weights so minimum is 1
    min_weight = min(class_weights.values())
    class_weights = {
        label: max(weight / min_weight, 1.0)
        for label, weight in class_weights.items()
    }
    
    # Generate weights in same order as JSON file
    weights = [class_weights[item['labels']] for item in subsampled_train]
    
    # Save weights
    np.savetxt(train_weight_output, weights, fmt='%.18e')
    
    # Create filtered class labels file
    labels_input = 'datafilles/vggsound/cluster_nodes/class_labels_indices_vgg.csv'
    labels_output = os.path.join(output_dir, f'class_labels_indices_vgg_{n_classes}.csv')
    
    # Read original labels file and filter for selected classes
    with open(labels_input, 'r') as f:
        lines = f.readlines()
        header = lines[0]  # Save header
        # Filter lines and store original index and rest of line
        filtered_lines = [(int(line.split(',')[0]), ','.join(line.split(',')[1:])) 
                         for line in lines[1:] 
                         if f"vgg_{line.split(',')[0]}" in selected_classes]
    
    # Write filtered labels file with new indices
    with open(labels_output, 'w') as f:
        f.write(header)  # Write header
        for new_idx, (_, rest_of_line) in enumerate(filtered_lines):
            f.write(f"{new_idx},{rest_of_line}")  # Write reindexed lines
    
    print(f"Created subsampled datasets with {n_classes} classes:")
    print(f"Training: {train_output} ({len(subsampled_train)} samples)")
    print(f"Training weights: {train_weight_output}")
    print(f"Eval: {eval_output} ({len(subsampled_eval)} samples)")
    print(f"Class labels: {labels_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subsampled VGGSound dataset")
    parser.add_argument("--num_classes", type=int, default=30, help="Number of classes to subsample")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # You can modify these numbers to get different class counts
    N_CLASSES = args.num_classes
    RANDOM_SEED = args.random_seed
    create_subsampled_dataset(N_CLASSES, random_seed=RANDOM_SEED)
