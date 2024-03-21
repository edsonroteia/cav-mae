import json
import csv

# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['data']

# Function to read CSV file and get class labels
def read_csv_file(file_path):
    with open(file_path, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header row
        class_labels = {rows[1]: rows[2] for rows in reader}
    return class_labels

# Function to select a balanced subset of the data
def select_balanced_subset(data, class_labels, max_percentage=0.05):
    label_count = {label: 0 for label in class_labels.keys()}
    selected_data = []
    total_samples = len(data)

    for sample in data:
        labels = sample['labels'].split(',')
        for label in labels:
            if label in label_count and (len(selected_data) < total_samples * max_percentage or label_count[label] < 3):
                selected_data.append(sample)
                label_count[label] += 1
                break

    return selected_data, label_count

# Function to write data to a JSON file
def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump({"data": data}, file, indent=4)

# Function to calculate and display statistics about a dataset
def display_dataset_statistics(dataset_name, dataset, label_count):
    total_samples = len(dataset)
    num_classes = len(label_count)
    avg_samples_per_class = total_samples / num_classes
    max_samples = max(label_count.values())
    min_samples = min(label_count.values())

    print(f"Total samples in {dataset_name}: {total_samples}")
    print(f"Average samples per class in {dataset_name}: {avg_samples_per_class:.2f}")
    print(f"Max samples in a single class in {dataset_name}: {max_samples}")
    print(f"Min samples in a single class in {dataset_name}: {min_samples}")

# Function to split the original data into two sets and save them
def split_and_save_data(original_data, selected_data, subset_file, remaining_file):
    remaining_data = [item for item in original_data if item not in selected_data]
    write_json_file(selected_data, subset_file)
    write_json_file(remaining_data, remaining_file)
    return remaining_data

# Main function to execute the script
def main():
    json_data = read_json_file('audioset_20k_cleaned.json')
    class_labels = read_csv_file('class_labels_indices.csv')
    validation_set, label_count_validation = select_balanced_subset(json_data, class_labels)
    training_set = split_and_save_data(json_data, validation_set, 'audioset_20k_cleaned_auto_val.json', 'audioset_20k_cleaned_auto_train.json')
    
    # Calculate label counts for the training set
    label_count_training = {label: 0 for label in class_labels.keys()}
    for sample in training_set:
        labels = sample['labels'].split(',')
        for label in labels:
            if label in label_count_training:
                label_count_training[label] += 1

    display_dataset_statistics("validation set", validation_set, label_count_validation)
    display_dataset_statistics("training set", training_set, label_count_training)

if __name__ == "__main__":
    main()
