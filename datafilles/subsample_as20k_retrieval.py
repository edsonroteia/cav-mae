import json
from collections import defaultdict

def load_data(filename):
    """ Load data from a JSON file """
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['data']

def save_data(data, filename):
    """ Save data to a JSON file """
    with open(filename, 'w') as file:
        json.dump({"data": data}, file, indent=4)

def select_samples(data):
    """ Select 5 samples per class from the dataset """
    class_count = defaultdict(int)
    selected_data = []

    for entry in data:
        # Multiple labels per entry are split and processed individually
        labels = entry['labels'].split(',')
        for label in labels:
            if class_count[label] < 5:
                selected_data.append(entry)
                class_count[label] += 1
                break  # Only add the entry once even if it has multiple qualifying labels

    return selected_data, class_count

def print_statistics(class_count):
    """ Print statistics of the number of entries selected per class """
    total_classes = len(class_count)
    print(f"Total classes: {total_classes}")
    for label, count in class_count.items():
        print(f"Class {label}: {count} entries")

def main():
    input_filename = 'audioset_20k_cleaned_auto_val.json'  # Specify the path to your input file
    output_filename = 'audioset_20k_cleaned_auto_val_retrieval.json'  # Specify the path to your output file

    data = load_data(input_filename)
    selected_data, class_count = select_samples(data)
    save_data(selected_data, output_filename)
    print_statistics(class_count)

if __name__ == "__main__":
    main()
