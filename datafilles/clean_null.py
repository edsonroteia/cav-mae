import json

def clean_json(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

    cleaned_data = {
        "data": [entry for entry in data['data'] if entry['labels'] is not None]
    }

    with open(output_path, 'w') as file:
        json.dump(cleaned_data, file, indent=4)

    # Print the number of samples in the original and cleaned data
    print(f"Number of samples in original data: {len(data['data'])}")
    print(f"Number of samples in cleaned data: {len(cleaned_data['data'])}")

# Example usage
input_path = 'as2m_web_mp4.json'
output_path = 'as2m_web_mp4_cleaned.json'
clean_json(input_path, output_path)