import json
import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_json_file(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'data' not in data:
            logging.warning(f"'data' key not found in {json_file}")
            return 0, 0, 0
        
        original_count = len(data['data'])
        cleaned_data = []
        removed_count = 0
        
        for item in data['data']:
            wav_path = item.get('wav')
            if wav_path and os.path.exists(wav_path):
                cleaned_data.append(item)
            else:
                removed_count += 1
                if wav_path:
                    logging.debug(f"Removed entry with non-existent WAV path: {wav_path}")
                else:
                    logging.debug(f"Removed entry without WAV path")
        
        data['data'] = cleaned_data
        
        # Create a backup of the original file
        backup_file = json_file.with_suffix(json_file.suffix + '.bak')
        os.rename(json_file, backup_file)
        
        # Write the cleaned data back to the original filename
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return original_count, len(cleaned_data), removed_count
    
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file: {json_file}")
        return 0, 0, 0
    except Exception as e:
        logging.error(f"Error processing file {json_file}: {str(e)}")
        return 0, 0, 0

def process_json_files(folder_path):
    results = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                json_file = Path(root) / file
                logging.info(f"Processing file: {json_file}")
                original, cleaned, removed = clean_json_file(json_file)
                results.append({
                    'file': str(json_file),
                    'original': original,
                    'cleaned': cleaned,
                    'removed': removed
                })
    
    return results

def print_report(results):
    print("Report of JSON file cleaning:")
    print("-" * 80)
    for result in results:
        print(f"File: {result['file']}")
        print(f"  Original entries: {result['original']}")
        print(f"  Cleaned entries: {result['cleaned']}")
        print(f"  Removed entries: {result['removed']}")
        if result['original'] > 0:
            percentage = (result['removed'] / result['original']) * 100
            print(f"  Percentage removed: {percentage:.2f}%")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Clean JSON files by removing entries with missing WAV files.")
    parser.add_argument("folder", help="Path to the folder containing JSON files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    folder_path = Path(args.folder)

    if not folder_path.is_dir():
        logging.error(f"Error: '{folder_path}' is not a valid directory.")
        sys.exit(1)

    results = process_json_files(folder_path)
    
    if not results:
        logging.warning("No JSON files were processed. Check if the folder contains JSON files.")
    else:
        print_report(results)

if __name__ == "__main__":
    main()