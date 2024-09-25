import json
import os
import sys
import argparse
import logging
from pathlib import Path
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_json_file(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'data' not in data:
            logging.warning(f"'data' key not found in {json_file}")
            return 0, 0, 0
        
        original_data = data['data']
        original_count = len(original_data)
        cleaned_data = []
        removed_indices = []
        
        for index, item in enumerate(original_data):
            wav_path = item.get('wav')
            if wav_path and os.path.exists(wav_path):
                cleaned_data.append(item)
            else:
                removed_indices.append(index)
                if wav_path:
                    logging.debug(f"Removed entry with non-existent WAV path: {wav_path}")
                else:
                    logging.debug(f"Removed entry without WAV path")
        
        data['data'] = cleaned_data
        removed_count = len(removed_indices)
        
        csv_file = json_file.with_name(json_file.stem + '_weight.csv')
        if csv_file.exists():
            with open(csv_file, 'r') as f:
                csv_data = list(csv.reader(f))
            
            # Remove CSV entries corresponding to removed JSON entries
            cleaned_csv_data = [csv_data[0]]  # Keep the header
            for i, row in enumerate(csv_data[1:], 1):
                if i - 1 not in removed_indices:
                    cleaned_csv_data.append(row)
            
            # Create a backup of the original CSV file
            csv_backup_file = csv_file.with_suffix(csv_file.suffix + '.bak')
            os.rename(csv_file, csv_backup_file)
            
            # Write the cleaned CSV data back to the original filename
            with open(csv_file, 'w', newline='') as f:
                csv.writer(f).writerows(cleaned_csv_data)
            
            logging.info(f"Updated corresponding CSV file: {csv_file}")

        # Create a backup of the original JSON file
        backup_file = json_file.with_suffix(json_file.suffix + '.bak')
        os.rename(json_file, backup_file)
        
        # Write the cleaned data back to the original JSON filename
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
    parser.add_argument("folders", nargs='+', help="Paths to folders containing JSON files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    all_results = []
    for folder in args.folders:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            logging.error(f"Error: '{folder_path}' is not a valid directory.")
            continue

        results = process_json_files(folder_path)
        all_results.extend(results)
    
    if not all_results:
        logging.warning("No JSON files were processed. Check if the folders contain JSON files.")
    else:
        print_report(all_results)

if __name__ == "__main__":
    main()