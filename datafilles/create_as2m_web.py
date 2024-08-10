import os
import tarfile
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import subprocess

def list_tar_contents(tar_file):
    """
    Uses 'tar -tvf' to list the contents of a tar file.
    """
    result = subprocess.run(["tar", "-tvf", tar_file], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Error listing tar contents: {result.stderr}")

    # Parse the output to get file names
    contents = result.stdout.strip().split("\n")
    filenames = [line.split()[-1] for line in contents]  # Extract the last field as filename

    return filenames

def process_tar_file(tar_file):
    """
    Process a single tar file, extracting metadata for each video entry.
    """
    data_entries = []

    # List all file names using the command-line utility
    filenames = list_tar_contents(tar_file)

    # Filter for `.mkv` files
    mkv_files = [filename for filename in filenames if filename.endswith(".mkv")]

    for mkv_file in mkv_files:
        video_id = mkv_file.split('/')[-1].split('.')[0]  # Extract the video ID
        labels = None  # Adjust if a mapping mechanism for labels exists

        # Collect metadata entry
        data_entries.append({
            "video_id": video_id,
            "video_path": f"{mkv_file}",  # Path within the tar
            "labels": labels,  # Or set manually
            "tar_file": str(tar_file)  # Convert to string for serialization
        })

    # Print the first entry of this tar file (if available) for quick verification
    if data_entries:
        print(f"Processed from {tar_file}: {data_entries[0]}")

    return data_entries

def collect_metadata(tar_dir):
    # Gather all tar files in the directory
    tar_files = list(Path(tar_dir).glob("*.tar"))

    # Create a pool with the desired number of workers (up to 48 cores in this case)
    num_workers = min(48, cpu_count())

    with Pool(num_workers) as pool:
        # Process tar files in parallel
        results = list(tqdm(pool.imap(process_tar_file, tar_files), total=len(tar_files), desc="Processing tar files"))

    # Flatten the list of lists into a single list of entries
    data_entries = [entry for sublist in results for entry in sublist]

    return data_entries

def save_to_json(data_entries, output_path):
    output_data = {"data": data_entries}

    # Write to the specified output file
    with open(output_path, "w") as outfile:
        json.dump(output_data, outfile, indent=4)

if __name__ == "__main__":
    tar_dir = "/mnt/CVAI/data/edson/datasets/audioset-2M-tar/unbalanced"  # Set your directory path here
    output_file = "as2m_web_nolabels.json"

    # Collect metadata with parallel processing
    data_entries = collect_metadata(tar_dir)
    save_to_json(data_entries, output_file)

    print(f"Dataset saved to {output_file}")
