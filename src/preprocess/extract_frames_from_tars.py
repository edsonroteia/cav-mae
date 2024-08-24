import os
import tarfile
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor
from extract_video_frame_parallel import process_videos
from tqdm import tqdm

def process_tar_file(tar_path, output_base_dir):
    # Extract tar file name without extension
    tar_name = os.path.splitext(os.path.basename(tar_path))[0]
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract tar file
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=temp_dir)
        
        # Get list of all MKV files in the extracted directory
        mkv_files = [os.path.join(root, file)
                     for root, _, files in os.walk(temp_dir)
                     for file in files if file.endswith('.mkv')]
        
        # Create output directory for this tar file
        tar_output_dir = os.path.join(output_base_dir, tar_name)
        os.makedirs(tar_output_dir, exist_ok=True)
        
        # Process videos using the existing script
        process_videos(mkv_files, tar_output_dir)
    
    return tar_path  # Return the processed tar_path

def main(tar_list_file, num_workers, output_base_dir):
    # Read tar files from the input file
    with open(tar_list_file, 'r') as f:
        tar_files = [line.strip() for line in f.readlines()]
    
    # Process tar files in parallel with a progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_tar_file, tar_files, [output_base_dir] * len(tar_files)),
                  total=len(tar_files), desc="Processing tar files", unit="file"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tar files and extract video frames.")
    parser.add_argument("tar_list_file", help="Path to the text file containing the list of tar files")
    parser.add_argument("num_workers", type=int, help="Number of worker processes")
    parser.add_argument("output_base_dir", help="Base output directory for extracted frames")
    
    args = parser.parse_args()
    
    main(args.tar_list_file, args.num_workers, args.output_base_dir)