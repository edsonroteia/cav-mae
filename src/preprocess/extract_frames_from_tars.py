import os
import tarfile
import io
import argparse
from tqdm import tqdm
import logging
from extract_video_frame_parallel import process_videos

def process_tar_chunk(tar_files_chunk, output_base_dir):
    results = []
    for tar_path in tar_files_chunk:
        logging.info(f"Processing tar file: {tar_path}")
        
        # Extract tar file name without extension
        tar_name = os.path.splitext(os.path.basename(tar_path))[0]
        
        # Create a dictionary to store extracted files in memory
        memory_fs = {}
        
        # Extract tar file to memory
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.mkv'):
                    f = tar.extractfile(member)
                    if f is not None:
                        memory_fs[member.name] = io.BytesIO(f.read())
        
        # Get list of all MKV files in the extracted memory filesystem
        mkv_files = list(memory_fs.keys())
        
        # Create output directory for this tar file
        tar_output_dir = os.path.join(output_base_dir, tar_name)
        os.makedirs(tar_output_dir, exist_ok=True)
        
        # Process videos using the existing script
        logging.info(f"Processing {len(mkv_files)} MKV files from {tar_path}")
        process_videos(mkv_files, tar_output_dir, memory_fs)
        
        logging.info(f"Finished processing {tar_path}")
        
        # Clear memory
        del memory_fs
        
        results.append(tar_path)
    
    return results

def main(tar_list_file, output_base_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Read tar files from the input file
    with open(tar_list_file, 'r') as f:
        tar_files = [line.strip() for line in f.readlines()]
    
    # Process tar files one at a time
    for tar_file in tqdm(tar_files, desc="Processing tar files", unit="tar"):
        process_tar_chunk([tar_file], output_base_dir)
    
    logging.info(f"Processed {len(tar_files)} tar files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tar files and extract video frames.")
    parser.add_argument("tar_list_file", help="Path to the text file containing the list of tar files")
    parser.add_argument("output_base_dir", help="Base output directory for extracted frames")
    
    args = parser.parse_args()
    
    main(args.tar_list_file, args.output_base_dir)