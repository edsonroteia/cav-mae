import os
import tarfile
import io
import argparse
from concurrent.futures import ProcessPoolExecutor
from extract_video_frame_parallel import process_videos
from tqdm import tqdm
import multiprocessing
import logging

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

def main(tar_list_file, num_workers, output_base_dir):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Read tar files from the input file
    with open(tar_list_file, 'r') as f:
        tar_files = [line.strip() for line in f.readlines()]
    
    # Calculate chunk size based on available memory and number of workers
    total_tars = len(tar_files)
    chunk_size = max(1, min(50, total_tars // num_workers))
    
    # Split tar files into chunks
    tar_chunks = [tar_files[i:i + chunk_size] for i in range(0, total_tars, chunk_size)]
    
    # If there's only one tar file, process it directly without multiprocessing
    if total_tars == 1:
        logging.info("Processing single tar file without multiprocessing")
        process_tar_chunk(tar_files, output_base_dir)
        processed_tars = 1
    else:
        # Process tar chunks in parallel with a progress bar
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_tar_chunk, tar_chunks, [output_base_dir] * len(tar_chunks)),
                total=len(tar_chunks),
                desc="Processing tar chunks",
                unit="chunk"
            ))
        
        # Flatten results
        processed_tars = [item for sublist in results for item in sublist]
    
    logging.info(f"Processed {len(processed_tars)} tar files out of {total_tars}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tar files and extract video frames.")
    parser.add_argument("tar_list_file", help="Path to the text file containing the list of tar files")
    parser.add_argument("num_workers", type=int, help="Number of worker processes")
    parser.add_argument("output_base_dir", help="Base output directory for extracted frames")
    
    args = parser.parse_args()
    
    main(args.tar_list_file, args.num_workers, args.output_base_dir)