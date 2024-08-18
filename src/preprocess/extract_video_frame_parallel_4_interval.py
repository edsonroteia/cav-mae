import os
import cv2
import numpy as np
import tarfile
import random
from tqdm import tqdm
from argparse import ArgumentParser
import tempfile

def extract_frames_from_video(video_bytes, video_path, target_folder):
    # Create a temporary file to store the MKV data
    with tempfile.NamedTemporaryFile(suffix='.mkv', delete=False) as temp_file:
        temp_file.write(video_bytes)
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        os.unlink(temp_file_path)
        return 0
    
    frame_count = 0
    saved_count = 0
    
    # Create directory structure mirroring the original
    video_dir = os.path.dirname(video_path)
    frame_dir = os.path.join(target_folder, video_dir)
    os.makedirs(frame_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 4 == 0:
            frame_filename = os.path.join(frame_dir, f"{video_name}_frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    os.unlink(temp_file_path)
    return saved_count

def process_tar_file(tar_file_path, target_folder, max_videos=10):
    total_frames_extracted = 0
    videos_processed = 0
    
    with tarfile.open(tar_file_path, 'r') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith('.mkv')]
        
        # Randomly shuffle the members list
        random.shuffle(members)
        
        for member in tqdm(members[:max_videos], desc="Processing videos"):
            video_file = tar.extractfile(member)
            video_bytes = video_file.read()
            
            frames_extracted = extract_frames_from_video(video_bytes, member.name, target_folder)
            total_frames_extracted += frames_extracted
            videos_processed += 1
            
            print(f"Processed: {member.name}, Frames extracted: {frames_extracted}")
            
            if videos_processed >= max_videos:
                break
    
    return total_frames_extracted, videos_processed

def main(tar_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    
    # Get all tar files in the folder
    tar_files = [f for f in os.listdir(tar_folder) if f.endswith('.tar')]
    
    if not tar_files:
        print("No tar files found in the specified folder.")
        return
    
    # Randomly select a tar file
    selected_tar = random.choice(tar_files)
    tar_file_path = os.path.join(tar_folder, selected_tar)
    
    print(f"Randomly selected tar file: {selected_tar}")
    print(f"Processing up to 10 MKV videos from {tar_file_path}")
    
    total_frames, videos_processed = process_tar_file(tar_file_path, target_folder)
    
    print(f"Total videos processed: {videos_processed}")
    print(f"Total frames extracted: {total_frames}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Test script to extract every 4th frame from 10 MKV videos in a randomly selected tar archive.")
    parser.add_argument("-tar_folder", type=str, required=True, help="Folder containing tar files.")
    parser.add_argument("-target_folder", type=str, required=True, help="Directory for extracted frames.")
    args = parser.parse_args()

    main(args.tar_folder, args.target_folder)
