import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
import io
import multiprocessing
import logging
import torch
import uuid

# Ensure this path matches where you mounted the RAM disk
RAM_DISK_PATH = '/mnt/ramdisk'

# Move this outside the function to avoid recreating it for each video
preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])

def extract_frame(input_video_path, target_fold, extract_frame_num=16, memory_fs=None):
    logging.info(f"Starting frame extraction for: {input_video_path}")
    video_id = os.path.splitext(os.path.basename(input_video_path))[0]
    logging.info(f"Video ID: {video_id}")
    
    temp_file_path = None
    try:
        if memory_fs and input_video_path in memory_fs:
            video_bytes = memory_fs[input_video_path].getvalue()
            # Create a temporary file in the RAM disk
            temp_file_path = os.path.join(RAM_DISK_PATH, f"{uuid.uuid4()}.mkv")
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(video_bytes)
            vidcap = cv2.VideoCapture(temp_file_path)
        else:
            vidcap = cv2.VideoCapture(input_video_path)
        
        if not vidcap.isOpened():
            logging.error(f"Failed to open video: {input_video_path}")
            return

        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, extract_frame_num, dtype=int)
        
        frames = []
        for idx in frame_indices:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = vidcap.read()
            if not success:
                logging.warning(f"Failed to read frame at index {idx} from {input_video_path}")
                continue
            frames.append(frame)
        
        vidcap.release()
        
        if not frames:
            logging.error(f"No frames extracted from {input_video_path}")
            return
        
        # Process all frames at once
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        frames = [Image.fromarray(frame) for frame in frames]
        image_tensors = torch.stack([preprocess(frame) for frame in frames])
        
        # Save all frames at once
        for i, image_tensor in enumerate(image_tensors):
            frame_dir = os.path.join(target_fold, f'frame_{i}')
            os.makedirs(frame_dir, exist_ok=True)
            output_path = os.path.join(frame_dir, f'{video_id}.jpg')
            save_image(image_tensor, output_path)
        
        logging.info(f"Extracted and saved {len(frames)} frames from {input_video_path}")
    
    except Exception as e:
        logging.error(f"Error processing video {input_video_path}: {str(e)}")
    
    finally:
        # Clean up the temporary file if it was created
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def process_videos(input_file_list, target_fold, memory_fs=None):
    logging.info(f"Processing {len(input_file_list)} videos")
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
        executor.map(extract_frame, input_file_list, [target_fold]*len(input_file_list), [16]*len(input_file_list), [memory_fs]*len(input_file_list))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = ArgumentParser(description="Python script to extract frames from a video, save as jpgs.")
    parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="CSV file of video paths.")
    parser.add_argument("-target_fold", type=str, default='./sample_frames/', help="Directory for extracted frames.")
    args = parser.parse_args()

    input_filelist = np.loadtxt(args.input_file_list, dtype=str, delimiter=',')
    print(f'Total {len(input_filelist)} videos are input')
    process_videos(input_filelist, args.target_fold)