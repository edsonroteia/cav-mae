import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser

preprocess = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor()])

def extract_frame(input_video_path, target_fold, extract_frame_num=16):
    ext_len = len(input_video_path.split('/')[-1].split('.')[-1])
    video_id = input_video_path.split('/')[-1][:-ext_len-1]
    vidcap = cv2.VideoCapture(input_video_path)
    
    if not vidcap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frame_num = min(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)), int(fps * 10))

    for i in range(extract_frame_num):
        frame_idx = int(i * (total_frame_num / extract_frame_num))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        success, frame = vidcap.read()
        if not success:
            print(f"Failed to read frame at index {frame_idx} from {input_video_path}")
            continue
        
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        image_tensor = preprocess(pil_im)
        
        frame_dir = os.path.join(target_fold, f'frame_{i}')
        os.makedirs(frame_dir, exist_ok=True)
        save_image(image_tensor, os.path.join(frame_dir, f'{video_id}.jpg'))

def process_videos(input_file_list, target_fold):
    with ProcessPoolExecutor(max_workers=48) as executor:
        executor.map(extract_frame, input_file_list, [target_fold]*len(input_file_list))

if __name__ == "__main__":
    parser = ArgumentParser(description="Python script to extract frames from a video, save as jpgs.")
    parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="CSV file of video paths.")
    parser.add_argument("-target_fold", type=str, default='./sample_frames/', help="Directory for extracted frames.")
    args = parser.parse_args()

    input_filelist = np.loadtxt(args.input_file_list, dtype=str, delimiter=',')
    print(f'Total {len(input_filelist)} videos are input')
    process_videos(input_filelist, args.target_fold)