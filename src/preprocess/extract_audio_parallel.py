import os
import numpy as np
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor

def extract_features(input_f, target_fold):
    ext_len = len(input_f.split('/')[-1].split('.')[-1])
    video_id = input_f.split('/')[-1][:-ext_len-1]
    output_f_1 = os.path.join(target_fold, video_id + '_intermediate.wav')
    output_f_2 = os.path.join(target_fold, video_id + '.wav')

    # Check if the intermediate file exists to avoid redundant processing
    if not os.path.exists(output_f_1):
        subprocess.run(['ffmpeg', '-i', input_f, '-vn', '-ar', '16000', output_f_1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    
    # Extract the first channel and remove the intermediate file
    if os.path.exists(output_f_1):
        subprocess.run(['sox', output_f_1, output_f_2, 'remix', '1'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        os.remove(output_f_1)

def main():
    parser = argparse.ArgumentParser(description='Easy video feature extractor')
    parser.add_argument("-input_file_list", type=str, default='sample_video_extract_list.csv', help="Should be a csv file of a single column, each row is the input video path.")
    parser.add_argument("-target_fold", type=str, default='./sample_audio/', help="The place to store the video frames.")
    args = parser.parse_args()

    input_filelist = np.loadtxt(args.input_file_list, delimiter=',', dtype=str)
    
    if not os.path.exists(args.target_fold):
        os.makedirs(args.target_fold)
    
    # Use as many workers as there are cores, or consider a few less if you need to leave resources for other processes
    with ProcessPoolExecutor(max_workers=44) as executor:
        executor.map(extract_features, input_filelist, [args.target_fold]*len(input_filelist))

if __name__ == '__main__':
    main()