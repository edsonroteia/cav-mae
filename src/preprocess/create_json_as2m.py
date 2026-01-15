# -*- coding: utf-8 -*-
# @Time    : 2025-01-15
# @Author  : Adapted for AudioSet 2M
# @File    : create_json_as2m.py

"""
Create JSON datafile for AudioSet 2M pretraining.

This script pairs audio files from AS2M-audios with frames from opt-audioset-2M-frames.
The frames are organized in 1000 batch directories (dataset_0000 to dataset_0999).

Output JSON format:
{
    "data": [
        {
            "video_id": "-03SKm9QfMU",
            "wav": "/path/to/audio/-03SKm9QfMU.wav",
            "video_path": "/path/to/frames/dataset_0000",
            "labels": "/m/09x0r"
        },
        ...
    ]
}
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# Configuration
DATASET_ROOT = "/weka/kuehne/kqr867/datasets/audioset"
AUDIO_DIR = os.path.join(DATASET_ROOT, "AS2M-audios/audio")
FRAMES_DIR = os.path.join(DATASET_ROOT, "opt-audioset-2M-frames")
OUTPUT_FILE = "/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_2m_pretrain.json"

# Dummy label for pretraining (not used in loss computation)
DUMMY_LABEL = "/m/09x0r"  # "Speech" label

def get_audio_files():
    """Get set of all available audio file IDs."""
    audio_ids = set()
    for f in os.listdir(AUDIO_DIR):
        if f.endswith('.wav'):
            audio_ids.add(f[:-4])  # Remove .wav extension
    return audio_ids

def create_as2m_json():
    """Create JSON datafile for AudioSet 2M pretraining."""
    print("Loading audio file IDs...")
    audio_ids = get_audio_files()
    print(f"Found {len(audio_ids)} audio files")

    data = []
    matched = 0
    unmatched = 0

    # Get list of batch directories
    batch_dirs = sorted([d for d in os.listdir(FRAMES_DIR) if d.startswith('dataset_')])
    print(f"Found {len(batch_dirs)} batch directories")

    # Process each batch directory
    for batch_dir in tqdm(batch_dirs, desc="Processing batches"):
        batch_path = os.path.join(FRAMES_DIR, batch_dir)
        frame0_path = os.path.join(batch_path, "frame_0")

        if not os.path.exists(frame0_path):
            print(f"Warning: {frame0_path} does not exist, skipping")
            continue

        # Get video IDs from frame_0 directory
        for frame_file in os.listdir(frame0_path):
            if frame_file.endswith('.jpg'):
                video_id = frame_file[:-4]  # Remove .jpg extension

                # Check if corresponding audio exists
                if video_id in audio_ids:
                    entry = {
                        "video_id": video_id,
                        "wav": os.path.join(AUDIO_DIR, f"{video_id}.wav"),
                        "video_path": batch_path,
                        "labels": DUMMY_LABEL
                    }
                    data.append(entry)
                    matched += 1
                else:
                    unmatched += 1

    print(f"\nMatched: {matched}, Unmatched: {unmatched}")
    print(f"Total entries: {len(data)}")

    # Save JSON
    output = {"data": data}
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=1)

    print("Done!")
    return len(data)

def create_eval_yuan_json():
    """Create JSON datafile for audioset-eval-yuan (evaluation set)."""
    eval_audio_dir = os.path.join(DATASET_ROOT, "audioset-eval-yuan/audio")
    eval_frames_dir = os.path.join(DATASET_ROOT, "audioset-eval-yuan/frames")
    output_file = "/weka/kuehne/kqr867/code/cav-mae/datafiles/audioset_eval_yuan.json"

    print("\nCreating evaluation JSON...")

    data = []
    for audio_file in tqdm(os.listdir(eval_audio_dir), desc="Processing eval audio"):
        if audio_file.endswith('.wav'):
            video_id = audio_file[:-4]

            # Verify frame exists
            frame0_path = os.path.join(eval_frames_dir, "frame_0", f"{video_id}.jpg")
            if os.path.exists(frame0_path):
                entry = {
                    "video_id": video_id,
                    "wav": os.path.join(eval_audio_dir, audio_file),
                    "video_path": eval_frames_dir,
                    "labels": DUMMY_LABEL
                }
                data.append(entry)

    print(f"Total eval entries: {len(data)}")

    output = {"data": data}
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=1)

    print(f"Saved to {output_file}")
    return len(data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all", choices=["train", "eval", "all"],
                       help="Which JSON to create: train (2M), eval (yuan), or all")
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        create_as2m_json()

    if args.mode in ["eval", "all"]:
        create_eval_yuan_json()
