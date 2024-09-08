import torch
from torch.utils.data import DataLoader
import numpy as np
from dataloader import AudiosetDataset as OldDataset
from dataloader_sync import AudiosetDataset as NewDataset
import argparse

def compare_dataloaders(dataset_json_file):
    # Common configuration
    audio_conf = {
        "num_mel_bins": 128,
        "target_length": 1024,
        "freqm": 48,
        "timem": 192,
        "mixup": 0,
        "mode": "train",
        "mean": -4.2677393,
        "std": 4.5689974,
        "noise": True,
        "skip_norm": False
    }
    label_csv = "data/class_labels_indices.csv"  # Adjust this path if needed

    # Initialize both datasets
    old_dataset = OldDataset(args.data_val, label_csv=args.label_csv, audio_conf=audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    new_dataset = NewDataset(args.data_val, label_csv=args.label_csv, audio_conf=audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Create DataLoaders
    old_loader = DataLoader(old_dataset, batch_size=1, shuffle=False)
    new_loader = DataLoader(new_dataset, batch_size=1, shuffle=False)

    # Compare a few samples
    num_samples_to_compare = 5
    for i, ((old_fbank, old_image, old_label), (new_fbank, new_image, new_label, _, _)) in enumerate(zip(old_loader, new_loader)):
        if i >= num_samples_to_compare:
            break

        print(f"Sample {i + 1}:")
        
        # Compare labels
        print("  Old labels:")
        print(old_label.squeeze().numpy())
        print("  New labels:")
        print(new_label.squeeze().numpy())

        # Calculate and print label differences
        label_diff = torch.abs(old_label - new_label)
        print("  Label differences:")
        print(label_diff.squeeze().numpy())

        max_diff = label_diff.max().item()
        print(f"  Maximum label difference: {max_diff:.6f}")

        non_zero_diffs = label_diff[label_diff > 0]
        if len(non_zero_diffs) > 0:
            print(f"  Non-zero differences: {len(non_zero_diffs)} out of {len(label_diff.squeeze())}")
            print(f"  Indices of non-zero differences: {torch.nonzero(label_diff.squeeze()).squeeze().tolist()}")
        else:
            print("  All label values are identical")

        print()

    print("Comparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare dataloaders using a specified dataset JSON file.")
    parser.add_argument("dataset_json", type=str, help="Path to the dataset JSON file")
    args = parser.parse_args()

    compare_dataloaders(args.dataset_json)