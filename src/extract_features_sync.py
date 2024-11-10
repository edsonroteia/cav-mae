# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import torch
from torch import nn
from tqdm import tqdm

def extract_features(audio_model, data_loader, args, split='train', model_id=None):
    """Extract features from the encoder and save them to a file"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()  # Set to evaluation mode
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, (a_input, v_input, labels, _, _) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Extracting {split} features"):
            # Move inputs to device
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            
            # Forward pass through encoder only
            if args.cls_token:
                # Get features including CLS tokens
                av_features = audio_model.module.get_features(a_input, v_input)
            else:
                print("Not supporting non-cls-token models yet")
                exit(1)
            
            # Move to CPU to save memory
            all_features.append(av_features.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Save features and labels
    output_path = os.path.join('/scratch/ssml/araujo/features/', f'{split}_features_{model_id}.pt')
    torch.save({
        'features': all_features,
        'labels': all_labels
    }, output_path)
    
    print(f"Features saved to {output_path}")
    print(f"Features shape: {all_features.shape}, Labels shape: {all_labels.shape}")
    
    return all_features, all_labels
