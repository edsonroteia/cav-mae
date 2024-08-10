# -*- coding: utf-8 -*-
# @Time    : 5/22/23 10:00 PM
# @Author  : Yuan Gong & Edson Araujo
# @Affiliation  : Massachusetts Institute of Technology & University of Bonn
# @Email   : yuangong@mit.edu @ edson@araujo.info
# @File    : localization.py

# audio-visual localization experiments

import argparse
import os
import models
import dataloader as dataloader
import torch
import numpy as np
from torch.cuda.amp import autocast
from torch import nn
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def calculate_iou(bbox, map, threshold):
    x1, y1, x2, y2 = bbox
    map_binary = np.zeros_like(map)
    map_binary[map > threshold] = 1
    bbox_map = np.zeros_like(map)
    bbox_map[y1:y2, x1:x2] = 1

    intersection = np.logical_and(map_binary, bbox_map).sum()
    union = np.logical_or(map_binary, bbox_map).sum()

    iou = intersection / union
    return iou

def get_similarity(a, b):
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim

def get_map(patches, audio):
    B, _, _ = patches.shape
    # Reshaping audio is not necessary in your case, it's already in the correct shape.
    # Here we are using np.einsum to compute the dot product along the last axis, 
    # and keeping the other axes separate. This operation is performed separately for each batch.
    similarity = np.einsum('biv,biv->bi', patches, audio)
    # We reshape it back to (B, num_vectors, 1)
    similarity = similarity.reshape(B, 14, 14)
    return similarity


def unnormalize(img, mean, std):
    img = img.permute(1, 2, 0)  # Convert image from (C,H,W) to (H,W,C)
    img = img * std + mean  # Unnormalize: reverse (image - mean) / std
    return img.clamp(0, 1)  # Clamp to make sure image range is between 0 and 1

# direction: 'audio' means audio->visual retrieval, 'video' means visual->audio retrieval
def get_localization_result(audio_model, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    ious = []
    with torch.no_grad():
        for i, (a_input, v_input, labels, bboxes) in enumerate(tqdm(val_loader)):
            audio_input, video_input = a_input.to(device), v_input.to(device)
            with autocast():
                audio_output, video_output = audio_model.module.forward_feat(audio_input, video_input)
                # mean pool all patches
                audio_output = torch.mean(audio_output, dim=1)
                # video_output = torch.mean(video_output, dim=1)
                # normalization
                audio_output = torch.nn.functional.normalize(audio_output, dim=-1).unsqueeze(1)
                video_output = torch.nn.functional.normalize(video_output, dim=-1)
            audio_output = audio_output.to('cpu').detach()
            video_output = video_output.to('cpu').detach()
            localization_map = get_map(video_output, audio_output)
            # Only visualize the first 10 images and stop.
            mean = np.array([0.4850, 0.4560, 0.4060])
            std = np.array([0.2290, 0.2240, 0.2250])
            bboxes = [(np.fromstring(bbox.strip('[]'), sep=',') * 224).astype(int) for bbox in bboxes]
            iou_dict = {}
            # # For each image in the batch
            for j in range(50):
                # Unnormalize the image
                img = unnormalize(v_input[j], mean, std).numpy()
                # Resize the heatmap to match the image size
                heatmap = torch.from_numpy(localization_map[j]).unsqueeze(0).unsqueeze(0)
                resized_map = F.interpolate(heatmap, size=(224, 224), mode='nearest')
                resized_map = resized_map.squeeze().numpy()  
                # Normalize the resized map
                resized_map = resized_map - np.min(resized_map)
                resized_map = resized_map / np.max(resized_map)
                threshold = 0.5
                iou = calculate_iou(bboxes[j], resized_map, threshold)
                # Create binary map
                map_binary = np.zeros_like(resized_map)
                map_binary[resized_map > threshold] = 1
                ious.append(iou)
                
                # Create a figure with 3 subplots
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                
                # Original image
                axs[0].axis('off')
                axs[0].imshow(img)
                axs[0].set_title('Original')

                # Image with resized_map and bbox
                axs[1].axis('off')
                axs[1].imshow(img)
                axs[1].imshow(resized_map, cmap='viridis', alpha=0.5)
                bbox = bboxes[j]
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=3, edgecolor='r', facecolor='none')
                axs[1].add_patch(rect)
                axs[1].set_title('Resized Map and BBox')

                # Image with binary map and bbox
                axs[2].axis('off')
                axs[2].imshow(img)
                axs[2].imshow(map_binary, cmap='viridis', alpha=0.5)
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=3, edgecolor='r', facecolor='none')
                axs[2].add_patch(rect)
                axs[2].set_title('Binary Map and BBox')

                # Save the figure
                img_filename = f"85_imgs/image_{i}_{j}.png"
                plt.savefig(img_filename, bbox_inches='tight', pad_inches=0)
                # Close the figure to free up memory
                plt.close(fig)

                # Store the IoU score and the corresponding image filename in the dictionary
                iou_dict[img_filename] = iou
            # Sort the dictionary by IoU scores in descending order
            sorted_iou_dict = {k: v for k, v in sorted(iou_dict.items(), key=lambda item: item[1], reverse=True)}
            # Print the image filenames in the sorted order
            for filename, iou_score in sorted_iou_dict.items():
                print(f"{filename}: IoU = {iou_score}")
            

            print(f"Finished batch {i}, current iou: {np.mean(ious)}")
            
    return np.mean(ious)

def eval_localization(model, data, audio_conf, label_csv, num_class, model_type='pretrain', batch_size=48):
    print(model)
    print(data)
    frame_use = 5
    # eval setting
    val_audio_conf = audio_conf
    val_audio_conf['frame_use'] = frame_use
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.data_val = data
    args.label_csv = label_csv
    args.exp_dir = './exp/dummy'
    args.loss_fn = torch.nn.BCELoss()
    val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf), batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)
    # cav-mae only been ssl pretrained
    if model_type == 'pretrain':
        audio_model = models.CAVMAE(modality_specific_depth=11)
    # cav-mae only been ssl pretrained + supervisedly finetuned
    elif model_type == 'finetune':
        audio_model = models.CAVMAEFT(label_dim=num_class, modality_specific_depth=11)
    sdA = torch.load(model, map_location=device)
    if isinstance(audio_model, torch.nn.DataParallel) == False:
        audio_model = torch.nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    audio_model.eval()
    iou = get_localization_result(audio_model, val_loader)
    return iou

# use cav-mae scale 108 (batch size) model that has only been pretrained
model = '/home/edson/code/cav-mae/pretrained_model/85.pth'
res = []

# for vggsound
data = '/data1/edson/datasets/VGGSS/sample_data.json'
label_csv = '/data1/edson/datasets/VGGSS/class_labels_indices_vgg.csv'
dataset = 'vggsound'
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}
iou = eval_localization(model, data, audio_conf=audio_conf, label_csv=label_csv, num_class=309, model_type='pretrain', batch_size=50)
print("Final IoI: ", iou)