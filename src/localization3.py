import argparse
import os
import models
import dataloader
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
from sklearn import metrics
from telegram_print import tprint
import cv2

class AVLocalization:
    def __init__(self, device, model):
        self.device = device
        self.mean = np.array([0.4850, 0.4560, 0.4060])
        self.std = np.array([0.2290, 0.2240, 0.2250])
        self.threshold = 0.5
        self.counter = 0
        self.model = model

    @staticmethod
    def avg_heads(cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    @staticmethod
    def apply_self_attention_rules(R_ss, cam_ss):
        R_ss_addition = torch.matmul(cam_ss, R_ss)
        return R_ss_addition

    def generate_relevance(self, model, m_input, index=None):
        audio_output, video_output = model.forward_feat(*m_input, register_hook=True)
        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0, index] = 1
        # one_hot_vector = one_hot
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # one_hot = torch.sum(one_hot * output)

        model.zero_grad()
        video_output.backward(retain_graph=True)

        grad = model.get_gradients()[-1].cpu().data.numpy()
        cam = model.get_activations()[-1].cpu().data.numpy()
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, m_input.shape[2:])

        cam_avg = self.avg_heads(cam, grad)
        cam_rule = self.apply_self_attention_rules(cam, cam_avg)
        return cam_rule

    def calculate_iou(self, bbox, lmap, threshold=0.5):
        # Reshape and interpolate the map to (224, 224)

        lmap, _ = self.process_map(lmap)
        x1, y1, x2, y2 = bbox
        map_binary = np.zeros_like(lmap)
        map_binary[lmap > threshold] = 1
        bbox_map = np.zeros_like(lmap)
        bbox_map[y1:y2, x1:x2] = 1

        intersection = np.logical_and(map_binary, bbox_map).sum()
        union = np.logical_or(map_binary, bbox_map).sum()
        
        if union == 0:
            self.counter += 1
            iou = 0
        else:
            iou = intersection / union

        return iou

    def calculate_iou_batch(self, bboxes, maps):
        iou_batch = [self.calculate_iou(bbox, map) for bbox, map in zip(bboxes, maps)]
        return iou_batch

    def calculate_auc(self, bbox, lmap):
        iou_thresholds = np.linspace(0, 1, 100)
        iou_scores = np.zeros_like(iou_thresholds)
        for idx, threshold in enumerate(iou_thresholds):
            iou = self.calculate_iou(bbox, lmap, threshold)
            iou_scores[idx] = iou
        auc_score = metrics.auc(iou_thresholds, iou_scores)
        return auc_score

    def calculate_auc_batch(self, bboxes, maps):
        auc_batch = [self.calculate_auc(bbox, map) for bbox, map in zip(bboxes, maps)]
        return auc_batch


    def get_args(self, data, label_csv):
        # Define how you process the input data and label_csv to generate arguments.
        # This is just a placeholder and needs to be replaced by your actual implementation.
        args = {
            'data_val': data,
            'label_csv': label_csv
        }
        return args

    def get_map(self, model, input, index=None):
        # Replace the dot product operation with the new generate_relevance method
        relevance_map = self.generate_relevance(model, input, index)

        # Rescale the map values to range between 0 and 1
        min_val = torch.min(relevance_map)
        max_val = torch.max(relevance_map)
        norm_map = (relevance_map - min_val) / (max_val - min_val)

        # Return the normalized relevance map
        return norm_map

    def unnormalize(self, img):
        img = img.permute(1, 2, 0)  # Convert image from (C,H,W) to (H,W,C)
        img = img * self.std + self.mean  # Unnormalize: reverse (image - mean) / std
        return img.clamp(0, 1)  # Clamp to make sure image range is between 0 and 1

    def evaluate(self, model, data_loader):
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        model = model.to(self.device)
        model.eval()

        ious = []
        aucs = [] 
        best_results = []
        worst_results = []

        with torch.no_grad():
            for i, (a_input, v_input, _, bboxes) in enumerate(data_loader):
                # audio_output, video_output = self.forward_pass_patches_audiocolumn(model, a_input, v_input)
                a_input, v_input = a_input.to(self.device), v_input.to(self.device)
                localization_map = self.get_map(model.module, (a_input, v_input), None)
                bboxes = [(np.fromstring(bbox.strip('[]'), sep=',') * 224).astype(int) for bbox in bboxes]
                iou_batch = self.calculate_iou_batch(bboxes, localization_map)
                ious.extend(iou_batch)
                auc_batch = self.calculate_auc_batch(bboxes, localization_map)  
                aucs.extend(auc_batch)

                # Save best and worst results
                for j in range(len(iou_batch)):
                    lmap, binary_map = self.process_map(localization_map[j])
                    result = {
                        'iou': iou_batch[j],
                        'auc': auc_batch[j],
                        'image': self.unnormalize(v_input[j]).numpy(),
                        'bbox': bboxes[j],
                        'map': lmap,
                        'binary_map': binary_map
                    }
                    if i == 0:
                        self.save_result_image(result, f"{self.model}_imgs/{i+1}_{j}.png")
                    if len(best_results) < 5:
                        best_results.append(result)
                        best_results = sorted(best_results, key=lambda x: x['iou'], reverse=True)
                    elif iou_batch[j] > best_results[-1]['iou']:
                        best_results[-1] = result
                        best_results = sorted(best_results, key=lambda x: x['iou'], reverse=True)

                    if len(worst_results) < 5:
                        worst_results.append(result)
                        worst_results = sorted(worst_results, key=lambda x: x['iou'])
                    elif iou_batch[j] < worst_results[-1]['iou']:
                        worst_results[-1] = result
                        worst_results = sorted(worst_results, key=lambda x: x['iou'])

        return np.mean(ious), np.mean(aucs), best_results, worst_results

    def forward_pass_patches_audiomean(self, model, a_input, v_input):
        audio_input, video_input = a_input.to(self.device), v_input.to(self.device)
        with autocast():
            audio_output, video_output = model.module.forward_feat(audio_input, video_input)
            audio_output = torch.mean(audio_output, dim=1)
            audio_output = torch.nn.functional.normalize(audio_output, dim=-1).unsqueeze(1)
            video_output = torch.nn.functional.normalize(video_output, dim=-1)
        audio_output = audio_output.to('cpu').detach()
        video_output = video_output.to('cpu').detach()
        return audio_output, video_output
    
    def forward_pass_patches_patchesmean(self, model, a_input, v_input):
        audio_input, video_input = a_input.to(self.device), v_input.to(self.device)
        with autocast():
            audio_output, video_output = model.module.forward_feat(audio_input, video_input)
            audio_output = torch.mean(video_output, dim=1) # Use video_output as audio_output
            audio_output = torch.nn.functional.normalize(audio_output, dim=-1).unsqueeze(1)
            video_output = torch.nn.functional.normalize(video_output, dim=-1)
        audio_output = audio_output.to('cpu').detach()
        video_output = video_output.to('cpu').detach()
        return audio_output, video_output

    def forward_pass_patches_audiocolumn(self, model, a_input, v_input):
        audio_input, video_input = a_input.to(self.device), v_input.to(self.device)
        with autocast():
            audio_output, video_output = model.module.forward_feat(audio_input, video_input)
            audio_output = audio_output.reshape(audio_output.shape[0], 8, audio_output.shape[1] // 8, audio_output.shape[2])
            audio_output = torch.mean(audio_output, dim=1)[:,32,:]
            audio_output = torch.nn.functional.normalize(audio_output, dim=-1).unsqueeze(1)
            video_output = torch.nn.functional.normalize(video_output, dim=-1)
        audio_output = audio_output.to('cpu').detach()
        video_output = video_output.to('cpu').detach()
        return audio_output, video_output


    def visualize_and_store_results(self, i, v_input, localization_map, bboxes, ious):
        # For each image in the batch
        for j in range(50):
            img = self.unnormalize(v_input[j]).numpy()
            resized_map, map_binary = self.process_map(localization_map[j])
            iou = self.calculate_iou(bboxes[j], resized_map)
            ious.append(iou)
            self.plot_and_save_images(i, j, img, bboxes[j], resized_map, map_binary)
        print(f"Finished batch {i}, current iou: {np.mean(ious)}")

    def process_map(self, localization_map):
        heatmap = torch.from_numpy(localization_map).unsqueeze(0).unsqueeze(0)
        resized_map = F.interpolate(heatmap, size=(224, 224), mode='nearest')
        resized_map = resized_map.squeeze().numpy()
        resized_map = (resized_map - np.min(resized_map)) / (np.max(resized_map) - np.min(resized_map))
        map_binary = np.zeros_like(resized_map)
        map_binary[resized_map > self.threshold] = 1
        return resized_map, map_binary

    @staticmethod
    def plot_image(ax, img, title):
        ax.axis('off')
        ax.imshow(img)
        ax.set_title(title)

    @staticmethod
    def plot_image_with_map_and_bbox(ax, img, lmap, bbox, title):
        ax.axis('off')
        ax.imshow(img)
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.imshow(lmap, cmap='viridis', alpha=0.5)
        ax.set_title(title)
        return ax

    def plot_and_save_images(self, i, j, img, bbox, resized_map, map_binary):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        self.plot_image(axs[0], img, 'Original')
        axs[1] = self.plot_image_with_map_and_bbox(axs[1], img, resized_map, bbox, 'Resized Map and BBox')
        axs[2] = self.plot_image_with_map_and_bbox(axs[2], img, map_binary, bbox, 'Binary Map and BBox')
        img_filename = f"{self.model}_imgs/image_{i}_{j}.png"
        return fig, axs, img_filename


    def save_result_image(self, result, filename):
        img = result['image']
        bbox = result['bbox']
        resized_map = result['map']
        binary_map = result['binary_map']

        fig, axs, _ = self.plot_and_save_images(0, 0, img, bbox, resized_map, binary_map)

        # Set the metrics as the title of the whole figure
        fig.suptitle(f"IoU: {result['iou']:.4f}, auc: {result['auc']:.4f}", fontsize=12)

        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # models_list = ['50', '65', 'base', '85', 'base++']
    models_list = ['base_ft_as_vgg_vgg', 'base_nonorm', 'basep', '50', '65', 'base', '85', 'basepp']
    results = {}


    for model in tqdm(models_list):
        os.makedirs(f'{model}_imgs', exist_ok=True)
        print(f"Running model {model}")
        av_localization = AVLocalization(device, model)
        model = f'/home/edson/code/cav-mae/pretrained_model/{model}.pth'
        data = '/data1/edson/datasets/VGGSS/sample_data.json'
        label_csv = '/data1/edson/datasets/VGGSS/class_labels_indices_vgg.csv'
        dataset = 'vggsound'
        audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': dataset,
                    'mode': 'eval', 'mean': -5.081, 'std': 4.4849, 'noise': False, 'im_res': 224, 'frame_use': 5}

        # Load model
        audio_model = models.CAVMAE(modality_specific_depth=11)
        sdA = torch.load(model, map_location=device)
        if isinstance(audio_model, torch.nn.DataParallel) == False:
            audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sdA, strict=False)
        audio_model.eval()

        # Load data
        args = av_localization.get_args(data, label_csv)
        val_loader = torch.utils.data.DataLoader(dataloader.AudiosetDataset(args['data_val'], label_csv=args['label_csv'], audio_conf=audio_conf), batch_size=50, shuffle=False, num_workers=32, pin_memory=True)

        # Calculate IoU and get best and worst results
        iou, auc, best_results, worst_results = av_localization.evaluate(audio_model, val_loader)
        print("Final IoU:", iou)
        print("Final AUC:", auc)

        results[model] = {'iou': iou, 'auc': auc}

        # Save best and worst results
        save_dir = f"./{model}"
        os.makedirs(save_dir, exist_ok=True)

        for i, result in enumerate(best_results):
            av_localization.save_result_image(result, f"best_result_{i+1}.png")

        for i, result in enumerate(worst_results):
            av_localization.save_result_image(result, f"worst_result_{i+1}.png")
        print("Global counter:", av_localization.counter)

    tprint(results)