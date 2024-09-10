import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import models
import dataloader
import dataloader_sync

def validate_inputs(model_path: str, data_path: str, label_csv_path: str) -> None:
    for path in [model_path, data_path, label_csv_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

def gpu_accelerated_similarity(a: torch.Tensor, b: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for similarity computation")

    a = a.cuda()
    b = b.cuda()

    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    a_loader = DataLoader(TensorDataset(a), batch_size=batch_size, shuffle=False)

    similarity_matrix = []

    for a_batch in tqdm(a_loader, desc="Computing similarity matrix"):
        a_batch = a_batch[0]
        sim_batch = torch.mm(a_batch, b.t())
        similarity_matrix.append(sim_batch.cpu())

    similarity_matrix = torch.cat(similarity_matrix, dim=0)

    return similarity_matrix

def compute_metrics(x: np.ndarray) -> dict:
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]

    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['MR'] = np.median(ind) + 1

    return metrics

def print_computed_metrics(metrics: dict) -> None:
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'
          .format(metrics['R1'], metrics['R5'], metrics['R10'], metrics['MR']))

def get_retrieval_result(audio_model: nn.Module, val_loader: DataLoader, direction: str = 'audio') -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing batches"):
            a_input, v_input = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast():
                audio_output, video_output = audio_model.module.forward_feat(a_input, v_input)
                audio_output = torch.mean(audio_output, dim=1)
                video_output = torch.mean(video_output, dim=1)
            A_a_feat.append(audio_output.cpu())
            A_v_feat.append(video_output.cpu())

    A_a_feat = torch.cat(A_a_feat)
    A_v_feat = torch.cat(A_v_feat)
    
    if direction == 'audio':
        sim_mat = gpu_accelerated_similarity(A_a_feat, A_v_feat)
    else:
        sim_mat = gpu_accelerated_similarity(A_v_feat, A_a_feat)
    
    result = compute_metrics(sim_mat.numpy())
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def get_sync_retrieval_result(audio_model: nn.Module, val_loader: DataLoader, direction: str = 'audio') -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    A_a_feat, A_v_feat = [], []
    video_ids = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Processing batches"):
            fbanks, images, _, batch_video_ids, _ = batch
            fbanks, images = fbanks.to(device), images.to(device)
            
            with torch.cuda.amp.autocast():
                audio_output, video_output = audio_model.module.forward_feat(fbanks, images)
                
            audio_output = audio_output.view(len(batch_video_ids), -1, audio_output.size(-1))
            video_output = video_output.view(len(batch_video_ids), -1, video_output.size(-1))
            
            A_a_feat.append(audio_output.cpu())
            A_v_feat.append(video_output.cpu())
            video_ids.extend(batch_video_ids)

    A_a_feat = torch.cat(A_a_feat, dim=0)
    A_v_feat = torch.cat(A_v_feat, dim=0)

    if direction == 'audio':
        sim_mat = get_sync_sim_mat(A_a_feat, A_v_feat, video_ids)
    else:
        sim_mat = get_sync_sim_mat(A_v_feat, A_a_feat, video_ids, is_video_query=True)

    result = compute_metrics(sim_mat)
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']

def get_sync_sim_mat(query_feat: torch.Tensor, target_feat: torch.Tensor, video_ids: list, is_video_query: bool = False) -> np.ndarray:
    unique_video_ids = list(set(video_ids))
    num_queries = len(unique_video_ids)
    num_targets = len(unique_video_ids)
    sim_mat = torch.zeros((num_queries, num_targets))

    for i, query_video_id in tqdm(enumerate(unique_video_ids), desc="Computing sync similarity matrix", total=num_queries):
        query_mask = torch.tensor([vid == query_video_id for vid in video_ids])
        query_features = query_feat[query_mask]

        for j, target_video_id in enumerate(unique_video_ids):
            target_mask = torch.tensor([vid == target_video_id for vid in video_ids])
            target_features = target_feat[target_mask]

            q_feat = query_features.view(-1, query_features.size(-1))
            t_feat = target_features.view(-1, target_features.size(-1))
            frame_similarities = torch.matmul(q_feat, t_feat.t())

            frame_similarities = frame_similarities.view(query_features.size(0), query_features.size(1), 
                                                         target_features.size(0), target_features.size(1))

            sim_mat[i, j] = frame_similarities.max()

    return sim_mat.numpy()

def eval_retrieval(model: str, data: str, audio_conf: dict, label_csv: str, direction: str, num_class: int, model_type: str = 'pretrain', batch_size: int = 48) -> tuple:
    frame_use = 10
    val_audio_conf = audio_conf.copy()
    val_audio_conf['frame_use'] = frame_use
    
    if model_type == 'sync_pretrain':
        val_dataset = dataloader_sync.AudiosetDataset(data, label_csv=label_csv, audio_conf=val_audio_conf)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            collate_fn=dataloader_sync.eval_collate_fn
        )
        audio_model = models.CAVMAESync(audio_length=val_audio_conf['target_length'], modality_specific_depth=11)
    else:
        val_dataset = dataloader.AudiosetDataset(data, label_csv=label_csv, audio_conf=val_audio_conf)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=32,
            pin_memory=True
        )
        if model_type == 'pretrain':
            audio_model = models.CAVMAE(modality_specific_depth=11)
        elif model_type == 'finetune':
            audio_model = models.CAVMAEFT(label_dim=num_class, modality_specific_depth=11)

    sdA = torch.load(model, map_location='cpu')
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    msg = audio_model.load_state_dict(sdA, strict=False)
    print(msg)
    audio_model.eval()
    
    if model_type == 'sync_pretrain':
        r1, r5, r10, mr = get_sync_retrieval_result(audio_model, val_loader, direction)
    else:
        r1, r5, r10, mr = get_retrieval_result(audio_model, val_loader, direction)
    
    return r1, r5, r10, mr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-Visual Retrieval Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to the label CSV file')
    parser.add_argument('--dataset', type=str, choices=['audioset', 'vggsound'], required=True, help='Dataset to use')
    parser.add_argument('--model_type', type=str, choices=['pretrain', 'sync_pretrain', 'finetune'], required=True, help='Type of model')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='./retrieval_result.csv', help='Path to save the results')

    args = parser.parse_args()

    validate_inputs(args.model, args.data, args.label_csv)

    target_length = 1024 if args.model_type != 'sync_pretrain' else 96

    res = []

    for direction in ['video', 'audio']:
        audio_conf = {
            'num_mel_bins': 128, 
            'target_length': target_length, 
            'freqm': 0, 
            'timem': 0, 
            'mixup': 0, 
            'dataset': args.dataset,
            'mode': 'eval', 
            'mean': -5.081, 
            'std': 4.4849, 
            'noise': False, 
            'im_res': 224, 
            'frame_use': 10,
            'num_samples': 100
        }
        
        if args.dataset == "audioset":
            num_class = 527
        elif args.dataset == "vggsound":
            num_class = 309
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")

        r1, r5, r10, mr = eval_retrieval(args.model, args.data, audio_conf=audio_conf, label_csv=args.label_csv, 
                                         num_class=num_class, direction=direction, model_type=args.model_type, 
                                         batch_size=args.batch_size)
        res.append([args.dataset, direction, r1, r5, r10, mr])

    np.savetxt(args.output, res, delimiter=',', fmt='%s')

    print(f"Results saved to {args.output}")