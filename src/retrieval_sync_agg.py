import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def optimized_sync_sim_mat(query_feat: torch.Tensor, target_feat: torch.Tensor, video_ids: list, is_video_query: bool = False, batch_size: int = 512) -> torch.Tensor:
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs for sync similarity computation")

    unique_video_ids = list(set(video_ids))
    num_queries = len(unique_video_ids)
    num_targets = len(unique_video_ids)

    # Create a mapping from video_id to index
    video_id_to_index = {vid: i for i, vid in enumerate(unique_video_ids)}

    # Prepare data for GPU processing
    query_features = []
    target_features = []
    for vid in unique_video_ids:
        mask = torch.tensor([v == vid for v in video_ids])
        query_features.append(query_feat[mask])
        target_features.append(target_feat[mask])

    # Pad sequences to same length
    max_length = max(max(f.size(0) for f in query_features), max(f.size(0) for f in target_features))
    query_features = [F.pad(f, (0, 0, 0, max_length - f.size(0))) for f in query_features]
    target_features = [F.pad(f, (0, 0, 0, max_length - f.size(0))) for f in target_features]

    # Stack features
    query_features = torch.stack(query_features)
    target_features = torch.stack(target_features)

    # Normalize features
    query_features = F.normalize(query_features, p=2, dim=-1)
    target_features = F.normalize(target_features, p=2, dim=-1)

    # Split data across GPUs
    query_features = query_features.chunk(num_gpus)
    target_features = target_features.chunk(num_gpus)

    # Move data to GPUs
    query_features = [qf.to(f'cuda:{i}') for i, qf in enumerate(query_features)]
    target_features = [tf.to(f'cuda:{i}') for i, tf in enumerate(target_features)]

    sim_mat = torch.zeros((num_queries, num_targets), device='cuda:0')

    # Process in batches
    for i in tqdm(range(0, num_queries, batch_size), desc="Computing sync similarity matrix"):
        batch_end = min(i + batch_size, num_queries)
        batch_size_actual = batch_end - i

        # Compute similarities for this batch on all GPUs
        batch_sims = []
        for gpu in range(num_gpus):
            q_batch = query_features[gpu][i:batch_end]
            t_all = target_features[gpu]

            # Reshape for batch matrix multiplication
            q_batch = q_batch.view(batch_size_actual * max_length, -1)
            t_all = t_all.view(num_targets * max_length, -1).t()

            # Compute similarities
            sims = torch.mm(q_batch, t_all)
            sims = sims.view(batch_size_actual, max_length, num_targets, max_length)
            sims = sims.max(dim=1)[0].max(dim=-1)[0]
            batch_sims.append(sims)

        # Combine results from all GPUs
        batch_sims = torch.cat(batch_sims, dim=0)
        sim_mat[i:batch_end] = batch_sims.max(dim=0)[0]

    return sim_mat.cpu()

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
        sim_mat = optimized_sync_sim_mat(A_a_feat, A_v_feat, video_ids)
    else:
        sim_mat = optimized_sync_sim_mat(A_v_feat, A_a_feat, video_ids, is_video_query=True)

    result = compute_metrics(sim_mat.numpy())
    print_computed_metrics(result)
    return result['R1'], result['R5'], result['R10'], result['MR']