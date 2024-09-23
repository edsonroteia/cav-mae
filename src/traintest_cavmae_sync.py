# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

# not rely on supervised feature

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import neptune
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from scipy.optimize import linear_sum_assignment
import io
from PIL import Image
from cosine_scheduler import CosineWarmupScheduler


def log_plot_to_neptune(run, plot_name, fig, step):
    """Log a matplotlib figure to Neptune without saving to disk."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image = Image.open(buffer)
    run[f"visualizations/{plot_name}"].log(neptune.types.File.as_image(image), step=step)
    plt.close(fig)

def visualize_and_log(a_input, v_input, audio_model, step, run):
    print("Logging reconstruction examples to Neptune...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_input, v_input = a_input.to(device), v_input.to(device)
        
    # Get reconstructions
    with torch.no_grad():
        _, _, _, _, _, mask_a, mask_v, _, recon_a, recon_v, _, _ = audio_model(a_input, v_input)
    
    # Select first sample and first frame of the batch for visualization
    a_input_np = a_input[0].cpu().numpy()
    v_input_np = v_input[0].cpu().numpy() 
    recon_a_np = recon_a[0].squeeze().cpu().numpy()
    recon_v_np = recon_v[0].squeeze().cpu().numpy()

    # Ensure correct shapes for visualization
    if v_input_np.shape[0] == 3:  # If channels are first
        v_input_np = np.transpose(v_input_np, (1, 2, 0))
    if recon_v_np.shape[0] == 3:  # If channels are first
        recon_v_np = np.transpose(recon_v_np, (1, 2, 0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Normalize if necessary
    v_input_np = (v_input_np - v_input_np.min()) / (v_input_np.max() - v_input_np.min())
    recon_v_np = (recon_v_np - recon_v_np.min()) / (recon_v_np.max() - recon_v_np.min())

    # Plot original and reconstructed images
    axes[0, 0].imshow(v_input_np)
    axes[0, 0].set_title("Original Image Frame")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon_v_np)
    axes[0, 1].set_title("Reconstructed Image Frame")
    axes[0, 1].axis('off')

    # Plot original and reconstructed audio
    axes[1, 0].imshow(a_input_np, aspect='auto', origin='lower')
    axes[1, 0].set_title("Original Audio")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(recon_a_np, aspect='auto', origin='lower')
    axes[1, 1].set_title("Reconstructed Audio")
    axes[1, 1].axis('off')

    # Save the figure
    fig_path = f"reconstruction_{step}.png"
    plt.savefig(fig_path)
    plt.close(fig)

    # Log the figure to Neptune
    run["visualizations/reconstruction"].upload(neptune.types.File(fig_path))

def train(audio_model, train_loader, train_dataset, test_loader, args, run):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_global_meter, loss_local_meter = AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    run['model_path'] = exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_loss,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    elif args.lr_scheduler == 'cosine':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=5e-6)
        max_iter = args.n_epochs * len(train_loader)
        print("Max Iterations {} = epochs {} * iter_per_epoch{}".format(max_iter, args.n_epochs, len(train_loader)))
        scheduler = scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_epochs=max_iter * 0.1,
            max_epochs=max_iter,
            min_lr=args.lr * 0.1,
            max_lr=args.lr
        )
        print('Using cosine annealing learning rate scheduler over {:d} epochs with minimum lr of 1e-6'.format(args.n_epochs))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 17])  # 8 original metrics + 8 new accuracy metrics
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('current masking ratio is {:.3f} for both modalities; audio mask mode {:s}'.format(args.masking_ratio, args.mask_mode))

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.n_epochs}', leave=True)

        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            (a_input, v_input, labels, video_ids, frame_indices) = batch
            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                if args.global_local_losses:
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, _, _, latent_c_a, latent_c_v, cls_a, cls_v, global_loss_c, local_loss_c = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                else:
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, _, recon_a, recon_v, latent_c_a, latent_c_v = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                    
                    # Calculate our own contrastive accuracies for validation
                    # accuracies = calculate_contrastive_accuracy(latent_c_a.float(), latent_c_v.float(), video_ids, run=run, mode='eval', global_step=global_step)
                
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c = loss.mean(), loss_mae.mean(), loss_mae_a.mean(), loss_mae_v.mean(), loss_c.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.lr_scheduler == 'cosine':
                scheduler.step()
            
            # Update meters
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            if args.global_local_losses:
                loss_global_meter.update(global_loss_c.item(), B)
                loss_local_meter.update(local_loss_c.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            # Log to Neptune
            run["train/total_loss"].append(loss_av_meter.val, step=global_step)
            run["train/mae_loss_audio"].append(loss_a_meter.val, step=global_step)
            run["train/mae_loss_visual"].append(loss_v_meter.val, step=global_step)
            run["train/contrastive_loss"].append(loss_c_meter.val, step=global_step)
            if args.global_local_losses:
                run["train/global_contrastive_loss"].append(loss_global_meter.val, step=global_step)
                run["train/local_contrastive_loss"].append(loss_local_meter.val, step=global_step)
            run["train/contrastive_accuracy"].append(c_acc, step=global_step)
            run["train/learning_rate"].append(optimizer.param_groups[0]['lr'], step=global_step)

            # print_step = global_step % args.n_print_steps == 0
            print_step = global_step % 2000 == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                # print('Epoch: [{0}][{1}/{2}]\t'
                #       'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                #       'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                #       'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                #       'Train Total Loss {loss_av_meter.val:.4f}\t'
                #       'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
                #       'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                #       'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                #       'Train Contrastive Acc {c_acc:.3f}\t'.format(
                #        epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                #           per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, 
                #           loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return
                visualize_and_log(a_input, v_input, audio_model, global_step, run)

            end_time = time.time()
            global_step += 1

        pbar.close()
        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_global, eval_loss_local, eval_accuracies = validate(audio_model, test_loader, args, run, global_step)

        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        if args.global_local_losses:
            print("Eval Global Contrastive Loss: {:.6f}".format(eval_loss_global))
            print("Eval Local Contrastive Loss: {:.6f}".format(eval_loss_local))
        print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        # for k, v in eval_accuracies.items():
        #     print(f"Eval Contrastive Accuracy ({k}): {v:.6f}")

        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        if args.global_local_losses:
            print("Train Global Contrastive Loss: {:.6f}".format(loss_global_meter.avg))
            print("Train Local Contrastive Loss: {:.6f}".format(loss_local_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))

        # Log validation metrics to Neptune
        run["eval/total_loss"].append(eval_loss_av, step=global_step)
        run["eval/mae_loss_audio"].append(eval_loss_mae_a, step=global_step)
        run["eval/mae_loss_visual"].append(eval_loss_mae_v, step=global_step)
        run["eval/contrastive_loss"].append(eval_loss_c, step=global_step)
        if args.global_local_losses:
            run["eval/global_contrastive_loss"].append(eval_loss_global, step=global_step)
            run["eval/local_contrastive_loss"].append(eval_loss_local, step=global_step)
        # for k, v in eval_accuracies.items():
        #     run[f"eval/contrastive_accuracy_{k}"].append(v, step=global_step)
        run["train/epoch_mae_loss_audio"].append(loss_a_meter.avg, step=global_step)
        run["train/epoch_mae_loss_visual"].append(loss_v_meter.avg, step=global_step)
        run["train/epoch_contrastive_loss"].append(loss_c_meter.avg, step=global_step)
        if args.global_local_losses:
            run["train/epoch_global_contrastive_loss"].append(loss_global_meter.avg, step=global_step)
            run["train/epoch_local_contrastive_loss"].append(loss_local_meter.avg, step=global_step)
        run["train/epoch_total_loss"].append(loss_av_meter.avg, step=global_step)
        run["epoch"].append(epoch, step=global_step)

        # Update the result array to include all accuracy metrics
        result[epoch-1, :] = [
            loss_a_meter.avg,
            loss_v_meter.avg,
            loss_c_meter.avg,
            loss_av_meter.avg,
            eval_loss_mae_a,
            eval_loss_mae_v,
            eval_loss_c,
            eval_loss_av
        ]
        # ] + list(eval_accuracies.values()) + [optimizer.param_groups[0]['lr']]
        header = ['train_loss_audio', 'train_loss_visual', 'train_loss_contrastive', 'train_loss_total',
          'eval_loss_mae_audio', 'eval_loss_mae_visual', 'eval_loss_contrastive', 'eval_loss_total',
          'acc_whole_avg', 'acc_whole_max', 'acc_diag_avg', 'acc_diag_max',
          'acc_optimal_avg', 'acc_optimal_max', 'acc_hungarian_avg', 'acc_hungarian_max',
          'learning_rate']
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',', header=','.join(header), comments='')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if args.lr_scheduler != 'cosine':
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(-eval_loss_av)
            else:
                scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()
        if args.global_local_losses:
            loss_global_meter.reset()
            loss_local_meter.reset()

def optimal_path_score(similarity_matrix):
    n, m = similarity_matrix.shape
    score_matrix = torch.zeros_like(similarity_matrix)
    path_matrix = torch.zeros_like(similarity_matrix, dtype=torch.long)

    # Initialize first row and column
    score_matrix[0, :] = torch.cumsum(similarity_matrix[0, :], dim=0)
    score_matrix[:, 0] = torch.cumsum(similarity_matrix[:, 0], dim=0)
    path_matrix[0, 1:] = 1  # horizontal
    path_matrix[1:, 0] = 2  # vertical

    # Fill the score and path matrices
    for i in range(1, n):
        for j in range(1, m):
            if score_matrix[i-1, j] > score_matrix[i, j-1]:
                score_matrix[i, j] = similarity_matrix[i, j] + score_matrix[i-1, j]
                path_matrix[i, j] = 2  # came from above
            else:
                score_matrix[i, j] = similarity_matrix[i, j] + score_matrix[i, j-1]
                path_matrix[i, j] = 1  # came from left

    # Compute the optimal path
    path = []
    i, j = n-1, m-1
    while i > 0 or j > 0:
        path.append((i, j))
        if path_matrix[i, j] == 2:
            i -= 1
        else:
            j -= 1
    path.append((0, 0))
    path.reverse()

    return score_matrix[-1, -1] / len(path), path

def hungarian_matching_score(similarity_matrix):
    # Convert to cost matrix (Hungarian algorithm minimizes cost)
    cost_matrix = 1 - similarity_matrix.cpu().numpy()
    
    # Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate the total similarity score
    total_similarity = similarity_matrix[row_ind, col_ind].sum().item()
    
    # Normalize the score
    normalized_score = total_similarity / len(row_ind)
    
    return normalized_score, list(zip(row_ind, col_ind))

def calculate_contrastive_accuracy(audio_rep, video_rep, video_ids, frame_indices=None, run=None, mode='train', global_step=None):
    # print(f"Mode: {mode}")
    # print(f"Audio rep shape: {audio_rep.shape}, Video rep shape: {video_rep.shape}")
    # print(f"Number of video_ids: {len(video_ids)}")

    audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
    video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
    
    # Compute similarities
    with torch.no_grad():
        similarities = torch.mm(audio_rep, video_rep.t())
    # print(f"Similarities shape: {similarities.shape}")
    
    # Visualize the similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(similarities.detach().cpu().numpy(), cmap='viridis', ax=ax)
    ax.set_title(f'Similarity Matrix - {mode.capitalize()} Mode')
    ax.set_xlabel('Video Representations')
    ax.set_ylabel('Audio Representations')
    
    # Log the figure to Neptune
    if run is not None:
        log_plot_to_neptune(run, f"similarity_matrix_{mode}", fig, global_step)
    else:
        plt.close(fig)
    
    if mode == 'train':
        # For training, we use argmax as before
        accuracy = (torch.argmax(similarities, dim=1) == torch.arange(similarities.shape[0], device=similarities.device)).float().mean().item()
        return accuracy
    else:  # eval mode
        # For eval, we calculate accuracy based on all aggregation methods
        unique_video_ids = list(set(video_ids))
        accuracies = {
            'whole_avg': [], 'whole_max': [],
            'diag_avg': [], 'diag_max': [],
            'optimal_avg': [], 'optimal_max': [],
            'hungarian_avg': [], 'hungarian_max': []
        }
        
        for vid in unique_video_ids:
            vid_indices = [i for i, v in enumerate(video_ids) if v == vid]
            max_index = similarities.shape[0] - 1
            vid_indices = [min(i, max_index) for i in vid_indices if i <= max_index]
            vid_similarities = similarities[vid_indices][:, vid_indices]
            
            # 1-2. Whole block average and max
            accuracies['whole_avg'].append(vid_similarities.mean().item())
            accuracies['whole_max'].append(vid_similarities.max().item())
            
            # 3-4. Diagonal average and max
            diag_similarities = torch.diag(vid_similarities)
            accuracies['diag_avg'].append(diag_similarities.mean().item())
            accuracies['diag_max'].append(diag_similarities.max().item())
            
            # 5-6. Optimal path average and max
            optimal_score, optimal_path = optimal_path_score(vid_similarities)
            optimal_similarities = torch.tensor([vid_similarities[i, j] for i, j in optimal_path])
            accuracies['optimal_avg'].append(optimal_similarities.mean().item())
            accuracies['optimal_max'].append(optimal_similarities.max().item())
            
            # 7-8. Hungarian matching average and max
            hungarian_score, matching = hungarian_matching_score(vid_similarities)
            hungarian_similarities = torch.tensor([vid_similarities[i, j] for i, j in matching])
            accuracies['hungarian_avg'].append(hungarian_similarities.mean().item())
            accuracies['hungarian_max'].append(hungarian_similarities.max().item())
            
            # # Visualizations
            # if run is not None:
            #     # Optimal path visualization
            #     fig, ax = plt.subplots(figsize=(8, 6))
            #     sns.heatmap(vid_similarities.cpu().numpy(), cmap='viridis', ax=ax)
            #     path_y, path_x = zip(*optimal_path)
            #     ax.plot(path_x, path_y, color='red', linewidth=2, linestyle='--')
            #     ax.set_title(f'Optimal Path for Video {vid}')
            #     ax.set_xlabel('Video Frames')
            #     ax.set_ylabel('Audio Frames')
            #     log_plot_to_neptune(run, f"optimal_path_{vid}", fig, global_step)
                
            #     # Hungarian matching visualization
            #     fig, ax = plt.subplots(figsize=(8, 6))
            #     sns.heatmap(vid_similarities.cpu().numpy(), cmap='viridis', ax=ax)
            #     match_y, match_x = zip(*matching)
            #     ax.scatter(match_x, match_y, color='red', s=100, marker='x')
            #     ax.set_title(f'Hungarian Matching for Video {vid}')
            #     ax.set_xlabel('Video Frames')
            #     ax.set_ylabel('Audio Frames')
            #     log_plot_to_neptune(run, f"hungarian_matching_{vid}", fig, global_step)
        
        # Calculate overall accuracies
        overall_accuracies = {k: np.mean(v) for k, v in accuracies.items()}
        
        # print(f"Computed accuracies: {overall_accuracies}")
        
        return overall_accuracies


def validate(audio_model, val_loader, args, run, global_step):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c = [], [], [], [], []
    A_loss_global, A_loss_local = [], []
    A_accuracies = {
        'whole_avg': [], 'whole_max': [],
        'diag_avg': [], 'diag_max': [],
        'optimal_avg': [], 'optimal_max': [],
        'hungarian_avg': [], 'hungarian_max': []
    }

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=True)
        accuracies ={}
        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            (a_input, v_input, labels, video_ids, frame_indices) = batch
            # print(f"Batch {i}: a_input shape: {a_input.shape}, v_input shape: {v_input.shape}")
            # print(f"Number of video_ids: {len(video_ids)}")
            # print(f"Number of unique video_ids: {len(set(video_ids))}")
            # print(f"Frame indices shape: {frame_indices.shape}")
            
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            
            with autocast():
                if args.global_local_losses:
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, _, recon_a, recon_v, latent_c_a, latent_c_v, cls_a, cls_v, global_loss_c, local_loss_c = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                else:
                    loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, _, recon_a, recon_v, latent_c_a, latent_c_v = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                
                # print(f"latent_c_a shape: {latent_c_a.shape}, latent_c_v shape: {latent_c_v.shape}")
                
                # Calculate our own contrastive accuracies for validation
                # accuracies_local = calculate_contrastive_accuracy(latent_c_a.float(), latent_c_v.float(), video_ids, run=run, mode='eval', global_step=global_step)
                # accuracies_global = calculate_contrastive_accuracy(cls_a.float(), cls_v.float(), video_ids, run=run, mode='eval', global_step=global_step)

                loss = loss.mean()
                loss_mae = loss_mae.mean()
                loss_mae_a = loss_mae_a.mean()
                loss_mae_v = loss_mae_v.mean()
                loss_c = loss_c.mean()
                if args.global_local_losses:
                    loss_global = global_loss_c.mean()
                    loss_local = local_loss_c.mean()
            
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            if args.global_local_losses:
                A_loss_global.append(global_loss_c.to('cpu').detach())
                A_loss_local.append(local_loss_c.to('cpu').detach())
            # for k, v in accuracies.items():
                # A_accuracies[k].append(v)
            
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 10 == 0:  # Print every 10 batches
            #     print(f'Validation Batch: [{i}/{len(val_loader)}]\t'
            #           f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           f'Loss {loss.item():.4f}')
                # for k, v in accuracies.items():
                #     print(f'{k}: {v:.4f}')

        pbar.close()
        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        if args.global_local_losses:
            loss_global = np.mean(A_loss_global)
            loss_local = np.mean(A_loss_local)
        # accuracies = {k: np.mean(v) for k, v in A_accuracies.items()}

    print(f"Validation Results - Loss: {loss:.4f}, MAE Loss: {loss_mae:.4f}, "
          f"MAE Loss Audio: {loss_mae_a:.4f}, MAE Loss Visual: {loss_mae_v:.4f}, "
          f"Contrastive Loss: {loss_c:.4f}")
    # for k, v in accuracies.items():
        # print(f"Accuracy ({k}): {v:.4f}")

    if args.global_local_losses:
        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, loss_global, loss_local, accuracies
    else:
        return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, accuracies
