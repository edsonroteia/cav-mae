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

def visualize_and_log(a_input, v_input, audio_model, step):
    print("Logging examples to wandb...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_input, v_input = a_input.to(device), v_input.to(device)
        
    # Get reconstructions
    with torch.no_grad():
        _, _, _, _, _, mask_a, mask_v, _, recon_a, recon_v, _, _ = audio_model(a_input, v_input)
    # Select first sample and first frame of the batch for visualization
    a_input_np = a_input[0].cpu().numpy()
    v_input_np = v_input[0, :].cpu().numpy() 
    recon_a_np = recon_a[0].squeeze().cpu().numpy()
    recon_v_np = recon_v[0, :].cpu().numpy()

    # print("Audio Shape:", a_input.shape) #torch.Size([200, 1024, 128])

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    if v_input_np.max() > 1 or v_input_np.min() < 0:
      v_input_np = (v_input_np - v_input_np.min()) / (v_input_np.max() - v_input_np.min())
      recon_v_np = (recon_v_np - recon_v_np.min()) / (recon_v_np.max() - recon_v_np.min())

    # Plot original and reconstructed images
    axes[0, 0].imshow(v_input_np.transpose(1, 2, 0))
    axes[0, 0].set_title("Original Image Frame")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(recon_v_np)
    axes[0, 1].set_title("Reconstructed Image Frame")
    axes[0, 1].axis('off')

    # Plot original and reconstructed audio
    axes[1, 0].imshow(a_input_np.transpose(1,0), aspect='equal', origin='lower')
    axes[1, 0].set_title("Original Audio")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(recon_a_np, aspect='equal', origin='lower')
    axes[1, 1].set_title("Reconstructed Audio")
    axes[1, 1].axis('off')

    # Save the figure and log it to W&B
    fig_path = os.path.join("visualizations", f"step_{step}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)

    wandb.log({"Visualizations": wandb.Image(fig_path)}, step=step)

def train(audio_model, train_loader, train_dataset, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

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
    result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('current masking ratio is {:.3f} for both modalities; audio mask mode {:s}'.format(args.masking_ratio, args.mask_mode))

        for i, (a_input, v_input, _) in tqdm(enumerate(train_loader)):
            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, _, _, _, _ = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            wandb.log({"Train Total Loss": loss_av_meter.val, "Train MAE Loss Audio": loss_a_meter.val, "Train MAE Loss Visual": loss_v_meter.val, "Train Contrastive Loss": loss_c_meter.val, "Train Contrastive Acc": c_acc}, step=global_step)

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Total Loss {loss_av_meter.val:.4f}\t'
                  'Train MAE Loss Audio {loss_a_meter.val:.4f}\t'
                  'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                  'Train Contrastive Acc {c_acc:.3f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

            if global_step % 2900 == 0 and args.mae_loss_weight != 0:
                visualize_and_log(a_input, v_input, audio_model, global_step)

        # Retrieve error counts at the end of the epoch
        audio_errors, image_errors = train_dataset.get_error_counts()
        print(f'Epoch {epoch}: Audio Loading Errors = {audio_errors}, Image Loading Errors = {image_errors}')
        # Reset counters after logging
        train_dataset.reset_error_counts()

        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(audio_model, test_loader, args)

        print("Eval Audio MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

        print("Train Audio MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))

        wandb.log({"Eval Total Loss": eval_loss_av, "Eval MAE Loss Audio": eval_loss_mae_a, "Eval MAE Loss Visual": eval_loss_mae_v, "Eval Contrastive Loss": eval_loss_c, "Eval Contrastive Acc": eval_c_acc}, step=global_step)

        # train audio mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval audio mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

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

def calculate_contrastive_accuracy(audio_rep, video_rep, video_ids):
    # print(f"Audio rep shape: {audio_rep.shape}, Video rep shape: {video_rep.shape}")
    # print(f"Number of video_ids: {len(video_ids)}")
    # print(f"Audio rep dtype: {audio_rep.dtype}, Video rep dtype: {video_rep.dtype}")

    if audio_rep.shape[0] != len(video_ids) or video_rep.shape[0] != len(video_ids):
        # print("Warning: Mismatch between number of representations and video IDs")
        min_len = min(audio_rep.shape[0], video_rep.shape[0], len(video_ids))
        audio_rep = audio_rep[:min_len]
        video_rep = video_rep[:min_len]
        video_ids = video_ids[:min_len]

    audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
    video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
    
    # Compute similarities
    similarities = torch.mm(audio_rep, video_rep.t())
    # print(f"Similarities shape: {similarities.shape}, dtype: {similarities.dtype}")
    
    # Create a mask for matching video IDs
    video_id_tensor = torch.tensor([hash(vid) for vid in video_ids], device=audio_rep.device)
    match_mask = (video_id_tensor.unsqueeze(0) == video_id_tensor.unsqueeze(1))
    
    # print(f"Match mask shape: {match_mask.shape}")
    
    if similarities.shape != match_mask.shape:
        # print("Warning: Mismatch between similarities and match_mask shapes")
        min_dim = min(similarities.shape[0], match_mask.shape[0])
        similarities = similarities[:min_dim, :min_dim]
        match_mask = match_mask[:min_dim, :min_dim]

    # Set similarities between non-matching video IDs to a large negative value
    min_similarity = torch.finfo(similarities.dtype).min
    similarities = torch.where(match_mask, similarities, torch.tensor(min_similarity, device=similarities.device, dtype=similarities.dtype))
    
    # Compute accuracy
    correct_matches = (torch.argmax(similarities, dim=1) == torch.arange(similarities.shape[0], device=similarities.device)).float()
    accuracy = correct_matches.mean().item()
    
    # print(f"Computed accuracy: {accuracy}")
    
    return accuracy

def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc = [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, labels, video_ids, frame_indices) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            
            # print(f"Batch {i}: a_input shape: {a_input.shape}, v_input shape: {v_input.shape}")
            # print(f"Number of video_ids: {len(video_ids)}")
            # print(f"Number of unique video_ids: {len(set(video_ids))}")
            # print(f"Frame indices shape: {frame_indices.shape}")
            
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, model_c_acc, recon_a, recon_v, latent_c_a, latent_c_v = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                
                # print(f"latent_c_a shape: {latent_c_a.shape}, latent_c_v shape: {latent_c_v.shape}")
                
                # Ensure latent_c_a and latent_c_v have the same first dimension as len(video_ids)
                min_len = min(latent_c_a.shape[0], latent_c_v.shape[0], len(video_ids))
                latent_c_a = latent_c_a[:min_len]
                latent_c_v = latent_c_v[:min_len]
                batch_video_ids = video_ids[:min_len]
                
                # Calculate our own contrastive accuracy for validation
                c_acc = calculate_contrastive_accuracy(latent_c_a.float(), latent_c_v.float(), batch_video_ids)
                
                loss = loss.mean()
                loss_mae = loss_mae.mean()
                loss_mae_a = loss_mae_a.mean()
                loss_mae_v = loss_mae_v.mean()
                loss_c = loss_c.mean()
            
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc)
            
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc