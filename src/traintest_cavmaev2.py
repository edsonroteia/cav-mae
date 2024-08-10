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
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

import os
import matplotlib.pyplot as plt
import torch
import wandb

import torch
import matplotlib.pyplot as plt
import os

def visualize_and_log(a_input, v_input, audio_model, step):
    print("Logging examples to wandb...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_input, v_input = a_input.to(device), v_input.to(device)
    
    # Get reconstructions
    with torch.no_grad():
        _, _, _, _, _, mask_a, mask_v, _, recon_a, recon_v = audio_model(a_input, v_input)

    num_frames = v_input.shape[2]
    # Create subplots: one row for audio, and one row per frame for video
    fig, axes = plt.subplots(2 + num_frames, 2, figsize=(12, 2 + 3 * num_frames))

    # Process audio masks
    mask_a = mask_a.detach()
    mask_a = mask_a.unsqueeze(-1).repeat(1, 1, audio_model.module.patch_embed_a.patch_size[0] ** 2 * 1)  # (N, H*W, p*p*1)
    mask_a = audio_model.module.unpatchify(mask_a, 1, 1, 8, 64, 16)  # 1 is removing, 0 is keeping
    mask_a = torch.einsum('ncthw->nthwc', mask_a).detach().cpu()
    mask_a = mask_a[0].squeeze().numpy()

    # # Process video masks
    # mask_v = mask_v.detach()
    # mask_v = mask_v.unsqueeze(-1).repeat(1, 1, audio_model.module.patch_embed_v.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    # mask_v = audio_model.module.unpatchify(mask_v, 3, 8, 14, 14, 16)  # 1 is removing, 0 is keeping

    # Process audio visualizations
    a_input_np = a_input[0].cpu().numpy().transpose(1, 0)
    recon_a_np = recon_a[0].squeeze().cpu().numpy()
    axes[0, 0].imshow(a_input_np, aspect='equal', origin='lower')
    axes[0, 0].set_title("Original Audio")
    axes[0, 0].axis('off')

    recon_a_np = a_input_np * (1 - mask_a) + recon_a_np * mask_a[0]

    axes[0, 1].imshow(recon_a_np, aspect='equal', origin='lower')
    axes[0, 1].set_title("Reconstructed Audio")
    axes[0, 1].axis('off')


    # Process each video frame and its reconstruction
    for frame_idx in range(num_frames):
        v_input_np = v_input.cpu().numpy().transpose(0,2,3,4,1)[0, frame_idx]
        recon_v_np = recon_v.cpu().numpy().transpose(0,4,2,3,1)[0, frame_idx]

        # Normalize for better visualization if necessary
        if v_input_np.max() > 1 or v_input_np.min() < 0:
            v_input_np = (v_input_np - v_input_np.min()) / (v_input_np.max() - v_input_np.min())
            recon_v_np = (recon_v_np - recon_v_np.min()) / (recon_v_np.max() - recon_v_np.min())

        # recon_v_np = v_input_np * (1 - mask_v) + recon_v_np * mask_v[frame_idx]

        # Plot original and reconstructed image frames
        axes[1 + frame_idx, 0].imshow(v_input_np)
        axes[1 + frame_idx, 0].set_title(f"Original Image Frame {frame_idx+1}")
        axes[1 + frame_idx, 0].axis('off')

        axes[1 + frame_idx, 1].imshow(recon_v_np)
        axes[1 + frame_idx, 1].set_title(f"Reconstructed Image Frame {frame_idx+1}")
        axes[1 + frame_idx, 1].axis('off')

    # Save the figure and log it to W&B
    fig_path = os.path.join("visualizations", f"step_{step}.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close(fig)

    wandb.log({"Visualizations": wandb.Image(fig_path)}, step=step)

class LinearProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbingClassifier, self).__init__()
        self.mlp = self._build_mlp(num_layers=2, input_dim=input_dim, mlp_dim=512, output_dim=num_classes)
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))
            nn.init.kaiming_normal_(mlp[-1].weight)  # Initialize weights

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
                mlp.append(nn.Dropout(p=0.5))  # Add dropout for regularization
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)


def linear_probing(audio_model, classifier, train_loader, test_loader, device, args):
    # Freeze the feature extractor
    for param in audio_model.parameters():
        param.requires_grad = False

    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training the classifier head
    for epoch in tqdm(range(1)):
        for a_input, v_input, labels in train_loader:
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            # Get the features from the audio model
            _, _, _, _, _, ca, cv = audio_model.module.forward_encoder(a_input, v_input, args.masking_ratio, args.masking_ratio, mask_mode=args.mask_mode)
            features = torch.cat((ca, cv), dim=1)
            features = features.mean(dim=1)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # import pdb; pdb.set_trace()
    # Evaluation
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for a_input, v_input, labels in test_loader:
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            labels = labels.to(device)

            _, _, _, _, _, ca, cv = audio_model.module.forward_encoder(a_input, v_input, args.masking_ratio, args.masking_ratio, mask_mode=args.mask_mode)
            features = torch.cat((ca, cv), dim=1)
            features = features.mean(dim=1)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            # Convert one-hot encoded labels to class indices
            labels = torch.argmax(labels, dim=1)
            print(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    estimated_finish_time = 0  # Initialize estimated finish time

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

    classifier = LinearProbingClassifier(input_dim=768, num_classes=args.n_class).to(device)

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])  # for each epoch, 10 metrics to record
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        unfreeze_model(audio_model)
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
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, _, _, c_acc, _, _ = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()
            # Debugging statement to check if loss requires gradients
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

            if global_step % 1 == 2900 and args.mae_loss_weight != 0:
                visualize_and_log(a_input, v_input, audio_model, global_step)

        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc = validate(audio_model, train_loader, test_loader, epoch, classifier, args)

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

        # if args.save_model == True:
        #     torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / epoch * args.n_epochs
        estimated_finish_time = start_time + estimated_total_time
        remaining_time = estimated_finish_time - time.time()
        # Convert remaining_time from seconds to a readable format (e.g., hours, minutes, seconds)
        remaining_hrs, remaining_min = divmod(remaining_time, 3600)
        remaining_min, remaining_sec = divmod(remaining_min, 60)
        # Print the estimated finish time and remaining time
        print(f'Current time: {time.ctime()}')
        print(f'Estimated finish time: {time.ctime(estimated_finish_time)} (in {int(remaining_hrs)}h {int(remaining_min)}m {int(remaining_sec)}s)')

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

def validate(audio_model, train_loader, test_loader, epoch, classifier, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model.eval()
    loss_av_meter = AverageMeter()
    loss_mae_meter = AverageMeter()
    loss_mae_a_meter = AverageMeter()
    loss_mae_v_meter = AverageMeter()
    loss_c_meter = AverageMeter()
    c_acc_meter = AverageMeter()

    with torch.no_grad():
        for a_input, v_input, labels in test_loader:
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, _, _, c_acc, _, _ = audio_model(a_input, v_input, args.masking_ratio, args.masking_ratio, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.mask_mode)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean()

            loss_av_meter.update(loss.item(), a_input.size(0))
            loss_mae_meter.update(loss_mae.item(), a_input.size(0))
            loss_mae_a_meter.update(loss_mae_a.item(), a_input.size(0))
            loss_mae_v_meter.update(loss_mae_v.item(), a_input.size(0))
            loss_c_meter.update(loss_c.item(), a_input.size(0))
            c_acc_meter.update(c_acc.item(), a_input.size(0))

    # Perform linear probing every 10 epochs
    # print("Starting Linear Probing")
    # if epoch % 1 == 0:
    #     probing_accuracy = linear_probing(audio_model, classifier, train_loader, test_loader, device, args)
    #     print(f'Linear Probing Accuracy at Epoch {epoch}: {probing_accuracy:.4f}')
    #     wandb.log({"Linear Probing Accuracy": probing_accuracy}, step=epoch)

    return loss_av_meter.avg, loss_mae_meter.avg, loss_mae_a_meter.avg, loss_mae_v_meter.avg, loss_c_meter.avg, c_acc_meter.avg
