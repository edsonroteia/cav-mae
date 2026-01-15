# -*- coding: utf-8 -*-
# @Time    : 1/15/26
# @Author  : Based on traintest_cavmae.py by Yuan Gong (MIT)
# @File    : traintest_cavjepa.py
# @Description: Training loop for CAV-JEPA with EMA updates

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
from torch.cuda.amp import autocast, GradScaler


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on ' + str(device))
    torch.set_grad_enabled(True)

    # Meters for tracking
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    loss_total_meter = AverageMeter()
    loss_jepa_meter = AverageMeter()
    loss_jepa_a_meter = AverageMeter()
    loss_jepa_v_meter = AverageMeter()
    loss_c_meter = AverageMeter()
    c_acc_meter = AverageMeter()

    progress = []
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    # Calculate total steps for momentum schedule
    total_steps = args.n_epochs * len(train_loader)
    print(f'Total training steps: {total_steps}')
    print(f'Momentum schedule: {args.momentum_start} -> {args.momentum_end}')

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_loss,
                         time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    def get_momentum(step):
        """Linear momentum schedule from start to end"""
        return args.momentum_start + (args.momentum_end - args.momentum_start) * (step / total_steps)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # Only train context encoder and predictor, not target encoder
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(
        sum(p.numel() for p in trainables) / 1e6))

    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # Learning rate scheduler
    if args.lr_adapt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Using adaptive learning rate scheduler.')
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
            gamma=args.lrscheduler_decay)
        print('Learning rate scheduler starts at {:d} epoch with decay rate {:.3f} every {:d} epochs'.format(
            args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('Training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("Current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("Start training...")

    # Results: [jepa_a, jepa_v, contrastive, total, eval_jepa_a, eval_jepa_v, eval_c, eval_total, eval_c_acc, lr]
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()

    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()

        print('---------------')
        print(datetime.datetime.now())
        print("Current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('Masking ratio: {:.3f} for both modalities'.format(args.masking_ratio))

        for i, (a_input, v_input, _) in enumerate(train_loader):
            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            # Update momentum before forward pass
            current_momentum = get_momentum(global_step)
            if hasattr(audio_model, 'module'):
                audio_model.module.update_momentum(current_momentum)
            else:
                audio_model.update_momentum(current_momentum)

            with autocast():
                loss, loss_jepa, loss_jepa_a, loss_jepa_v, loss_c, mask_a, mask_v, c_acc = audio_model(
                    a_input, v_input,
                    args.masking_ratio, args.masking_ratio,
                    jepa_loss_weight=args.jepa_loss_weight,
                    contrast_loss_weight=args.contrast_loss_weight
                )
                # For DataParallel: manually average losses across GPUs
                loss = loss.sum()
                loss_jepa = loss_jepa.sum()
                loss_jepa_a = loss_jepa_a.sum()
                loss_jepa_v = loss_jepa_v.sum()
                loss_c = loss_c.sum()
                c_acc = c_acc.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update of target encoder (after optimizer step)
            if hasattr(audio_model, 'module'):
                audio_model.module.update_target_encoder()
            else:
                audio_model.update_target_encoder()

            # Update meters
            loss_total_meter.update(loss.item(), B)
            loss_jepa_meter.update(loss_jepa.item(), B)
            loss_jepa_a_meter.update(loss_jepa_a.item(), B)
            loss_jepa_v_meter.update(loss_jepa_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            c_acc_meter.update(c_acc.item(), B)

            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time) / a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps / 10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Per Sample Time {per_sample_time.avg:.5f}\t'
                      'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                      'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                      'Train Total Loss {loss_total_meter.val:.4f}\t'
                      'Train JEPA Loss Audio {loss_jepa_a_meter.val:.4f}\t'
                      'Train JEPA Loss Visual {loss_jepa_v_meter.val:.4f}\t'
                      'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                      'Train Contrastive Acc {c_acc:.3f}\t'
                      'Momentum {momentum:.6f}'.format(
                    epoch, i, len(train_loader),
                    per_sample_time=per_sample_time,
                    per_sample_data_time=per_sample_data_time,
                    per_sample_dnn_time=per_sample_dnn_time,
                    loss_total_meter=loss_total_meter,
                    loss_jepa_a_meter=loss_jepa_a_meter,
                    loss_jepa_v_meter=loss_jepa_v_meter,
                    loss_c_meter=loss_c_meter,
                    c_acc=c_acc,
                    momentum=current_momentum
                ), flush=True)

                if np.isnan(loss_total_meter.avg):
                    print("Training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        # Validation
        print('Start validation')
        eval_loss, eval_loss_jepa, eval_loss_jepa_a, eval_loss_jepa_v, eval_loss_c, eval_c_acc = validate(
            audio_model, test_loader, args)

        print("Eval JEPA Loss Audio: {:.6f}".format(eval_loss_jepa_a))
        print("Eval JEPA Loss Visual: {:.6f}".format(eval_loss_jepa_v))
        print("Eval JEPA Loss Total: {:.6f}".format(eval_loss_jepa))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Total Loss: {:.6f}".format(eval_loss))
        print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))

        print("Train JEPA Loss Audio: {:.6f}".format(loss_jepa_a_meter.avg))
        print("Train JEPA Loss Visual: {:.6f}".format(loss_jepa_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_total_meter.avg))

        # Save results
        result[epoch - 1, :] = [
            loss_jepa_a_meter.avg, loss_jepa_v_meter.avg, loss_c_meter.avg, loss_total_meter.avg,
            eval_loss_jepa_a, eval_loss_jepa_v, eval_loss_c, eval_loss, eval_c_acc,
            optimizer.param_groups[0]['lr']
        ]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('Validation finished')

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % exp_dir)
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % exp_dir)

        if args.save_model:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('Epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        # Reset meters
        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_total_meter.reset()
        loss_jepa_meter.reset()
        loss_jepa_a_meter.reset()
        loss_jepa_v_meter.reset()
        loss_c_meter.reset()
        c_acc_meter.reset()


def validate(audio_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_loss, A_loss_jepa, A_loss_jepa_a, A_loss_jepa_v, A_loss_c, A_c_acc = [], [], [], [], [], []

    with torch.no_grad():
        for i, (a_input, v_input, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)

            with autocast():
                loss, loss_jepa, loss_jepa_a, loss_jepa_v, loss_c, mask_a, mask_v, c_acc = audio_model(
                    a_input, v_input,
                    args.masking_ratio, args.masking_ratio,
                    jepa_loss_weight=args.jepa_loss_weight,
                    contrast_loss_weight=args.contrast_loss_weight
                )
                loss = loss.sum()
                loss_jepa = loss_jepa.sum()
                loss_jepa_a = loss_jepa_a.sum()
                loss_jepa_v = loss_jepa_v.sum()
                loss_c = loss_c.sum()
                c_acc = c_acc.mean()

            A_loss.append(loss.to('cpu').detach())
            A_loss_jepa.append(loss_jepa.to('cpu').detach())
            A_loss_jepa_a.append(loss_jepa_a.to('cpu').detach())
            A_loss_jepa_v.append(loss_jepa_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_jepa = np.mean(A_loss_jepa)
        loss_jepa_a = np.mean(A_loss_jepa_a)
        loss_jepa_v = np.mean(A_loss_jepa_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)

    return loss, loss_jepa, loss_jepa_a, loss_jepa_v, loss_c, c_acc
