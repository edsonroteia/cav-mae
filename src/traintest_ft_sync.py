# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from cosine_scheduler import CosineWarmupScheduler

def log_plot_to_neptune(run, plot_name, fig, step):
    """Log a matplotlib figure to Neptune without saving to disk."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image = Image.open(buffer)
    run[f"visualizations/{plot_name}"].log(neptune.types.File.as_image(image), step=step)
    plt.close(fig)

def visualize_confusion_matrix(y_true, y_pred, class_names, step, run):
    print("Visualizing Confusion Matrix...")
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    
    # Get the number of classes from class_names
    n_classes = len(class_names)
    
    # Compute confusion matrix with explicit label ordering
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    fig, ax = plt.subplots(figsize=(20, 20))  # Increased figure size
    sns.heatmap(cm, annot=False, cmap='Blues', square=True, ax=ax)  # Removed annot for better visibility
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    
    # Set ticks to show every 10th class
    tick_positions = np.arange(0, n_classes, 50)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([class_names[i] for i in tick_positions], rotation=90)
    ax.set_yticklabels([class_names[i] for i in tick_positions], rotation=0)
    
    log_plot_to_neptune(run, "confusion_matrix", fig, step)

def visualize_roc_curve(y_true, y_score, step, run):
    print("Visualizing ROC Curve...")
    print(f"y_true shape: {y_true.shape}, y_score shape: {y_score.shape}")
    n_classes = y_true.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots()
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (AUC = {roc_auc[i]:0.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    log_plot_to_neptune(run, "roc_curve", fig, step)

def visualize_class_distribution(y_true, step, run):
    print("Visualizing Class Distribution...")
    print(f"y_true shape: {y_true.shape}")
    class_counts = np.sum(y_true, axis=0)
    fig, ax = plt.subplots()
    ax.bar(range(len(class_counts)), class_counts)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    log_plot_to_neptune(run, "class_distribution", fig, step)

def train(audio_model, train_loader, test_loader, args, run):
    params = vars(args)
    # Log parameters at the beginning of the run
    run["parameters"] = params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, loss_meter, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    progress = []
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)

    # possible mlp layer name list, mlp layers are newly initialized layers in the finetuning stage (i.e., not pretrained) and should use a larger lr during finetuning
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'mlp_head.3.weight', 'mlp_head.3.bias', 'mlp_head.5.weight', 'mlp_head.5.bias',
                'mlp_head.7.weight', 'mlp_head.7.bias']
    concat_mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                       'mlp_head.3.weight', 'mlp_head.3.bias', 'mlp_head.5.weight', 'mlp_head.5.bias',
                       'mlp_head.7.weight', 'mlp_head.7.bias']
    transformer_cls_list = ['cls_token', 'classifier_layers.0.weight', 'classifier_layers.0.bias',
                            'classifier_layers.1.weight', 'classifier_layers.1.bias',
                            'classifier_norm.weight', 'classifier_norm.bias',
                            'classifier_head.weight', 'classifier_head.bias']

    if args.aggregate == 'concat_mlp':
        cls_params = list(filter(lambda kv: kv[0] in concat_mlp_list, audio_model.module.named_parameters()))
    elif args.aggregate == 'self_attention_cls':
        cls_params = list(filter(lambda kv: kv[0] in transformer_cls_list, audio_model.module.named_parameters()))
    else:
        cls_params = list(filter(lambda kv: kv[0] in mlp_list, audio_model.module.named_parameters()))

    base_params = list(filter(lambda kv: kv[0] not in (mlp_list + concat_mlp_list + transformer_cls_list), audio_model.module.named_parameters()))
    cls_params = [i[1] for i in cls_params]
    base_params = [i[1] for i in base_params]

    # if freeze the pretrained parameters and only train the newly initialized model (linear probing)
    if args.freeze_base == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    print('The newly initialized mlp layer uses {:.3f} x larger lr'.format(args.head_lr))
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': cls_params, 'lr': args.lr * args.head_lr}], weight_decay=5e-7, betas=(0.95, 0.999))
    base_lr = optimizer.param_groups[0]['lr']
    mlp_lr = optimizer.param_groups[1]['lr']
    lr_list = [args.lr, mlp_lr]
    print('base lr, mlp lr : ', base_lr, mlp_lr)

    print('Total newly initialized MLP parameter number is : {:.3f} million'.format(sum(p.numel() for p in cls_params) / 1e6))
    print('Total pretrained backbone parameter number is : {:.3f} million'.format(sum(p.numel() for p in base_params) / 1e6))

    # Configure learning rate scheduler
    if args.lr_adapt:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Using adaptive learning rate scheduler with patience {:d}'.format(args.lr_patience))
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
        print('Using cosine annealing learning rate scheduler over {:d} epochs with minimum lr of {:d}'.format(args.n_epochs, args.lr * 0.1))
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.n_epochs*0.5), int(args.n_epochs*0.75)], gamma=0.1)
        print('Using step learning rate scheduler with milestones at 50% and 75% of training, decay rate of 0.1')
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 4])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (a_input, v_input, labels, _, _) in tqdm(enumerate(train_loader)):
                
            B = a_input.size(0)
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / B)
            dnn_start_time = time.time()

            with autocast():
                # print(a_input.shape, v_input.shape)
                audio_output = audio_model(a_input, v_input, args.ftmode)
                if args.aggregate != 'None':
                    # Reshape labels to match the new shape of audio_output
                    labels = labels.view(B//10, 10, -1)[:, 0, :]  # Take the first frame's label for each video
                loss = loss_fn(audio_output, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.lr_scheduler == 'cosine':
                scheduler.step()

            # Use a combination of epoch and batch index for unique step values
            current_step = (epoch - 1) * len(train_loader) + i

            run["train/loss"].log(loss.item(), step=current_step)
            run["train/batch_time"].log(time.time() - end_time, step=current_step)
            run["train/learning_rate"].log(optimizer.param_groups[0]["lr"], step=current_step)

            if args.aggregate != 'None':
                B = B // 10  # Adjust B to reflect the number of videos, not frames
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/B)
            per_sample_dnn_time.update((time.time() - dnn_start_time)/B)

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')

        stats, valid_loss, A_predictions, A_targets = validate(audio_model, test_loader, args, output_pred=True)

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc'] # this is just a trick, acc of each class entry is the same, which is the accuracy of all classes, not class-wise accuracy

        print("mAP: {:.6f}".format(mAP))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        # Log important metrics
        run["valid/mAP"].log(mAP, step=epoch)
        run["valid/AUC"].log(mAUC, step=epoch)
        run["valid/d_prime"].log(d_prime(mAUC), step=epoch)
        run["train/epoch_loss"].log(loss_meter.avg, step=epoch)
        run["valid/loss"].log(valid_loss, step=epoch)

        # # Visualizations
        # try:
        #     print("Starting visualizations...")
        #     print(f"Stats keys: {stats[0].keys()}")  # Print available keys in stats

        #     # Use A_predictions and A_targets instead of stats for raw data
        #     y_pred = A_predictions.cpu().numpy()
        #     y_true = A_targets.cpu().numpy()
            
        #     print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
            
        #     y_pred_classes = np.argmax(y_pred, axis=1)
        #     y_true_classes = np.argmax(y_true, axis=1)

        #     class_names = [f'Class {i}' for i in range(args.n_class)]

        #     # Confusion Matrix
        #     visualize_confusion_matrix(y_true_classes, y_pred_classes, class_names, epoch, run)

        #     # ROC Curve
        #     visualize_roc_curve(y_true, y_pred, epoch, run)

        #     # Class Distribution
        #     visualize_class_distribution(y_true, epoch, run)

        #     # # Log additional metrics from stats
        #     # for i, stat in enumerate(stats):
        #     #     run[f"valid/class_{i}_AP"].log(stat["AP"], step=epoch)
        #     #     run[f"valid/class_{i}_AUC"].log(stat["auc"], step=epoch)

        # except Exception as e:
        #     print(f"Error in visualization: {str(e)}")
        #     print(f"Stats structure: {stats[0]}")  # Print the structure of the first stats entry
        #     import traceback
        #     traceback.print_exc()  # Print the full traceback for more detailed error information

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if args.lr_scheduler != 'cosine':
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if main_metrics == 'mAP':
                    scheduler.step(mAP)
                elif main_metrics == 'acc':
                    scheduler.step(acc)
            else:
                scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        run["train/epoch_time"].log(finish_time - begin_time, step=epoch)
        run["valid/epoch_mAP"].log(mAP, step=epoch)
        run["valid/epoch_acc"].log(acc, step=epoch)
        run["valid/epoch_auc"].log(mAUC, step=epoch)

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    run.stop()

def validate(audio_model, val_loader, args, output_pred=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions, A_targets, A_loss = [], [], []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation"):
            if batch is None:
                print(f"Skipping empty batch {i}")
                continue
            a_input, v_input, labels, _, _ = batch
            
            a_input, v_input = a_input.to(device, non_blocking=True), v_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # print(a_input.shape, v_input.shape)
            with autocast():
                audio_output = audio_model(a_input, v_input, args.ftmode)
                
            if args.aggregate != 'None':
                # Reshape labels to match the new shape of audio_output
                B = a_input.size(0)
                labels = labels.view(B//10, 10, -1)[:, 0, :]  # Take the first frame's label for each video
            
            loss = args.loss_fn(audio_output, labels)
            
            A_predictions.append(audio_output.to('cpu').detach())
            A_targets.append(labels.to('cpu').detach())
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = torch.stack(A_loss).mean().item()

        stats = calculate_stats(audio_output, target)

    if output_pred == False:
        return stats, loss
    else:
        # used for multi-frame evaluation (i.e., ensemble over frames), so return prediction and target
        return stats, loss, audio_output, target
