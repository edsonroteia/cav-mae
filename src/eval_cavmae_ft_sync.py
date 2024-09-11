import argparse
import os
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
import dataloader_sync as dataloader
import models
import numpy as np
import json
from sklearn import metrics
from traintest_ft_sync import validate

# Load the pre-trained model
def load_pretrained_model(args):
    if args.model == 'cav-mae-ft':
        print('Loading a CAV-MAE model with 11 modality-specific layers and 1 modality-sharing layer')
        audio_model = models.CAVMAEFTSync(audio_length=args.target_length, label_dim=args.n_class, modality_specific_depth=11)
    else:
        raise ValueError('Model not supported')

    # Load the model weights
    sdA = torch.load(args.weight_file, map_location='cpu')
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sdA, strict=True)
    audio_model.eval()
    return audio_model

# Evaluate the model using multi-frame evaluation
def multi_frame_evaluate(audio_model, val_loader, args):
    res = []
    multiframe_pred = []
    total_frames = 10  # Change if your total frame is different
    for frame in range(total_frames):
        val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                          'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': 224, 'num_samples': args.num_samples,
                          'frame_use': frame}
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        stats, audio_output, target = validate(audio_model, val_loader, args, output_pred=True)
        print(audio_output.shape)
        if args.metrics == 'acc':
            audio_output = torch.nn.functional.softmax(audio_output.float(), dim=-1)
        elif args.metrics == 'mAP':
            audio_output = torch.nn.functional.sigmoid(audio_output.float())

        audio_output, target = audio_output.numpy(), target.numpy()
        multiframe_pred.append(audio_output)
        if args.metrics == 'mAP':
            cur_res = np.mean([stat['AP'] for stat in stats])
            print('mAP of frame {0:d} is {1:.4f}'.format(frame, cur_res))
        elif args.metrics == 'acc':
            cur_res = stats[0]['acc']
            print('acc of frame {0:d} is {1:.4f}'.format(frame, cur_res))
        res.append(cur_res)

    # Ensemble over frames
    multiframe_pred = np.mean(multiframe_pred, axis=0)
    if args.metrics == 'acc':
        acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(multiframe_pred, 1))
        print('multi-frame acc is {0:f}'.format(acc))
        res.append(acc)
    elif args.metrics == 'mAP':
        AP = []
        for k in range(args.n_class):
            # Average precision
            avg_precision = metrics.average_precision_score(target[:, k], multiframe_pred[:, k], average=None)
            AP.append(avg_precision)
        mAP = np.mean(AP)
        print('multi-frame mAP is {0:.4f}'.format(mAP))
        res.append(mAP)
    np.savetxt(args.exp_dir + '/mul_frame_res.csv', res, delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-val", type=str, default='', help="validation data json")
    parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
    parser.add_argument("--n_class", type=int, default=527, help="number of classes")
    parser.add_argument("--model", type=str, default='cav-mae-ft', help="the model used")
    parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400"])
    parser.add_argument("--dataset_mean", type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", type=float, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", type=int, help="the input length in frames")
    parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
    parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
    parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
    parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (default: use all samples)")
    parser.add_argument("--ftmode", type=str, default="multimodal", help="finetuning mode")
    
    # New arguments
    parser.add_argument("--loss", type=str, default="CE", help="loss function", choices=["CE", "BCE"])
    parser.add_argument("--freqm", type=int, default=0, help="frequency mask max length")
    parser.add_argument("--timem", type=int, default=0, help="time mask max length")
    parser.add_argument("--noise", type=bool, default=False, help="if add noise in data augmentation")
    parser.add_argument("--freeze_base", type=bool, default=False, help="if freeze the base model")
    parser.add_argument("--head_lr", type=float, default=1.0, help="learning rate multiplier for head")

    args = parser.parse_args()

    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                      'mode': 'eval', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False, 'im_res': 224, 'num_samples': args.num_samples}
    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = load_pretrained_model(args)
    
    # Set up loss function
    if args.loss == 'BCE':
        args.loss_fn = torch.nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        args.loss_fn = torch.nn.CrossEntropyLoss()
    
    multi_frame_evaluate(audio_model, val_loader, args)