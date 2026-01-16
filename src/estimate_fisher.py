# -*- coding: utf-8 -*-
# Estimate diagonal Fisher information for CAV-MAE checkpoints.
# Saves a state_dict-like mapping from parameter name to Fisher diagonal.

import argparse
import ast
import os
import sys
import time
from collections import OrderedDict

import torch
from torch.cuda.amp import autocast

basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

import dataloader as dataloader
import models


def build_loader(args):
    im_res = 224
    audio_conf = {
        'num_mel_bins': 128,
        'target_length': args.target_length,
        'freqm': 0,
        'timem': 0,
        'mixup': 0,
        'dataset': args.dataset,
        'mode': 'eval',
        'mean': args.dataset_mean,
        'std': args.dataset_std,
        'noise': False,
        'label_smooth': 0,
        'im_res': im_res,
    }
    dataset = dataloader.AudiosetDataset(args.data, label_csv=args.label_csv, audio_conf=audio_conf)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader


def load_cavmae(model_path, device, target_length, norm_pix_loss=None, tr_pos=None):
    model = models.CAVMAE(audio_length=target_length, norm_pix_loss=norm_pix_loss, modality_specific_depth=11, tr_pos=tr_pos)
    state = torch.load(model_path, map_location='cpu')
    use_dp = any(k.startswith("module.") for k in state.keys())
    if use_dp:
        model = torch.nn.DataParallel(model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()
    return model


def estimate_fisher(model, loader, args, device):
    torch.set_grad_enabled(True)
    fisher = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param, device='cpu')

    n_batches = 0
    start = time.time()
    for step, (a_input, v_input, _) in enumerate(loader):
        if step >= args.n_batches:
            break
        a_input = a_input.to(device, non_blocking=True)
        v_input = v_input.to(device, non_blocking=True)

        with autocast(enabled=args.use_amp):
            loss, _, _, _, _, _, _, _ = model(
                a_input,
                v_input,
                args.masking_ratio,
                args.masking_ratio,
                mae_loss_weight=args.mae_loss_weight,
                contrast_loss_weight=args.contrast_loss_weight,
                mask_mode=args.mask_mode,
            )
            loss = loss.sum()

        model.zero_grad(set_to_none=True)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.detach().cpu() ** 2)
        model.zero_grad(set_to_none=True)
        n_batches += 1

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - start
            print(f"Processed {step + 1} batches in {elapsed:.1f}s")

    denom = max(1, n_batches)
    for name in fisher.keys():
        fisher[name] = fisher[name] / denom

    print(f"Fisher estimation finished: {n_batches} batches")
    return fisher


def main():
    parser = argparse.ArgumentParser(
        description='Estimate diagonal Fisher for CAV-MAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, required=True, help="model checkpoint to estimate Fisher for")
    parser.add_argument("--output", type=str, required=True, help="output path for Fisher diagonal")

    parser.add_argument("--data", type=str, required=True, help="data json for Fisher estimation")
    parser.add_argument("--label-csv", type=str, required=True, help="label CSV path")
    parser.add_argument("--dataset", type=str, default="audioset",
                        choices=["audioset", "esc50", "speechcommands", "fsd50k", "vggsound", "epic", "k400", "msrvtt"])
    parser.add_argument("--dataset_mean", type=float, default=None, help="dataset audio spec mean")
    parser.add_argument("--dataset_std", type=float, default=None, help="dataset audio spec std")
    parser.add_argument("--target_length", type=int, default=1024, help="input length in frames")

    parser.add_argument('--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# workers for dataloading')
    parser.add_argument("--n-batches", type=int, default=100, help="number of batches to use")
    parser.add_argument("--log-every", type=int, default=25, help="log progress every N batches")

    parser.add_argument("--mae_loss_weight", type=float, default=1.0, help="weight for MAE loss")
    parser.add_argument("--contrast_loss_weight", type=float, default=0.0, help="weight for contrastive loss")
    parser.add_argument("--masking_ratio", type=float, default=0.75, help="masking ratio")
    parser.add_argument("--mask_mode", type=str, default='unstructured', choices=['unstructured', 'time', 'freq', 'tf'])
    parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
    parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default=None)
    parser.add_argument('--use_amp', help='if use AMP autocast', type=ast.literal_eval, default=False)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    loader = build_loader(args)
    model = load_cavmae(args.model, device, args.target_length, norm_pix_loss=args.norm_pix_loss, tr_pos=args.tr_pos)

    fisher = estimate_fisher(model, loader, args, device)

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    torch.save(fisher, args.output)
    print(f"Saved Fisher diagonal to: {args.output}")


if __name__ == '__main__':
    main()
