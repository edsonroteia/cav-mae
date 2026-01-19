# -*- coding: utf-8 -*-
# @Time    : 1/19/26
# @Author  : Adapted from I-JEPA (Meta)
# @Description: Learning rate and weight decay schedulers for CAV-JEPA training

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineSchedule(_LRScheduler):
    """
    Warmup + Cosine Annealing learning rate schedule.

    Following I-JEPA training protocol:
    - Linear warmup: start_lr -> ref_lr over warmup_steps
    - Cosine decay: ref_lr -> final_lr over remaining steps

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: number of warmup steps
        start_lr: initial learning rate during warmup
        ref_lr: peak learning rate after warmup
        total_steps: total training steps
        final_lr: final learning rate at end of training
        last_epoch: last epoch number for resuming
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        total_steps,
        final_lr=1e-6,
        last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.total_steps = total_steps
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            progress = step / max(1, self.warmup_steps)
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.base_lrs]


class CosineWDSchedule:
    """
    Cosine weight decay schedule.

    Following I-JEPA training protocol:
    - Weight decay increases from start_wd to final_wd over training
    - Uses cosine schedule for smooth transition

    Args:
        start_wd: initial weight decay
        final_wd: final weight decay
        total_steps: total training steps
    """

    def __init__(self, start_wd, final_wd, total_steps):
        self.start_wd = start_wd
        self.final_wd = final_wd
        self.total_steps = total_steps

    def __call__(self, step):
        """Get weight decay for current step."""
        if self.total_steps <= 0:
            return self.final_wd

        progress = min(step / self.total_steps, 1.0)
        # Cosine schedule from start to final
        wd = self.start_wd + 0.5 * (self.final_wd - self.start_wd) * (1 - math.cos(math.pi * progress))
        return wd


class LinearMomentumSchedule:
    """
    Linear momentum schedule for EMA target encoder.

    Following I-JEPA/BYOL training protocol:
    - Momentum increases linearly from start to end over training
    - Higher momentum at end provides more stable targets

    Args:
        start_momentum: initial EMA momentum (e.g., 0.996)
        end_momentum: final EMA momentum (e.g., 0.999)
        total_steps: total training steps
    """

    def __init__(self, start_momentum, end_momentum, total_steps):
        self.start_momentum = start_momentum
        self.end_momentum = end_momentum
        self.total_steps = total_steps

    def __call__(self, step):
        """Get momentum for current step."""
        if self.total_steps <= 0:
            return self.end_momentum

        progress = min(step / self.total_steps, 1.0)
        momentum = self.start_momentum + progress * (self.end_momentum - self.start_momentum)
        return momentum


def get_param_groups_with_wd_exclusion(model, weight_decay, no_wd_keywords=None):
    """
    Create parameter groups with weight decay exclusion.

    Excludes certain parameters from weight decay (as per I-JEPA):
    - Bias terms
    - LayerNorm parameters
    - Positional embeddings
    - Any parameters matching specified keywords

    Args:
        model: PyTorch model
        weight_decay: base weight decay value
        no_wd_keywords: list of keywords to exclude from weight decay

    Returns:
        list of param groups for optimizer
    """
    if no_wd_keywords is None:
        no_wd_keywords = ['bias', 'ln', 'norm', 'pos_embed', 'modality_']

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should be excluded from weight decay
        exclude = False
        for keyword in no_wd_keywords:
            if keyword in name.lower():
                exclude = True
                break

        # Also exclude 1D parameters (biases, LayerNorm weights)
        if param.ndim <= 1:
            exclude = True

        if exclude:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def update_weight_decay(optimizer, wd_schedule, step):
    """
    Update optimizer weight decay based on schedule.

    Args:
        optimizer: PyTorch optimizer
        wd_schedule: CosineWDSchedule instance
        step: current training step
    """
    new_wd = wd_schedule(step)
    for param_group in optimizer.param_groups:
        # Only update groups that had non-zero weight decay initially
        if param_group.get('weight_decay', 0) > 0:
            param_group['weight_decay'] = new_wd
    return new_wd


class CombinedScheduler:
    """
    Combined scheduler that manages LR, WD, and momentum together.

    Convenience class for managing all three schedules in training loop.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: number of warmup epochs
        total_epochs: total training epochs
        steps_per_epoch: training steps per epoch
        start_lr: initial LR during warmup
        ref_lr: peak LR after warmup
        final_lr: final LR at end
        start_wd: initial weight decay
        final_wd: final weight decay
        start_momentum: initial EMA momentum
        end_momentum: final EMA momentum
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        steps_per_epoch,
        start_lr=1e-4,
        ref_lr=1e-3,
        final_lr=1e-6,
        start_wd=0.04,
        final_wd=0.4,
        start_momentum=0.996,
        end_momentum=0.999
    ):
        warmup_steps = warmup_epochs * steps_per_epoch
        total_steps = total_epochs * steps_per_epoch

        self.lr_scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            ref_lr=ref_lr,
            total_steps=total_steps,
            final_lr=final_lr
        )

        self.wd_schedule = CosineWDSchedule(
            start_wd=start_wd,
            final_wd=final_wd,
            total_steps=total_steps
        )

        self.momentum_schedule = LinearMomentumSchedule(
            start_momentum=start_momentum,
            end_momentum=end_momentum,
            total_steps=total_steps
        )

        self.optimizer = optimizer
        self.current_step = 0

    def step(self):
        """Advance all schedules by one step."""
        self.lr_scheduler.step()
        self.current_step += 1

    def get_momentum(self):
        """Get current EMA momentum."""
        return self.momentum_schedule(self.current_step)

    def get_wd(self):
        """Get current weight decay."""
        return self.wd_schedule(self.current_step)

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def update_wd(self):
        """Update optimizer weight decay."""
        return update_weight_decay(self.optimizer, self.wd_schedule, self.current_step)

    def state_dict(self):
        """Get scheduler state for checkpointing."""
        return {
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.current_step = state_dict['current_step']
