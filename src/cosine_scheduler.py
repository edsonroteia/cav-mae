import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.max_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.base_lrs]
        elif self.last_epoch > self.max_epochs:
            return [self.min_lr for _ in self.base_lrs]
        else:
            decay_ratio = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return [self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.base_lrs]