import torch
import math
from torch.optim.lr_scheduler import _param_groups_val_list
from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmupScheduler(LRScheduler):
    def __init__(
        self, optimizer, warmup, T_max, eta_min: float = 0.0, last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch + 1
        # warmup
        if self.last_epoch <= self.warmup:
            return [base_lr * T / self.warmup for base_lr in self.base_lrs]
        # cosine annealing
        else:
            decay_T = self.T_max - self.warmup
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (T - self.warmup) / decay_T))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
