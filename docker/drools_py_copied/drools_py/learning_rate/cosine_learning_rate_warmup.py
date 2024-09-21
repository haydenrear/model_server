import numpy as np
from torch import optim

from python_util.logger.logger import LoggerFacade


class CosineWarmupScheduler(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.last_epoch, self.epoch = 0, 0
        self.warmup = warmup
        self.max_num_iters = max_iters
        LoggerFacade.info("Initialized cosine warmup scheduler.")
        super().__init__(optimizer)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.last_epoch = epoch
        LoggerFacade.info(f"Initialized epoch: {self.epoch} and {self.last_epoch}.")

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.epoch if self.epoch else self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
