"""
Basic training metrics logging
"""

import torch
import wandb
import time
from typing import Dict

def log_training_metrics(model: torch.nn.Module,
                        loss: float,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        step: int):
    """Log training metrics to wandb"""
    wandb.log({
        'train/loss': loss,  # loss is already a float
        'train/learning_rate': optimizer.param_groups[0]['lr'],
        'train/epoch': epoch,
        'train/step': step
    }, step=step)

def log_validation_metrics(loss: float, epoch: int, step: int):
    """Log validation metrics"""
    wandb.log({
        'val/loss': loss,
        'val/epoch': epoch,
        'val/step': step
    }, step=step) 