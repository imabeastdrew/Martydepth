"""
Basic training metrics logging
"""

import torch
import wandb
import time
from typing import Dict

class MetricsTracker:
    def __init__(self):
        self.epoch_start_time = None
        
    def start_epoch(self):
        """Mark start of epoch"""
        self.epoch_start_time = time.time()
    
    def end_epoch(self, epoch: int):
        """Calculate epoch time"""
        epoch_time = time.time() - self.epoch_start_time
        wandb.log({
            'time/epoch_minutes': epoch_time / 60,
            'time/epoch': epoch,
        })

def log_training_metrics(model: torch.nn.Module,
                        loss: torch.Tensor,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        step: int):
    """Log training metrics to wandb"""
    # Get learning rate
    lr = optimizer.param_groups[0]['lr']
    
    # Calculate gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # Calculate parameter norm
    param_norm = 0.0
    for p in model.parameters():
        param_norm += p.detach().data.norm(2).item() ** 2
    param_norm = param_norm ** 0.5
    
    # Log metrics
    metrics = {
        'train/loss': loss.item(),
        'train/learning_rate': lr,
        'train/epoch': epoch,
        'train/grad_norm': total_norm,
        'train/param_norm': param_norm,
    }
    
    # Add basic GPU memory metrics if available
    if torch.cuda.is_available():
        metrics.update({
            'system/gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,  # GB
            'system/gpu_memory_reserved': torch.cuda.max_memory_reserved() / 1e9,  # GB
        })
    
    wandb.log(metrics, step=step)

def log_validation_metrics(loss: float, epoch: int, step: int):
    """Log validation metrics"""
    wandb.log({
        'val/loss': loss,
        'val/epoch': epoch,
    }, step=step) 