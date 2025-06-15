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
    """Log basic training metrics"""
    # Training metrics
    metrics = {
        'train/loss': loss.item(),
        'train/learning_rate': optimizer.param_groups[0]['lr'],
        'train/epoch': epoch,
    }
    
    # Gradient norms (for plotting)
    grad_norms = {
        f'gradients/layer_{i}_norm': p.grad.norm().item()
        for i, p in enumerate(model.parameters())
        if p.grad is not None
    }
    metrics.update(grad_norms)
    
    # Parameter norms (for plotting)
    param_norms = {
        f'parameters/layer_{i}_norm': p.norm().item()
        for i, p in enumerate(model.parameters())
    }
    metrics.update(param_norms)
    
    # Hardware metrics
    if torch.cuda.is_available():
        metrics.update({
            'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1024**3,
            'system/gpu_utilization': torch.cuda.utilization(),
        })
    
    wandb.log(metrics, step=step)

def log_validation_metrics(loss: float, epoch: int, step: int):
    """Log validation metrics"""
    wandb.log({
        'val/loss': loss,
        'val/epoch': epoch,
    }, step=step) 