#!/usr/bin/env python3
"""
W&B logging utilities
"""

import os
from pathlib import Path
import wandb
import torch
from typing import Any, Dict, Optional
from wandb.util import generate_id
import torch.nn as nn
import json

def log_model_artifact(model: nn.Module, 
                       name: str, 
                       tokenizer_info: Dict,
                       metadata: Optional[Dict[str, Any]] = None):
    """
    Save model checkpoint and log it as a W&B artifact.
    """
    # Default to empty dictionary if metadata is None
    metadata = metadata or {}
    
    # Get current run's checkpoint directory
    checkpoint_dir = Path(wandb.run.dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_path = checkpoint_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save tokenizer info
    tokenizer_path = checkpoint_dir / "tokenizer_info.json"
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_info, f, indent=4)
    
    # Create artifact
    artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=metadata
    )
    artifact.add_file(str(model_path))
    artifact.add_file(str(tokenizer_path))
    
    # Log artifact
    wandb.log_artifact(artifact)
    print(f"Logged model artifact: {name}")

def log_training_metrics(model: nn.Module, 
                         loss: float, 
                         optimizer: torch.optim.Optimizer, 
                         epoch: int, 
                         step: int):
    """Log metrics during training"""
    lr = optimizer.param_groups[0]['lr']
    wandb.log({
        'train/loss': loss,
        'train/learning_rate': lr,
        'train/epoch': epoch,
    }, step=step)

def log_validation_metrics(loss: float, epoch: int, step: int):
    """Log metrics during validation"""
    wandb.log({
        'valid/loss': loss,
        'valid/epoch': epoch
    }, step=step)

def log_gradients(model: torch.nn.Module):
    """Log model gradients to wandb"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}) 