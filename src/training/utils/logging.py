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

from src.training.config import TrainingConfig

def init_wandb(config: "TrainingConfig", 
               name: Optional[str] = None, 
               job_type: Optional[str] = None):
    """
    Initialize a new W&B run.
    
    Args:
        config: Training configuration object
        name: Optional name for the run
        job_type: Optional type for the run (e.g., 'train', 'eval')
    """
    run_name = name or f"run_{generate_id()}"
    
    print(f" Initializing W&B run: {run_name}")
    print(f"  Project: {config.wandb_project}")
    print(f"  Entity: {config.wandb_entity}")
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.to_dict(),
        name=run_name,
        job_type=job_type,
        reinit=True
    )
    
    return run

def log_model_artifact(model: nn.Module, 
                       name: str, 
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
    checkpoint_path = checkpoint_dir / f"{name}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    
    # Create artifact
    artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=metadata
    )
    artifact.add_file(str(checkpoint_path))
    
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