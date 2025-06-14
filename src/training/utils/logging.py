"""
Logging utilities for training
"""

import os
from pathlib import Path
import wandb
import torch

from src.training.config import TrainingConfig

def init_wandb(config: TrainingConfig, name: str) -> wandb.Run:
    """Initialize Weights & Biases run"""
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=name,
        config=config.to_dict()
    )
    
    return run

def log_metrics(metrics: dict, step: int):
    """Log metrics to wandb"""
    wandb.log(metrics, step=step)

def log_model_artifact(model: torch.nn.Module,
                      name: str,
                      metadata: dict = None):
    """Save model checkpoint and log as wandb artifact"""
    # Save checkpoint
    checkpoint_path = Path("checkpoints") / f"{name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {}
    }, checkpoint_path)
    
    # Log as artifact
    artifact = wandb.Artifact(
        name=name,
        type='model',
        description='Model checkpoint',
        metadata=metadata or {}
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    
    # Clean up local checkpoint
    os.remove(checkpoint_path)

def log_gradients(model: torch.nn.Module):
    """Log model gradients to wandb"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())}) 