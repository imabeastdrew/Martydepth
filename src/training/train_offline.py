#!/usr/bin/env python3
"""
Training script for OfflineTeacherModel
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

from src.models.offline_teacher import OfflineTeacherModel
from src.data.dataset import create_dataloader
from src.training.config import TrainingConfig
from src.training.utils.logging import init_wandb, log_model_artifact

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create linear schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                  float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: torch.device,
                config: TrainingConfig,
                step: int) -> int:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        # Move batch to device
        melody_tokens = batch['melody_tokens'].to(device)
        chord_input = batch['chord_input'].to(device)
        chord_target = batch['chord_target'].to(device)
        
        # Forward pass
        chord_logits = model(melody_tokens, chord_input)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            chord_logits.view(-1, chord_logits.size(-1)),
            chord_target.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Log metrics
        if step % config.log_every_n_steps == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0]
            }, step=step)
        
        step += 1
    
    # Log epoch metrics
    avg_loss = total_loss / num_batches
    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/epoch": step // len(train_loader)
    })
    
    return step

def validate(model: nn.Module,
            val_loader: DataLoader,
            device: torch.device) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            melody_tokens = batch['melody_tokens'].to(device)
            chord_input = batch['chord_input'].to(device)
            chord_target = batch['chord_target'].to(device)
            
            # Forward pass
            chord_logits = model(melody_tokens, chord_input)
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                chord_logits.view(-1, chord_logits.size(-1)),
                chord_target.view(-1)
            )
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/interim")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="martydepth")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Initialize wandb
    run = init_wandb(config, name=f"offline_teacher_{wandb.util.generate_id()}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode="offline"  # Use offline mode for teacher model
    )
    
    val_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="valid",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode="offline"  # Use offline mode for teacher model
    )
    
    # Set vocab size from dataset
    config.vocab_size = train_loader.dataset.total_vocab_size

    # Create model
    model = OfflineTeacherModel(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.max_epochs):
        # Train
        step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config, step
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        wandb.log({"val/loss": val_loss}, step=step)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log_model_artifact(
                model,
                f"offline_teacher_epoch_{epoch}",
                metadata={"val_loss": best_val_loss}
            )
    
    # Cleanup
    run.finish()

if __name__ == "__main__":
    main() 