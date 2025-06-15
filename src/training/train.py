#!/usr/bin/env python3
"""
Training script for OnlineTransformer model
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

from src.models.online_transformer import OnlineTransformer
from src.data.dataset import create_dataloader
from src.training.config import TrainingConfig
from src.training.utils.logging import init_wandb, log_model_artifact
from src.training.utils.metrics import MetricsTracker, log_training_metrics, log_validation_metrics

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
                metrics: MetricsTracker,
                step: int) -> int:
    """Standard autoregressive training"""
    model.train()
    metrics.start_epoch()
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch['input_tokens'])
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch['target_tokens'].view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        optimizer.step()
        scheduler.step()
        
        # Log metrics
        log_training_metrics(
            model=model,
            loss=loss,
            optimizer=optimizer,
            epoch=step // len(train_loader),
            step=step
        )
        
        step += 1
    
    return step

def validate(model: nn.Module,
            val_loader: DataLoader,
            device: torch.device,
            epoch: int,
            step: int) -> float:
    """Standard validation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            logits = model(batch['input_tokens'])
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['target_tokens'].view(-1)
            )
            total_loss += loss.item()
            num_batches += 1
    
    val_loss = total_loss / num_batches
    log_validation_metrics(val_loss, epoch, step)
    return val_loss

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
    
    # Log GPU info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
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
    run = init_wandb(config, name=f"online_transformer_{wandb.util.generate_id()}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode="online"
    )
    
    val_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="valid",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode="online"
    )
    
    # Set vocab_size dynamically from train dataset
    config.vocab_size = train_loader.dataset.total_vocab_size

    # Create model
    model = OnlineTransformer(config).to(device)
    
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
    
    # Initialize metrics tracker
    metrics = MetricsTracker()
    
    # Training loop
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.max_epochs):
        # Train
        step = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, config, metrics, step
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, step)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            log_model_artifact(
                model,
                f"model_epoch_{epoch}",
                metadata={"val_loss": best_val_loss}
            )
        
        # Log epoch time
        metrics.end_epoch(epoch)
    
    # Log final metrics
    wandb.run.summary.update({
        'final_epoch': config.max_epochs - 1,
        'best_val_loss': best_val_loss
    })
    
    # Cleanup
    run.finish()

if __name__ == "__main__":
    main() 