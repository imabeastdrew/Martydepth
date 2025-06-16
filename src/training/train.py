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
from transformers import Adafactor
from torch.optim.lr_scheduler import LambdaLR
import wandb
from jsonargparse import ArgumentParser

from src.models.online_transformer import OnlineTransformer
from src.data.dataset import create_dataloader
from src.training.config import TrainingConfig
from src.training.utils.logging import init_wandb, log_model_artifact
from src.training.utils.metrics import log_training_metrics, log_validation_metrics

def get_warmup_schedule(optimizer, num_warmup_steps):
    """Create warmup schedule"""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))  # Linear warmup
        return 1.0  # Full learning rate
    return LambdaLR(optimizer, lr_lambda)

def train_step(model: nn.Module,
               batch: dict,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               device: torch.device,
               config: TrainingConfig) -> float:
    """Single training step with interleaved sequence prediction"""
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Forward pass on interleaved sequence
    logits = model(batch['input_tokens'], padding_mask=batch.get('padding_mask'))
    
    # Calculate loss (predict next token)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        batch['target_tokens'].reshape(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
    optimizer.step()
    scheduler.step()
    
    return loss.item()

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          device: torch.device,
          config: TrainingConfig):
    """Training loop with step-based training and validation"""
    model.train()
    
    global_step = 0
    best_val_loss = float('inf')
    steps_per_epoch = len(train_loader)
    
    print(f"\nTraining Info:")
    print(f"  Total steps: {config.total_steps:,}")
    print(f"  Warmup steps: {config.warmup_steps:,}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Estimated epochs: {config.total_steps / steps_per_epoch:.1f}")
    
    train_iter = iter(train_loader)
    current_epoch = 0
    
    while global_step < config.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            current_epoch += 1
            
            # Validate at epoch boundaries
            val_loss = validate(model, val_loader, device)
            log_validation_metrics(loss=val_loss, epoch=current_epoch, step=global_step)
            print(f"\nEpoch {current_epoch} Validation:")
            print(f"  Loss: {val_loss:.4f}")
            
            # Save if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("  New best validation loss! Saving checkpoint...")
                log_model_artifact(
                    model,
                    f"model_step_{global_step}",
                    metadata={"val_loss": val_loss}
                )
        
        # Training step
        loss = train_step(model, batch, optimizer, scheduler, device, config)
        
        # Log metrics
        log_training_metrics(
            model=model,
            loss=loss,
            optimizer=optimizer,
            epoch=current_epoch,
            step=global_step
        )
        
        global_step += 1
        
        if global_step % 100 == 0:
            print(f"\rStep {global_step}/{config.total_steps} "
                  f"({global_step/config.total_steps*100:.1f}%) "
                  f"Loss: {loss:.4f}", end="")
    
    print("\n\nTraining complete!")
    print(f"  Steps completed: {global_step:,}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    
    wandb.run.summary.update({
        'final_step': global_step,
        'best_val_loss': best_val_loss
    })

def validate(model: nn.Module,
            val_loader: DataLoader,
            device: torch.device) -> float:
    """Validation with interleaved sequence prediction"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch['input_tokens'], padding_mask=batch.get('padding_mask'))
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                batch['target_tokens'].reshape(-1)
            )
            
            total_loss += loss.item()  # Convert to float here
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss

def main(config: TrainingConfig):
    """Main entry point for training."""
    # Initialize wandb
    run = init_wandb(
        config, 
        name=f"online_transformer_{wandb.util.generate_id()}",
        job_type="online_training"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode='online'
    )
    
    val_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="valid",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode='online'
    )
    
    # Update config with vocab sizes from the dataset
    dataset_info = train_loader.dataset.tokenizer_info
    config.vocab_size = dataset_info['total_vocab_size'] + 1
    config.melody_vocab_size = dataset_info['melody_vocab_size'] + 1
    config.chord_vocab_size = dataset_info['chord_vocab_size'] + 1

    # Create model
    model = OnlineTransformer(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_seq_length=config.max_sequence_length
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        scale_parameter=False,
        relative_step=False
    )
    
    scheduler = get_warmup_schedule(
        optimizer,
        num_warmup_steps=config.warmup_steps
    )
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Cleanup
    run.finish()

if __name__ == "__main__":
    # Use jsonargparse to automatically handle config loading from YAML and CLI
    parser = ArgumentParser()
    parser.add_class_arguments(TrainingConfig, "config")
    
    cfg = parser.parse_args()
    
    # The config object is nested under 'config' namespace
    main(config=cfg.config) 