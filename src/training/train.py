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
import wandb
from jsonargparse import ArgumentParser

from src.models.online_transformer import OnlineTransformer
from src.data.dataset import create_dataloader
from src.training.config import TrainingConfig
from src.training.utils.logging import init_wandb, log_model_artifact
from src.training.utils.metrics import log_training_metrics, log_validation_metrics
from src.training.utils.schedulers import get_warmup_schedule

def train_step(model: nn.Module,
               batch: dict,
               optimizer: torch.optim.Optimizer,
               scheduler,
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
          config: TrainingConfig,
          tokenizer_info: dict):
    """Training loop with epoch-based training and validation"""
    model.train()
    
    global_step = 0
    best_val_loss = float('inf')
    steps_without_improvement = 0
    
    print(f"\nTraining Info:")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Early stopping patience: {config.early_stopping_patience}")

    for epoch in range(config.max_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.max_epochs} ---")
        model.train()
        
        for i, batch in enumerate(train_loader):
            loss = train_step(model, batch, optimizer, scheduler, device, config)
            global_step += 1
            
            if global_step % config.log_every_n_steps == 0:
                log_training_metrics(
                    model=model,
                    loss=loss,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step
                )
                print(f"\rEpoch {epoch+1}, Step {global_step} | Loss: {loss:.4f}", end="")

        # --- Validation at the end of the epoch ---
        val_loss = validate(model, val_loader, device)
        log_validation_metrics(loss=val_loss, epoch=epoch, step=global_step)
        print(f"\nEpoch {epoch+1} Validation:")
        print(f"  Loss: {val_loss:.4f}")
        
        # Save if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            steps_without_improvement = 0
            print("  New best validation loss! Saving checkpoint...")
            log_model_artifact(
                model,
                f"model_epoch_{epoch+1}_loss_{val_loss:.2f}",
                tokenizer_info=tokenizer_info,
                metadata={"val_loss": val_loss, "epoch": epoch+1}
            )
        else:
            steps_without_improvement += 1
            print(f"  Validation loss did not improve. Patience: {steps_without_improvement}/{config.early_stopping_patience}")

        # Check for early stopping
        if steps_without_improvement >= config.early_stopping_patience:
            print(f"\nStopping early after {config.early_stopping_patience} epochs with no improvement.")
            break
    
    print("\n\nTraining complete!")
    print(f"  Epochs completed: {epoch+1}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    
    wandb.run.summary.update({
        'final_epoch': epoch+1,
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
    config.vocab_size = dataset_info['total_vocab_size']
    config.melody_vocab_size = dataset_info['melody_vocab_size']
    config.chord_vocab_size = dataset_info['chord_vocab_size']

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
        config=config,
        tokenizer_info=dataset_info
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