#!/usr/bin/env python3
"""
Training script for the Offline Teacher model.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Adafactor
from torch.optim.lr_scheduler import LambdaLR
import wandb
from jsonargparse import ArgumentParser

from src.models.offline_teacher import OfflineTeacherModel
from src.data.dataset import create_dataloader
from src.training.config import TrainingConfig
from src.training.utils.logging import init_wandb, log_model_artifact
from src.training.utils.metrics import log_training_metrics, log_validation_metrics

def get_warmup_schedule(optimizer, num_warmup_steps):
    """Create a linear warmup schedule."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

def train_step(model: nn.Module,
               batch: dict,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler.LRScheduler,
               device: torch.device,
               config: TrainingConfig) -> float:
    """A single training step for the offline teacher model."""
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Forward pass: provide full melody and groud-truth chords (teacher forcing)
    logits = model(
        melody_tokens=batch['melody_tokens'],
        chord_tokens=batch['chord_input']
    )
    
    # Calculate loss against the target chord sequence
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, config.chord_vocab_size),
        batch['chord_target'].reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_val)
    optimizer.step()
    scheduler.step()
    
    return loss.item()

def validate(model: nn.Module,
             val_loader: DataLoader,
             device: torch.device,
             config: TrainingConfig) -> float:
    """Validation loop for the offline teacher model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = model(
                melody_tokens=batch['melody_tokens'],
                chord_tokens=batch['chord_input']
            )
            
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, config.chord_vocab_size),
                batch['chord_target'].reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler,
          device: torch.device,
          config: TrainingConfig):
    """Main training loop."""
    model.train()
    global_step = 0
    best_val_loss = float('inf')
    
    train_iter = iter(train_loader)
    while global_step < config.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            current_epoch = global_step // len(train_loader)
            val_loss = validate(model, val_loader, device, config)
            log_validation_metrics(loss=val_loss, epoch=current_epoch, step=global_step)
            print(f"\nStep {global_step} | Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("  New best validation loss! Saving checkpoint...")
                log_model_artifact(model, f"offline_teacher_step_{global_step}", metadata={"val_loss": val_loss})

            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        loss = train_step(model, batch, optimizer, scheduler, device, config)
        
        if global_step % config.log_every_n_steps == 0:
            current_epoch = global_step // len(train_loader)
            log_training_metrics(model=model, loss=loss, optimizer=optimizer, epoch=current_epoch, step=global_step)
            print(f"\rStep {global_step}/{config.total_steps} | Loss: {loss:.4f}", end="")
        
        global_step += 1
    
    print("\nTraining complete.")

def main(config: TrainingConfig):
    """Main entry point for training."""
    run = init_wandb(config, name=f"offline_teacher_{wandb.util.generate_id()}", job_type="offline_training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode='offline'
    )
    
    val_loader = create_dataloader(
        data_dir=Path(config.data_dir),
        split="valid",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sequence_length=config.max_sequence_length,
        mode='offline'
    )
    
    # Update config with vocab sizes from the dataset
    dataset_info = train_loader.dataset.tokenizer_info
    config.melody_vocab_size = dataset_info['melody_vocab_size']
    config.chord_vocab_size = dataset_info['chord_vocab_size']

    model = OfflineTeacherModel(
        melody_vocab_size=config.melody_vocab_size,
        chord_vocab_size=config.chord_vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_seq_length=config.max_sequence_length
    ).to(device)

    optimizer = Adafactor(
        model.parameters(),
        lr=config.learning_rate,
        scale_parameter=False,
        relative_step=False
    )
    
    scheduler = get_warmup_schedule(optimizer, num_warmup_steps=config.warmup_steps)
    
    train(model, train_loader, val_loader, optimizer, scheduler, device, config)
    
    run.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_class_arguments(TrainingConfig, "config")
    cfg = parser.parse_args()
    main(config=cfg.config) 