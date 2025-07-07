#!/usr/bin/env python3
"""
Training script for both Contrastive and Discriminative Reward Models.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
import tempfile
import json
import numpy as np
import yaml

from src.data.dataset import create_dataloader
from src.models.contrastive_reward_model import ContrastiveRewardModel
from src.models.discriminative_reward_model import DiscriminativeRewardModel
from src.config.tokenization_config import PAD_TOKEN

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive learning.
    Calculates the loss for a batch of positive pairs by using in-batch negatives.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, melody_embeds, chord_embeds):
        """
        Args:
            melody_embeds (torch.Tensor): Embeddings for melody [batch_size, embed_dim]
            chord_embeds (torch.Tensor): Embeddings for chords [batch_size, embed_dim]
        """
        # Normalize embeddings
        melody_embeds = F.normalize(melody_embeds, p=2, dim=1)
        chord_embeds = F.normalize(chord_embeds, p=2, dim=1)

        # Calculate cosine similarity
        logits = torch.matmul(melody_embeds, chord_embeds.T) / self.temperature

        # The labels are the diagonal elements (i.e., the positive pairs)
        batch_size = melody_embeds.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # The loss is the cross-entropy between the logits and the labels
        loss_melody = self.criterion(logits, labels)
        loss_chord = self.criterion(logits.T, labels)
        
        return (loss_melody + loss_chord) / 2

class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy Loss for discriminative model.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits (torch.Tensor): Raw logits from model [batch_size, 1]
            labels (torch.Tensor): Binary labels [batch_size, 1]
        """
        return self.criterion(logits, labels)

def create_model(model_type, config, tokenizer_info, device):
    """
    Create either a contrastive or discriminative model based on config.
    """
    if model_type == 'contrastive':
        return ContrastiveRewardModel(
            melody_vocab_size=tokenizer_info['melody_vocab_size'],
            chord_vocab_size=tokenizer_info['chord_vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_length=config['max_seq_length'],
            pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN),
            scale_factor=config.get('scale_factor', 1.0)
        ).to(device)
    elif model_type == 'discriminative':
        return DiscriminativeRewardModel(
            vocab_size=tokenizer_info['total_vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_length=config['max_seq_length'],
            pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN),
            scale_factor=config.get('scale_factor', 1.0)
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_epoch(model, loader, optimizer, loss_fn, device, model_type):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        if model_type == 'contrastive':
            melody_tokens = batch['melody_tokens'].to(device)
            chord_tokens = batch['chord_tokens'].to(device)
            melody_mask = (melody_tokens == model.pad_token_id)
            chord_mask = (chord_tokens == model.pad_token_id)
            
            melody_embeds, chord_embeds = model(
                melody_tokens, 
                chord_tokens,
                melody_padding_mask=melody_mask,
                chord_padding_mask=chord_mask
            )
            loss = loss_fn(melody_embeds, chord_embeds)
        else:  # discriminative
            tokens = batch['interleaved_tokens'].to(device)
            labels = batch['labels'].float().to(device)
            padding_mask = (tokens == model.pad_token_id)
            
            logits = model(tokens, padding_mask)
            loss = loss_fn(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches

def validate(model, loader, loss_fn, device, model_type):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if model_type == 'contrastive':
                melody_tokens = batch['melody_tokens'].to(device)
                chord_tokens = batch['chord_tokens'].to(device)
                melody_mask = (melody_tokens == model.pad_token_id)
                chord_mask = (chord_tokens == model.pad_token_id)
                
                melody_embeds, chord_embeds = model(
                    melody_tokens, 
                    chord_tokens,
                    melody_padding_mask=melody_mask,
                    chord_padding_mask=chord_mask
                )
                loss = loss_fn(melody_embeds, chord_embeds)
            else:  # discriminative
                tokens = batch['interleaved_tokens'].to(device)
                labels = batch['labels'].float().to(device)
                padding_mask = (tokens == model.pad_token_id)
                
                logits = model(tokens, padding_mask)
                loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main(config):
    """Main training function."""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if 'wandb_project' not in config:
        raise ValueError("wandb_project not found in config")

    model_type = config.get('model_type', 'contrastive')
    scale_factor = config.get('scale_factor', 1.0)

    # --- W&B Setup ---
    run_name = (
        f"{model_type}_L{config['num_layers']}_H{config['num_heads']}"
        f"_D{config['embed_dim']}_seq{config['max_seq_length']}"
        f"_scale{scale_factor}_bs{config['batch_size']}"
        f"_lr{config['learning_rate']}"
    )

    wandb.init(
        project=config['wandb_project'],
        name=run_name,
        config=config,
        job_type=f"{model_type}_training"
    )

    # Dataloaders
    train_loader, tokenizer_info = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode=model_type,
        shuffle=True
    )
    valid_loader, _ = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode=model_type,
        shuffle=False
    )

    # Create model
    model = create_model(model_type, config, tokenizer_info, device)
    wandb.watch(model, log='all')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Loss and optimizer
    if model_type == 'contrastive':
        loss_fn = InfoNCELoss(temperature=config['temperature'])
    else:
        loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    best_valid_loss = float('inf')
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, model_type)
        valid_loss = validate(model, valid_loader, loss_fn, device, model_type)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "epoch": epoch + 1
        })
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'config': config,
                'tokenizer_info': tokenizer_info
            }
            
            checkpoint_path = Path(config['checkpoint_dir']) / f"{run_name}_best.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # Log best model to wandb
            artifact = wandb.Artifact(
                name=f"{model_type}_reward_model",
                type="model",
                description=f"Best {model_type} reward model checkpoint"
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
            
    wandb.finish()

def get_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Reward Model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick check to see if model and data load.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory specified in the config.")
    
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Add smoke_test flag to config
    config['smoke_test'] = args.smoke_test

    # Override data_dir if provided
    if args.data_dir:
        config['data_dir'] = args.data_dir

    main(config)