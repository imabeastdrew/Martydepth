#!/usr/bin/env python3
"""
Training script for the Contrastive Reward Model.
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
        # The logits are the similarity matrix between every melody and every chord in the batch
        logits = torch.matmul(melody_embeds, chord_embeds.T) / self.temperature

        # The labels are the diagonal elements (i.e., the positive pairs)
        batch_size = melody_embeds.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # The loss is the cross-entropy between the logits and the labels
        loss_melody = self.criterion(logits, labels)
        loss_chord = self.criterion(logits.T, labels)
        
        return (loss_melody + loss_chord) / 2

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

    # --- W&B Setup ---
    run_name = (
        f"contrastive_L{config['num_layers']}_H{config['num_heads']}"
        f"_D{config['embed_dim']}_seq{config['max_seq_length']}"
        f"_bs{config['batch_size']}_lr{config['learning_rate']}"
    )

    wandb.init(
        project=config['wandb_project'],
        name=run_name,
        config=config,
        job_type="contrastive_training"
    )

    # Dataloaders
    train_loader = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode='contrastive',
        shuffle=True
    )
    valid_loader = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode='contrastive',
        shuffle=False
    )
    
    tokenizer_info = train_loader.dataset.tokenizer_info
    
    # Model
    model = ContrastiveRewardModel(
        melody_vocab_size=tokenizer_info['melody_vocab_size'],
        chord_vocab_size=tokenizer_info['chord_vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    ).to(device)
    
    wandb.watch(model, log='all')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- Smoke Test ---
    if config['smoke_test']:
        print("\n--- Smoke test successful: Model and data loaded correctly. ---")
        # Optional: try one forward pass
        try:
            batch = next(iter(train_loader))
            melody_tokens = batch['melody_tokens'].to(device)
            chord_tokens = batch['chord_tokens'].to(device)
            model(melody_tokens, chord_tokens)
            print("--- Smoke test successful: Single forward pass completed. ---")
        except Exception as e:
            print(f"--- Smoke test failed during forward pass: {e} ---")
        return

    # Loss and optimizer
    loss_fn = InfoNCELoss(temperature=config['temperature'])
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    best_valid_loss = float('inf')
    global_step = 0

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
        for batch in pbar:
            melody_tokens = batch['melody_tokens'].to(device)
            chord_tokens = batch['chord_tokens'].to(device)

            optimizer.zero_grad()
            
            mel_embeds, chord_embeds = model(melody_tokens, chord_tokens)
            loss = loss_fn(mel_embeds, chord_embeds)
            
            loss.backward()
            optimizer.step()
            
            lr = optimizer.param_groups[0]['lr']
            total_train_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item(), 'lr': lr})
            wandb.log({'train/step_loss': loss.item(), 'train/learning_rate': lr}, step=global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation loop
        model.eval()
        total_valid_loss = 0
        all_ranks = []
        with torch.no_grad():
            pbar_valid = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
            for batch in pbar_valid:
                melody_tokens = batch['melody_tokens'].to(device)
                chord_tokens = batch['chord_tokens'].to(device)
                
                mel_embeds, chord_embeds = model(melody_tokens, chord_tokens)
                loss = loss_fn(mel_embeds, chord_embeds)
                
                total_valid_loss += loss.item()
                pbar_valid.set_postfix({'loss': loss.item()})

                # Calculate Top-1 Accuracy
                logits = torch.matmul(F.normalize(mel_embeds, p=2, dim=1), F.normalize(chord_embeds, p=2, dim=1).T)
                # Get the rank of the positive example for each row
                sorted_indices = torch.argsort(logits, descending=True, dim=1)
                # Ground truth is a diagonal matrix, so we want the rank of element `i` in row `i`
                labels = torch.arange(len(logits), device=logits.device)
                ranks = (sorted_indices == labels[:, None]).nonzero(as_tuple=True)[1] + 1
                all_ranks.extend(ranks.cpu().numpy())

        avg_valid_loss = total_valid_loss / len(valid_loader)
        top1_accuracy = np.mean(np.array(all_ranks) == 1) * 100
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Top-1 Acc: {top1_accuracy:.2f}%")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'valid/epoch_loss': avg_valid_loss,
            'valid/top1_accuracy': top1_accuracy,
            'epoch': epoch + 1
        })
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "reward_model.pth"
                torch.save(model.state_dict(), model_path)
                
                artifact = wandb.Artifact(
                    f"reward_model-{run_name}",
                    type="model",
                    description="Contrastive reward model trained on melody-chord similarity.",
                    metadata=config
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                
            print(f"New best model saved with validation loss: {best_valid_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Contrastive Reward Model.")
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