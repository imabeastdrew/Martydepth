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
    train_loader, tokenizer_info = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode='contrastive',
        shuffle=True
    )
    valid_loader, _ = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode='contrastive',
        shuffle=False
    )
    

    model = ContrastiveRewardModel(
        melody_vocab_size=tokenizer_info['melody_vocab_size'],
        chord_vocab_size=tokenizer_info['chord_vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length'],
        pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
    ).to(device)
    
    wandb.watch(model, log='all')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Loss and optimizer
    loss_fn = InfoNCELoss(temperature=config['temperature'])
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # --- Smoke Test ---
    if config.get('smoke_test', False):
        print("\n--- Starting Smoke Test ---")
        
        # Override config with minimal values for smoke test
        smoke_config = config.copy()
        smoke_config.update({
            'batch_size': 2,  # Minimal batch size
            'max_seq_length': 32,  # Shorter sequences
            'num_workers': 0,  # No parallel loading
        })
        
        try:
            print("1. Testing data loading...")
            # Create minimal dataloader
            smoke_loader, _ = create_dataloader(
                data_dir=Path(config['data_dir']),
                split="valid",  # Use validation set (usually smaller)
                batch_size=smoke_config['batch_size'],
                num_workers=smoke_config['num_workers'],
                sequence_length=smoke_config['max_seq_length'],
                mode='contrastive',
                shuffle=False  # No need to shuffle for smoke test
            )
            
            print("2. Testing model initialization...")
            # Create model with minimal config
            smoke_model = ContrastiveRewardModel(
                melody_vocab_size=tokenizer_info['melody_vocab_size'],
                chord_vocab_size=tokenizer_info['chord_vocab_size'],
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                max_seq_length=smoke_config['max_seq_length'],
                pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
            ).to(device)
            
            print("3. Testing forward pass...")
            # Get single batch
            batch = next(iter(smoke_loader))
            melody_tokens = batch['melody_tokens'].to(device)
            chord_tokens = batch['chord_tokens'].to(device)
            
            # Create padding masks
            melody_mask = (melody_tokens == smoke_model.pad_token_id)
            chord_mask = (chord_tokens == smoke_model.pad_token_id)
            
            # Run forward pass
            with torch.no_grad():  # No need for gradients in smoke test
                melody_embeds, chord_embeds = smoke_model(
                    melody_tokens,
                    chord_tokens,
                    melody_padding_mask=melody_mask,
                    chord_padding_mask=chord_mask
                )
            
            print("4. Testing loss function...")
            loss_fn = InfoNCELoss(temperature=config['temperature'])
            loss = loss_fn(melody_embeds, chord_embeds)
            
            print("5. Testing optimizer creation...")
            smoke_optimizer = Adam(smoke_model.parameters(), lr=config['learning_rate'])
            
            # Clean up
            del smoke_model, melody_embeds, chord_embeds, melody_mask, chord_mask
            torch.cuda.empty_cache()
            
            print("--- Smoke test successful: All components verified. ---")
            
        except Exception as e:
            print(f"--- Smoke test failed: {str(e)} ---")
            raise e
        
        return

    # Training loop
    best_valid_loss = float('inf')
    global_step = 0

    print(f"\n--- Starting Training ---")
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
        for batch in pbar:
            melody_tokens = batch['melody_tokens'].to(device)
            chord_tokens = batch['chord_tokens'].to(device)
            
            # Create padding masks
            melody_mask = (melody_tokens == model.pad_token_id)
            chord_mask = (chord_tokens == model.pad_token_id)

            optimizer.zero_grad()
            
            melody_embeds, chord_embeds = model(
                melody_tokens, 
                chord_tokens,
                melody_padding_mask=melody_mask,
                chord_padding_mask=chord_mask
            )
            loss = loss_fn(melody_embeds, chord_embeds)

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
                
                # Create padding masks
                melody_mask = (melody_tokens == model.pad_token_id)
                chord_mask = (chord_tokens == model.pad_token_id)
                
                melody_embeds, chord_embeds = model(
                    melody_tokens, 
                    chord_tokens,
                    melody_padding_mask=melody_mask,
                    chord_padding_mask=chord_mask
                )
                loss = loss_fn(melody_embeds, chord_embeds)
                
                total_valid_loss += loss.item()
                pbar_valid.set_postfix({'loss': loss.item()})

                # Calculate Top-1 Accuracy
                logits = torch.matmul(F.normalize(melody_embeds, p=2, dim=1), F.normalize(chord_embeds, p=2, dim=1).T)
                sorted_indices = torch.argsort(logits, descending=True, dim=1)
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
        }, step=global_step)
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Save model weights with a consistent name
                model_path = tmpdir_path / "model.pth"
                torch.save(model.state_dict(), model_path)
                
                # Save the tokenizer info
                tokenizer_path = tmpdir_path / "tokenizer_info.json"
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer_info, f, indent=4)
                
                # Save metadata
                metadata_path = tmpdir_path / "metadata.json"
                with open(metadata_path, 'w') as f:
                    # Add validation metrics to metadata
                    config['best_val_loss'] = best_valid_loss
                    config['top1_accuracy'] = top1_accuracy
                    json.dump(config, f, indent=4)
                
                # Log as artifact
                artifact = wandb.Artifact(
                    name=f"contrastive-L{config['max_seq_length']}-{wandb.run.id}",
                    type="reward_model",
                    metadata=config
                )
                artifact.add_dir(str(tmpdir_path))
                wandb.log_artifact(artifact)
                print(f"  Best model saved with val_loss: {best_valid_loss:.4f}")
    
    wandb.finish()

def get_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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