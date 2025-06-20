#!/usr/bin/env python3
"""
Training script for the Discriminative Reward Model.
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from tqdm import tqdm
import wandb
import tempfile
import yaml

from src.data.dataset import create_dataloader
from src.models.discriminative_reward_model import DiscriminativeRewardModel

def create_negative_samples(interleaved_tokens: torch.Tensor) -> torch.Tensor:
    """
    Creates negative samples by shuffling chord progressions within a batch.
    
    Args:
        interleaved_tokens (torch.Tensor): A batch of real, interleaved sequences 
                                           [batch_size, seq_length] where tokens are
                                           arranged as [c, m, c, m, ...].
    
    Returns:
        torch.Tensor: A batch of fake sequences where melodies are paired with
                      chords from other sequences in the batch.
    """
    batch_size, seq_length = interleaved_tokens.shape
    
    # De-interleave into melody and chord tracks
    melody_tokens = interleaved_tokens[:, 1::2]
    chord_tokens = interleaved_tokens[:, 0::2]
    
    # Shuffle chord tokens across the batch dimension
    # This creates the negative pairs
    shuffled_indices = torch.randperm(batch_size)
    shuffled_chord_tokens = chord_tokens[shuffled_indices]
    
    # Re-interleave to create the fake sequences
    fake_interleaved = torch.empty_like(interleaved_tokens)
    fake_interleaved[:, 1::2] = melody_tokens
    fake_interleaved[:, 0::2] = shuffled_chord_tokens
    
    return fake_interleaved

def main(config):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # W&B setup
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config
    )
    run_name = wandb.run.name

    # Dataloaders - use 'online' mode for interleaved sequences
    train_loader = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'] * 2, # seq_len for dataloader is interleaved
        mode='online',
        shuffle=True
    )
    valid_loader = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'] * 2, # seq_len for dataloader is interleaved
        mode='online',
        shuffle=False
    )
    
    tokenizer_info = train_loader.dataset.tokenizer_info
    
    # Model
    model = DiscriminativeRewardModel(
        vocab_size=tokenizer_info['total_vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length'] * 2 # Pass the full length to the model
    ).to(device)
    
    wandb.watch(model, log='all')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- Smoke Test ---
    if config['smoke_test']:
        print("\n--- Smoke test successful: Model and data loaded correctly. ---")
        try:
            batch = next(iter(train_loader))
            real_tokens = batch['input_tokens'].to(device)
            real_padding_mask = batch['padding_mask'].to(device)
            model(real_tokens, padding_mask=real_padding_mask)
            print("--- Smoke test successful: Single forward pass completed. ---")
        except Exception as e:
            print(f"--- Smoke test failed during forward pass: {e} ---")
        return

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    best_valid_loss = float('inf')

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss, total_train_acc = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
        for batch in pbar:
            real_tokens = batch['input_tokens'].to(device)
            real_padding_mask = batch['padding_mask'].to(device)
            
            # Create negative samples
            fake_tokens = create_negative_samples(real_tokens).to(device)
            # For simplicity, we assume padding is the same for real/fake pairs
            # as shuffling chords shouldn't change padding distribution significantly.
            
            # Combine real and fake samples
            all_tokens = torch.cat([real_tokens, fake_tokens], dim=0)
            all_padding_masks = torch.cat([real_padding_mask, real_padding_mask], dim=0)
            
            # Create labels: 1 for real, 0 for fake
            real_labels = torch.ones(real_tokens.size(0), 1, device=device)
            fake_labels = torch.zeros(fake_tokens.size(0), 1, device=device)
            all_labels = torch.cat([real_labels, fake_labels], dim=0)

            optimizer.zero_grad()
            
            logits = model(all_tokens, padding_mask=all_padding_masks)
            loss = loss_fn(logits, all_labels)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Calculate accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == all_labels).float().mean()
            total_train_acc += acc.item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            wandb.log({'train/step_loss': loss.item(), 'train/step_acc': acc.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        
        # Validation loop
        model.eval()
        total_valid_loss, total_valid_acc = 0, 0
        with torch.no_grad():
            pbar_valid = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
            for batch in pbar_valid:
                real_tokens = batch['input_tokens'].to(device)
                real_padding_mask = batch['padding_mask'].to(device)
                
                fake_tokens = create_negative_samples(real_tokens).to(device)
                
                all_tokens = torch.cat([real_tokens, fake_tokens], dim=0)
                all_padding_masks = torch.cat([real_padding_mask, real_padding_mask], dim=0)
                
                real_labels = torch.ones(real_tokens.size(0), 1, device=device)
                fake_labels = torch.zeros(fake_tokens.size(0), 1, device=device)
                all_labels = torch.cat([real_labels, fake_labels], dim=0)
                
                logits = model(all_tokens, padding_mask=all_padding_masks)
                loss = loss_fn(logits, all_labels)
                total_valid_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == all_labels).float().mean()
                total_valid_acc += acc.item()

                pbar_valid.set_postfix({'loss': loss.item(), 'acc': acc.item()})

        avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_valid_acc = total_valid_acc / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {avg_valid_acc:.4f}")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'train/epoch_acc': avg_train_acc,
            'valid/epoch_loss': avg_valid_loss,
            'valid/epoch_acc': avg_valid_acc,
            'epoch': epoch + 1
        })
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "discriminator_model.pth"
                torch.save(model.state_dict(), model_path)
                
                artifact = wandb.Artifact(
                    f"discriminator-{run_name}",
                    type="model",
                    description="Discriminative reward model for melody-chord pair classification.",
                    metadata=config
                )
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)
                
            print(f"New best model saved with validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Discriminative Reward Model.")
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

    # If a run name isn't specified, create a default one
    if 'wandb_run_name' not in config or config['wandb_run_name'] is None:
        config['wandb_run_name'] = f"discriminator_bs{config['batch_size']}_lr{config['learning_rate']}"

    main(config) 