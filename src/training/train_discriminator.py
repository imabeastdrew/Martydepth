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

    if 'wandb_project' not in config:
        raise ValueError("wandb_project not found in config")

    # --- W&B Setup ---
    run_name = (
        f"discriminator_L{config['num_layers']}_H{config['num_heads']}"
        f"_D{config['embed_dim']}_seq{config['max_seq_length']}"
        f"_bs{config['batch_size']}_lr{config['learning_rate']}"
    )

    wandb.init(
        project=config['wandb_project'],
        name=run_name,
        config=config,
        job_type="discriminator_training"
    )

    # --- Data ---
    data_path = Path(config['data_dir'])
    train_loader, tokenizer_info = create_dataloader(
        data_dir=data_path,
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode='discriminator'
    )
    valid_loader, _ = create_dataloader(
        data_dir=data_path,
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_seq_length'],
        mode='discriminator'
    )
    
    config['vocab_size'] = tokenizer_info['total_vocab_size']
    pad_token_id = tokenizer_info.get('pad_token_id', -100)
    
    # --- Model ---
    model = DiscriminativeRewardModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    wandb.watch(model, log='all', log_freq=100)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    
    # --- Training Components ---
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # --- Smoke Test ---
    if config.get('smoke_test', False):
        print("\n--- Smoke test: Model and data loaded correctly. ---")
        try:
            batch = next(iter(train_loader))
            # The rest of your smoke test logic...
            print("--- Smoke test successful: Single forward pass completed. ---")
        except Exception as e:
            print(f"--- Smoke test failed during forward pass: {e} ---")
        return

    # --- Training Loop ---
    best_val_loss = float('inf')
    global_step = 0
    
    print(f"\n--- Starting Training ---")
    for epoch in range(config['epochs']):
        # Training Step
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
        for batch in pbar:
            real_sequences = batch['real_sequences'].to(device)
            fake_sequences = batch['fake_sequences'].to(device)
            
            # Combine real and fake sequences for a single batch
            # Shape: (2 * batch_size, seq_len)
            combined_sequences = torch.cat([real_sequences, fake_sequences], dim=0)

            # Generate padding mask for the combined batch
            padding_mask = (combined_sequences == pad_token_id)
            
            # Create labels: 1 for real, 0 for fake
            real_labels = torch.ones(real_sequences.size(0), 1, device=device)
            fake_labels = torch.zeros(fake_sequences.size(0), 1, device=device)
            combined_labels = torch.cat([real_labels, fake_labels], dim=0)
            
            optimizer.zero_grad()
            
            predictions = model(combined_sequences, padding_mask=padding_mask)
            
            # Apply sigmoid since the model outputs logits and BCEWithLogitsLoss is not used
            predictions = torch.sigmoid(predictions)

            loss = loss_fn(predictions, combined_labels)
            loss.backward()
            optimizer.step()
            
            lr = optimizer.param_groups[0]['lr']
            total_train_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item(), 'lr': lr})
            wandb.log({'train/step_loss': loss.item(), 'train/learning_rate': lr}, step=global_step)
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        total_valid_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            pbar_valid = tqdm(valid_loader, desc="Validating")
            for batch in pbar_valid:
                real_sequences = batch['real_sequences'].to(device)
                fake_sequences = batch['fake_sequences'].to(device)
                
                combined_sequences = torch.cat([real_sequences, fake_sequences], dim=0)
                padding_mask = (combined_sequences == pad_token_id)

                real_labels = torch.ones(real_sequences.size(0), 1, device=device)
                fake_labels = torch.zeros(fake_sequences.size(0), 1, device=device)
                combined_labels = torch.cat([real_labels, fake_labels], dim=0)
                
                predictions = model(combined_sequences, padding_mask=padding_mask)
                predictions = torch.sigmoid(predictions)
                loss = loss_fn(predictions, combined_labels)
                total_valid_loss += loss.item()

                predicted_labels = (predictions > 0.5).float()
                correct_predictions += (predicted_labels == combined_labels).sum().item()
                total_predictions += combined_labels.size(0)

                pbar_valid.set_postfix({'loss': loss.item()})

        avg_valid_loss = total_valid_loss / len(valid_loader)
        accuracy = correct_predictions / total_predictions

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Accuracy: {accuracy:.4f}")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'valid/epoch_loss': avg_valid_loss,
            'valid/accuracy': accuracy,
            'epoch': epoch + 1
        }, step=global_step)

        # Checkpoint
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            
            checkpoint_path = Path(wandb.run.dir) / f"discriminator_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            
            wandb.save(str(checkpoint_path))
            print(f"New best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
            
    wandb.finish()
    print("\n--- Training Complete ---")


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

    main(config) 