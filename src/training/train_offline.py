#!/usr/bin/env python3
"""
Training script for the Offline Teacher model.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import wandb
import yaml
from tqdm import tqdm
from torch.optim import AdamW  # Switched from Adafactor to AdamW

from src.models.offline_teacher_t5 import T5OfflineTeacherModel
from src.data.dataset import create_dataloader
from src.training.utils.logging import log_model_artifact
from src.training.utils.schedulers import get_warmup_schedule
from src.config.tokenization_config import PAD_TOKEN

def main(config: dict):
    """Main training function."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- W&B Setup ---
    run_name = (
        f"offline_L{config['num_layers']}_H{config['num_heads']}"
        f"_D{config['embed_dim']}_seq{config['max_sequence_length']}"
        f"_bs{config['batch_size']}_lr{config['learning_rate']}"
    )

    wandb.init(
        project=config['wandb_project'],
        name=run_name,
        config=config,
        job_type="offline_training"
    )

    # --- Dataloaders ---
    train_loader, tokenizer_info = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_sequence_length'],
        mode='offline',
        shuffle=True
    )
    
    val_loader, _ = create_dataloader(
        data_dir=Path(config['data_dir']),
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_sequence_length'],
        mode='offline',
        shuffle=False
    )
    
    config['melody_vocab_size'] = tokenizer_info['melody_vocab_size']
    config['chord_vocab_size'] = tokenizer_info['chord_vocab_size']
    
    # --- Model, Optimizer, Scheduler ---
    model = T5OfflineTeacherModel(
        melody_vocab_size=config['melody_vocab_size'],
        chord_vocab_size=config['chord_vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_sequence_length'],
        pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN),
        total_vocab_size=tokenizer_info.get('total_vocab_size', 4779)  # Use unified vocabulary
    ).to(device)

    # Use AdamW optimizer instead of Adafactor for better stability and simpler tuning
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    scheduler = get_warmup_schedule(optimizer, num_warmup_steps=config['warmup_steps'])

    wandb.watch(model, log='all')

    # --- Smoke Test ---
    if config['smoke_test']:
        print("\n--- Starting Smoke Test ---")
        
        # Override config with minimal values for smoke test
        smoke_config = config.copy()
        smoke_config.update({
            'batch_size': 2,  # Minimal batch size
            'max_sequence_length': 32,  # Shorter sequences
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
                sequence_length=smoke_config['max_sequence_length'],
                mode='offline',
                shuffle=False  # No need to shuffle for smoke test
            )
            
            print("2. Testing model initialization...")
            # Create model with minimal config
            smoke_model = T5OfflineTeacherModel(
                melody_vocab_size=config['melody_vocab_size'],
                chord_vocab_size=config['chord_vocab_size'],
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                max_seq_length=smoke_config['max_sequence_length'],
                pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
            ).to(device)
            
            print("3. Testing forward pass...")
            # Get single batch
            batch = next(iter(smoke_loader))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Create masks
            melody_mask = (batch['melody_tokens'] == smoke_model.pad_token_id)
            chord_mask = (batch['chord_input'] == smoke_model.pad_token_id)
            
            # Run forward pass
            with torch.no_grad():  # No need for gradients in smoke test
                smoke_model(
                    melody_tokens=batch['melody_tokens'],
                    chord_tokens=batch['chord_input'],
                    melody_mask=melody_mask,
                    chord_mask=chord_mask
                )
            
            print("4. Testing optimizer creation...")
            smoke_optimizer = AdamW(
                smoke_model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.01)
            )
            
            # Clean up
            del smoke_model, batch, melody_mask, chord_mask
            torch.cuda.empty_cache()
            
            print("--- Smoke test successful: All components verified. ---")
            
        except Exception as e:
            print(f"--- Smoke test failed: {str(e)} ---")
            raise e
        
        return

    # --- Training Loop ---
    best_val_loss = float('inf')
    global_step = 0

    print(f"\n--- Offline Training Info ---")
    print(f"  Max epochs: {config['max_epochs']}")

    for epoch in range(config['max_epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['max_epochs']} ---")
        
        # Training
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Create masks
            melody_mask = (batch['melody_tokens'] == model.pad_token_id)
            chord_mask = (batch['chord_input'] == model.pad_token_id)
            
            logits = model(
                melody_tokens=batch['melody_tokens'],
                chord_tokens=batch['chord_input'],
                melody_mask=melody_mask,
                chord_mask=chord_mask
            )
            
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, model.total_vocab_size),
                batch['chord_target'].reshape(-1),
                ignore_index=model.pad_token_id
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip_val'])
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[0]['lr']
            total_train_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item(), 'lr': lr})
            wandb.log({'train/step_loss': loss.item(), 'train/learning_rate': lr}, step=global_step)
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Create masks
                melody_mask = (batch['melody_tokens'] == model.pad_token_id)
                chord_mask = (batch['chord_input'] == model.pad_token_id)

                logits = model(
                    melody_tokens=batch['melody_tokens'],
                    chord_tokens=batch['chord_input'],
                    melody_mask=melody_mask,
                    chord_mask=chord_mask
                )
                
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, model.total_vocab_size),
                    batch['chord_target'].reshape(-1),
                    ignore_index=model.pad_token_id
                )
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'valid/epoch_loss': avg_val_loss,
            'epoch': epoch + 1
        }, step=global_step)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss! Saving model artifact...")
            log_model_artifact(
                model,
                f"offline_teacher_epoch_{epoch+1}_loss_{avg_val_loss:.2f}",
                tokenizer_info=tokenizer_info,
                metadata={"val_loss": avg_val_loss, "epoch": epoch+1, **config}
            )
            
    print("\nTraining complete.")
    wandb.run.summary.update({
        'final_epoch': epoch+1,
        'best_val_loss': best_val_loss
    })
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Offline Teacher model.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick check to see if model and data load.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory specified in the config.")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['smoke_test'] = args.smoke_test

    # Override data_dir if provided
    if args.data_dir:
        config['data_dir'] = args.data_dir
        
    main(config=config) 