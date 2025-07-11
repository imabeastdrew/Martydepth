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

from src.models.offline_teacher import OfflineTeacherModel
from src.models.offline_teacher_t5 import T5OfflineTeacherModel
from src.data.dataset import create_dataloader
from src.training.utils.logging import log_model_artifact
from src.training.utils.schedulers import get_warmup_schedule
from src.config.tokenization_config import PAD_TOKEN

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"✅ Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered after {self.counter} epochs without improvement")

def main(config: dict):
    """Main training function."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- W&B Setup ---
    model_type_for_name = config.get('model_type', 'custom')
    run_name = (
        f"offline_{model_type_for_name}_L{config['num_layers']}_H{config['num_heads']}"
        f"_D{config['embed_dim']}_seq{config['max_sequence_length']}"
        f"_bs{config['batch_size']}_lr{config['learning_rate']}"
    )

    wandb.init(
        project=config['wandb_project'],
        name=run_name,
        config=config,
        job_type="offline_training"
    )

    # --- Initialize Early Stopping ---
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 5),
        min_delta=config.get('early_stopping_min_delta', 0.001),
        verbose=True
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
    config['total_vocab_size'] = tokenizer_info['total_vocab_size']
    
    # --- Model Creation (Configurable) ---
    def create_model(model_type: str):
        """Create model based on configuration"""
        if model_type == "custom":
            return OfflineTeacherModel(
                melody_vocab_size=config['melody_vocab_size'],
                chord_vocab_size=config['chord_vocab_size'],
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                max_seq_length=config['max_sequence_length'],
                pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
            )
        elif model_type == "t5":
            return T5OfflineTeacherModel(
                melody_vocab_size=config['melody_vocab_size'],
                chord_vocab_size=config['chord_vocab_size'],
                embed_dim=config['embed_dim'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                max_seq_length=config['max_sequence_length'],
                pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN),
                total_vocab_size=config['total_vocab_size']
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'custom' or 't5'")
    
    model_type = config.get('model_type', 'custom')  # Default to custom model
    model = create_model(model_type).to(device)
    
    print(f"🎵 Using {model_type.upper()} model architecture")

    # Use AdamW optimizer instead of Adafactor for better stability and simpler tuning
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    scheduler = get_warmup_schedule(optimizer, num_warmup_steps=config['warmup_steps'])

    wandb.watch(model, log='all')

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory at start: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

    # --- Smoke Test ---
    if config.get('smoke_test', False):
        print("\n--- Starting Smoke Test ---")
        
        # Create a smaller smoke test model
        smoke_model_type = config.get('model_type', 'custom')
        smoke_model = create_model(smoke_model_type).to(device)
        print(f"   Testing {smoke_model_type.upper()} model architecture")
        
        try:
            print("1. Testing data loading...")
            # Get a single batch
            batch = next(iter(train_loader))
            print(f"   Batch keys: {list(batch.keys())}")
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            print("2. Testing model forward pass...")
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
            
            print("3. Testing optimizer creation...")
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
    print(f"  Early stopping patience: {config.get('early_stopping_patience', 5)}")

    for epoch in range(config['max_epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['max_epochs']} ---")
        
        # GPU memory tracking
        if torch.cuda.is_available():
            print(f"GPU memory at start of epoch: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        
        # Training
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Training")
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
            
            # Use chord vocab size for both models to isolate architectural differences
            vocab_size_for_loss = config['chord_vocab_size']
            
            # Extract chord-only logits for T5 model (which outputs full vocab)
            if model_type == "t5":
                # Chord tokens start at CHORD_TOKEN_START (179) in the full vocabulary
                from src.config.tokenization_config import CHORD_TOKEN_START
                chord_logits = logits[:, :, CHORD_TOKEN_START:]  # [batch, seq, chord_vocab_size]
                logits_for_loss = chord_logits
                
                # Adjust targets from full vocab space to chord-only space
                # Original targets: [179, 180, ..., 4778] -> [0, 1, ..., 4599]
                targets_for_loss = batch['chord_target'] - CHORD_TOKEN_START
                # Handle PAD tokens (they should remain as pad_token_id for ignore_index)
                pad_mask = (batch['chord_target'] == model.pad_token_id)
                targets_for_loss[pad_mask] = model.pad_token_id
            else:  # custom model already outputs chord-only
                logits_for_loss = logits
                targets_for_loss = batch['chord_target']  # Already in chord space
                
            loss = nn.functional.cross_entropy(
                logits_for_loss.reshape(-1, vocab_size_for_loss),
                targets_for_loss.reshape(-1),
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
            pbar_val = tqdm(val_loader, desc="Validating")
            for batch in pbar_val:
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
                
                # Use chord vocab size for both models to isolate architectural differences
                vocab_size_for_loss = config['chord_vocab_size']
                
                # Extract chord-only logits for T5 model (which outputs full vocab)
                if model_type == "t5":
                    # Chord tokens start at CHORD_TOKEN_START (179) in the full vocabulary
                    from src.config.tokenization_config import CHORD_TOKEN_START
                    chord_logits = logits[:, :, CHORD_TOKEN_START:]  # [batch, seq, chord_vocab_size]
                    logits_for_loss = chord_logits
                    
                    # Adjust targets from full vocab space to chord-only space
                    # Original targets: [179, 180, ..., 4778] -> [0, 1, ..., 4599]
                    targets_for_loss = batch['chord_target'] - CHORD_TOKEN_START
                    # Handle PAD tokens (they should remain as pad_token_id for ignore_index)
                    pad_mask = (batch['chord_target'] == model.pad_token_id)
                    targets_for_loss[pad_mask] = model.pad_token_id
                else:  # custom model already outputs chord-only
                    logits_for_loss = logits
                    targets_for_loss = batch['chord_target']  # Already in chord space
                    
                loss = nn.functional.cross_entropy(
                    logits_for_loss.reshape(-1, vocab_size_for_loss),
                    targets_for_loss.reshape(-1),
                    ignore_index=model.pad_token_id
                )
                total_val_loss += loss.item()
                pbar_val.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_val_loss:.4f}")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'valid/epoch_loss': avg_val_loss,
            'epoch': epoch + 1
        }, step=global_step)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"\nSaved checkpoint with validation loss: {avg_val_loss:.4f}")
            log_model_artifact(
                model,
                f"offline_teacher_epoch_{epoch+1}_loss_{avg_val_loss:.4f}",
                tokenizer_info=tokenizer_info,
                metadata={"val_loss": avg_val_loss, "epoch": epoch+1, **config}
            )
        
        # Early stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            break
            
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    wandb.run.summary.update({
        'final_epoch': epoch+1,
        'best_val_loss': best_val_loss,
        'early_stopped': early_stopping.early_stop
    })
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Offline Teacher model.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick check to see if model and data load.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory specified in the config.")
    parser.add_argument("--model_type", type=str, choices=['custom', 't5'], default=None, 
                       help="Override model type: 'custom' for OfflineTeacherModel, 't5' for T5OfflineTeacherModel")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['smoke_test'] = args.smoke_test

    # Override config values if provided
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.model_type:
        config['model_type'] = args.model_type
        
    main(config=config) 