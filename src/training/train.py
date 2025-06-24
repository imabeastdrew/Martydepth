#!/usr/bin/env python3
"""
Training script for the Online Transformer model.
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import wandb
import yaml
import json
from transformers import Adafactor
from torch.optim.lr_scheduler import LambdaLR

from src.models.online_transformer import OnlineTransformer
from src.data.dataset import create_dataloader
from src.training.utils.schedulers import get_warmup_schedule
from src.training.utils.logging import log_model_artifact

def main(config: dict):
    """Main training function."""
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- W&B Setup ---
    run_name = (
        f"online_L{config['num_layers']}_H{config['num_heads']}"
        f"_D{config['embed_dim']}_seq{config['max_sequence_length']}"
        f"_bs{config['batch_size']}_lr{config['learning_rate']}"
    )

    wandb.init(
        project=config['wandb_project'],
        name=run_name,
        config=config,
        job_type="online_training"
    )
    
    # --- Data ---
    data_path = Path(config['data_dir'])
    train_loader, tokenizer_info = create_dataloader(
        data_dir=data_path,
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_sequence_length'],
        mode='online'
    )
    valid_loader, _ = create_dataloader(
        data_dir=data_path,
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_sequence_length'],
        mode='online'
    )
    
    config['vocab_size'] = tokenizer_info['total_vocab_size']
    
    # --- Model ---
    model = OnlineTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_sequence_length'],
        pad_token_id=tokenizer_info.get('pad_token_id', -100)
    ).to(device)
    
    wandb.watch(model, log='all')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- Training Components ---
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_info.get('pad_token_id', -100))
    optimizer = Adafactor(
        model.parameters(), 
        scale_parameter=True, 
        relative_step=True
    )
    total_steps = len(train_loader) * config['max_epochs']
    scheduler = get_warmup_schedule(
        optimizer,
        num_warmup_steps=config['warmup_steps']
    )

    # --- Smoke Test ---
    if config.get('smoke_test', False):
        print("\n--- Smoke test successful: Model and data loaded correctly. ---")
        try:
            batch = next(iter(train_loader))
            input_tokens = batch['input_tokens'].to(device)
            model(input_tokens)
            print("--- Smoke test successful: Single forward pass completed. ---")
        except Exception as e:
            print(f"--- Smoke test failed during forward pass: {e} ---")
        return

    # --- Training Loop ---
    best_val_loss = float('inf')
    global_step = 0

    print(f"\n--- Starting Training ---")
    for epoch in range(config['max_epochs']):
        # --- Training Step ---
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Training]")
        for batch in pbar:
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_tokens, padding_mask=padding_mask)
            loss = loss_fn(logits.view(-1, config['vocab_size']), target_tokens.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])
            optimizer.step()
            
            # The scheduler is still needed to manage the warmup phase with Adafactor
            scheduler.step()

            # When using relative_step, the LR is managed internally by Adafactor
            # We can log the last_lr from the scheduler to see the warmup progress
            lr = scheduler.get_last_lr()[0]
            total_train_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item(), 'lr': lr})
            wandb.log({'train/step_loss': loss.item(), 'train/learning_rate': lr}, step=global_step)

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        total_valid_loss = 0
        nan_batches = 0
        with torch.no_grad():
            pbar_valid = tqdm(valid_loader, desc="Validating")
            for i, batch in enumerate(pbar_valid):
                input_tokens = batch['input_tokens'].to(device)
                target_tokens = batch['target_tokens'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                
                logits = model(input_tokens, padding_mask=padding_mask)
                
                # Check for NaN in logits before loss calculation
                if torch.isnan(logits).any():
                    print(f"\nNaN detected in model logits at validation batch {i}. Skipping loss calculation.")
                    nan_batches += 1
                    continue

                loss = loss_fn(logits.view(-1, config['vocab_size']), target_tokens.view(-1))

                # Check for NaN in the final loss
                if torch.isnan(loss):
                    print(f"\nNaN loss detected at validation batch {i}. Logits min/max: {logits.min()}, {logits.max()}.")
                    nan_batches += 1
                    continue
                
                total_valid_loss += loss.item()
                pbar_valid.set_postfix({'loss': loss.item()})
                
        # Avoid division by zero if all batches were NaN
        num_valid_batches = len(valid_loader) - nan_batches
        avg_valid_loss = total_valid_loss / num_valid_batches if num_valid_batches > 0 else float('nan')
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'valid/epoch_loss': avg_valid_loss,
            'epoch': epoch + 1
        }, step=global_step)
        
        # --- Checkpoint ---
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            
            checkpoint_dir = Path(config['checkpoint_dir']) / run_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the existing logger for artifacts
            log_model_artifact(
                model=model,
                name=f"{run_name}-epoch-{epoch+1}",
                tokenizer_info=tokenizer_info,
                metadata={"val_loss": best_val_loss, "epoch": epoch+1, **config}
            )
            
    wandb.finish()
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the OnlineTransformer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick check to see if model and data load.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory specified in the config.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['smoke_test'] = args.smoke_test

    # Override data_dir if provided
    if args.data_dir:
        config['data_dir'] = args.data_dir

    if 'feedforward_dim' not in config:
        config['feedforward_dim'] = 4 * config['embed_dim']
        
    main(config) 