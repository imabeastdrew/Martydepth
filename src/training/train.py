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

    wandb.init(
        project=config['wandb_project'],
        name=config.get('wandb_run_name'),
        config=config,
        job_type="online_training"
    )
    run_name = wandb.run.name
    
    # --- Data ---
    data_path = Path(config['data_dir'])
    train_loader = create_dataloader(
        data_dir=data_path,
        split="train",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_sequence_length'],
        mode='online'
    )
    valid_loader = create_dataloader(
        data_dir=data_path,
        split="valid",
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['max_sequence_length'],
        mode='online'
    )
    
    tokenizer_info = train_loader.dataset.tokenizer_info
    config['vocab_size'] = tokenizer_info['total_vocab_size']
    
    # --- Model ---
    model = OnlineTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_sequence_length']
    ).to(device)
    
    wandb.watch(model, log='all')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- Training Components ---
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_info.get('silence_token_idx', -100))
    optimizer = Adafactor(
        model.parameters(), 
        lr=config['learning_rate'], 
        scale_parameter=False, 
        relative_step=False
    )
    total_steps = len(train_loader) * config['max_epochs']
    scheduler = get_warmup_schedule(
        optimizer,
        num_warmup_steps=config['warmup_steps']
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    steps_without_improvement = 0

    print(f"\n--- Starting Training ---")
    for epoch in range(config['max_epochs']):
        model.train()
        total_train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Training]")
        for batch in pbar:
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            optimizer.zero_grad()
            logits = model(input_tokens)
            loss = loss_fn(logits.view(-1, config['vocab_size']), target_tokens.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])
            optimizer.step()
            scheduler.step()
            
            lr = optimizer.param_groups[0]['lr']
            total_train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': lr})
            wandb.log({'train/step_loss': loss.item(), 'train/learning_rate': lr})

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            pbar_valid = tqdm(valid_loader, desc="Validating")
            for batch in pbar_valid:
                input_tokens = batch['input_tokens'].to(device)
                target_tokens = batch['target_tokens'].to(device)
                
                logits = model(input_tokens)
                loss = loss_fn(logits.view(-1, config['vocab_size']), target_tokens.view(-1))
                total_valid_loss += loss.item()
                pbar_valid.set_postfix({'loss': loss.item()})

        avg_valid_loss = total_valid_loss / len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        wandb.log({
            'train/epoch_loss': avg_train_loss,
            'valid/epoch_loss': avg_valid_loss,
            'epoch': epoch + 1
        })
        
        # --- Checkpoint & Early Stopping ---
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            steps_without_improvement = 0
            
            checkpoint_dir = Path(config['checkpoint_dir']) / run_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the existing logger for artifacts
            log_model_artifact(
                model,
                run_name,
                checkpoint_dir,
                tokenizer_info,
                {"val_loss": best_val_loss, "epoch": epoch+1, **config}
            )
            print(f"New best model saved to {checkpoint_dir} with validation loss: {best_val_loss:.4f}")
        else:
            steps_without_improvement += 1
            print(f"Patience: {steps_without_improvement}/{config['early_stopping_patience']}")

        if steps_without_improvement >= config['early_stopping_patience']:
            print(f"Stopping early after {steps_without_improvement} epochs with no improvement.")
            break
            
    wandb.finish()
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the OnlineTransformer model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'feedforward_dim' not in config:
        config['feedforward_dim'] = 4 * config['embed_dim']
        
    main(config) 