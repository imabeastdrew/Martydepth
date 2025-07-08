#!/usr/bin/env python3
"""
Enhanced training script for balanced chord prediction
Addresses class imbalance and diversity issues
"""

import os
import sys
import torch
import torch.nn as nn
import wandb
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.online_transformer import OnlineTransformer
from src.data.dataset import create_dataloader
from src.training.losses import (
    create_loss_function, analyze_prediction_diversity, MixedLossTrainer
)
from src.training.sampling import AdvancedSampler, create_sampler
from src.evaluation.training_diagnostics import quick_model_diagnostic
from src.data.analyze_token_distribution import analyze_chord_distribution, calculate_class_weights

class BalancedTrainer:
    """Enhanced trainer with diversity monitoring and balanced loss functions."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.step = 0
        self.best_eval_loss = float('inf')
        
        # Initialize model
        self.model = self._create_model()
        
        # Analyze data distribution and create loss function
        self.token_counts = self._analyze_data_distribution()
        self.criterion = self._create_loss_function()
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create samplers for evaluation
        self.sampler = self._create_sampler()
        
        # Metrics tracking
        self.diversity_history = []
        self.loss_history = []
        
    def _create_model(self) -> OnlineTransformer:
        """Create and initialize the model."""
        model_config = self.config['model']
        
        model = OnlineTransformer(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            n_layers=model_config['n_layers'],
            sequence_length=model_config['sequence_length'],
            dropout=model_config['dropout']
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return model
    
    def _analyze_data_distribution(self) -> Optional[Dict[int, int]]:
        """Analyze training data distribution for class weights."""
        print("🔍 Analyzing data distribution...")
        
        try:
            data_dir = Path(self.config['data']['data_dir'])
            results = analyze_chord_distribution(data_dir, split="train", max_files=2000)
            
            token_counts = results['token_counts']
            stats = results['distribution_stats']
            
            print(f"   Found {len(token_counts)} unique chord tokens")
            print(f"   Dominance ratio: {stats['dominance_ratio']:.3f}")
            print(f"   Gini coefficient: {stats['gini_coefficient']:.3f}")
            
            if stats['dominance_ratio'] > 0.5:
                print("   ⚠️  HIGH DOMINANCE detected - using strong balancing")
            elif stats['dominance_ratio'] > 0.3:
                print("   ⚠️  MODERATE DOMINANCE detected - using balanced training")
            else:
                print("   ✓  Reasonable distribution detected")
            
            return token_counts
            
        except Exception as e:
            print(f"   ⚠️  Could not analyze distribution: {e}")
            print("   Using standard loss function")
            return None
    
    def _create_loss_function(self):
        """Create loss function based on configuration and data analysis."""
        loss_config = self.config['loss']
        vocab_size = self.config['model']['vocab_size']
        ignore_index = loss_config.get('ignore_index', -1)
        
        print(f"🎯 Creating loss function: {loss_config['type']}")
        
        return create_loss_function(
            loss_config=loss_config,
            vocab_size=vocab_size,
            token_counts=self.token_counts,
            ignore_index=ignore_index,
            device=self.device
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        train_config = self.config['training']
        
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            betas=(0.9, 0.95),  # Slightly adjusted betas
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        train_config = self.config['training']
        scheduler_config = train_config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine_with_warmup':
            from transformers import get_cosine_schedule_with_warmup
            
            # Estimate total steps
            # This is approximate - in practice you'd calculate from dataloader
            estimated_steps_per_epoch = 1000  # Rough estimate
            total_steps = estimated_steps_per_epoch * train_config['max_epochs']
            warmup_steps = int(total_steps * scheduler_config.get('warmup_ratio', 0.1))
            
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # Fallback to simple step scheduler
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=1000, 
                gamma=0.95
            )
    
    def _create_sampler(self) -> AdvancedSampler:
        """Create advanced sampler for evaluation."""
        sampling_config = self.config.get('sampling', {})
        
        strategies = {
            'nucleus': {
                'p': sampling_config.get('top_p', 0.9),
                'temperature': sampling_config.get('temperature', 1.0)
            },
            'top_k': {
                'k': sampling_config.get('top_k', 50),
                'temperature': sampling_config.get('temperature', 1.0)
            }
        }
        
        return AdvancedSampler(strategies=strategies)
    
    def train_step(self, batch: Dict) -> Tuple[float, Dict[str, float]]:
        """Single training step with diversity monitoring."""
        self.model.train()
        
        input_tokens = batch['input_tokens'].to(self.device)
        target_tokens = batch['target_tokens'].to(self.device)
        
        # Forward pass
        logits = self.model(input_tokens)
        
        # Calculate loss
        if isinstance(self.criterion, MixedLossTrainer):
            loss, loss_metrics = self.criterion.compute_loss(logits, target_tokens, self.step)
        else:
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
            loss_metrics = {'loss': loss.item()}
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config['training']['grad_clip_norm']
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Add gradient norm to metrics
        loss_metrics['grad_norm'] = grad_norm.item()
        loss_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return loss.item(), loss_metrics
    
    def evaluate_step(self, batch: Dict) -> Tuple[float, Dict[str, float]]:
        """Single evaluation step with diversity analysis."""
        self.model.eval()
        
        with torch.no_grad():
            input_tokens = batch['input_tokens'].to(self.device)
            target_tokens = batch['target_tokens'].to(self.device)
            
            logits = self.model(input_tokens)
            
            # Calculate loss
            if isinstance(self.criterion, MixedLossTrainer):
                loss, loss_metrics = self.criterion.compute_loss(logits, target_tokens, self.step)
            else:
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_tokens.view(-1))
                loss_metrics = {'eval_loss': loss.item()}
            
            # Analyze prediction diversity
            diversity_metrics = analyze_prediction_diversity(
                logits, target_tokens, self.config['model']['vocab_size']
            )
            
            # Combine metrics
            all_metrics = {**loss_metrics, **diversity_metrics}
            
            return loss.item(), all_metrics
    
    def train_epoch(self, train_loader, eval_loader, epoch: int):
        """Train for one epoch with periodic evaluation."""
        train_config = self.config['training']
        eval_config = self.config['evaluation']
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss, metrics = self.train_step(batch)
            epoch_loss += loss
            epoch_steps += 1
            self.step += 1
            
            # Log training metrics
            if self.step % self.config['logging']['log_every'] == 0:
                wandb.log({"train/" + k: v for k, v in metrics.items()}, step=self.step)
            
            # Evaluation
            if self.step % eval_config['eval_every'] == 0:
                eval_loss, eval_metrics = self.evaluate(eval_loader, eval_config['eval_steps'])
                
                # Log evaluation metrics
                wandb.log({"eval/" + k: v for k, v in eval_metrics.items()}, step=self.step)
                
                # Check for best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    if eval_config.get('save_best', True):
                        self.save_checkpoint("best_model.pt")
                
                # Diversity monitoring and alerts
                self._monitor_diversity(eval_metrics)
            
            # Checkpointing
            if self.step % self.config['checkpointing']['save_every'] == 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return epoch_loss / epoch_steps
    
    def evaluate(self, eval_loader, max_steps: int = None) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = {}
        step_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if max_steps is not None and batch_idx >= max_steps:
                    break
                
                loss, metrics = self.evaluate_step(batch)
                total_loss += loss
                step_count += 1
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        avg_loss = total_loss / step_count
        
        return avg_loss, avg_metrics
    
    def _monitor_diversity(self, eval_metrics: Dict[str, float]):
        """Monitor diversity metrics and issue alerts."""
        diversity_config = self.config['evaluation']['diversity_metrics']
        
        if not diversity_config.get('enabled', False):
            return
        
        dominance_ratio = eval_metrics.get('dominance_ratio', 0.0)
        vocab_coverage = eval_metrics.get('vocabulary_coverage', 0.0)
        
        # Store history
        self.diversity_history.append({
            'step': self.step,
            'dominance_ratio': dominance_ratio,
            'vocab_coverage': vocab_coverage
        })
        
        # Check thresholds and issue alerts
        dominance_threshold = diversity_config.get('dominance_threshold', 0.3)
        coverage_threshold = diversity_config.get('vocab_coverage_threshold', 0.1)
        
        if dominance_ratio > dominance_threshold:
            print(f"⚠️  HIGH DOMINANCE ALERT: {dominance_ratio:.3f} > {dominance_threshold}")
            wandb.log({"alerts/high_dominance": 1}, step=self.step)
        
        if vocab_coverage < coverage_threshold:
            print(f"⚠️  LOW COVERAGE ALERT: {vocab_coverage:.3f} < {coverage_threshold}")
            wandb.log({"alerts/low_coverage": 1}, step=self.step)
        
        # Log diversity trends
        if len(self.diversity_history) >= 5:
            recent_dominance = [h['dominance_ratio'] for h in self.diversity_history[-5:]]
            dominance_trend = np.mean(recent_dominance[-2:]) - np.mean(recent_dominance[:3])
            wandb.log({"diversity/dominance_trend": dominance_trend}, step=self.step)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'config': self.config,
            'diversity_history': self.diversity_history
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"💾 Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = Path("checkpoints") / filename
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {filename}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_eval_loss = checkpoint['best_eval_loss']
        self.diversity_history = checkpoint.get('diversity_history', [])
        
        print(f"✅ Loaded checkpoint: {filename} (step {self.step})")
        return True

def load_config(config_path: str) -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_wandb(config: Dict):
    """Initialize Weights & Biases logging."""
    wandb_config = config.get('logging', {}).get('wandb', {})
    
    wandb.init(
        project=wandb_config.get('project', 'martydepth'),
        name=wandb_config.get('name', 'balanced_training'),
        config=config
    )

def main():
    parser = argparse.ArgumentParser(description="Train balanced chord prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting balanced training on {device}")
    print(f"📋 Config: {args.config}")
    
    # Setup logging
    setup_wandb(config)
    
    # Create trainer
    trainer = BalancedTrainer(config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create data loaders
    print("📊 Creating data loaders...")
    data_config = config['data']
    
    train_loader, train_info = create_dataloader(
        data_dir=Path(data_config['data_dir']),
        split="train",
        batch_size=config['training']['batch_size'],
        num_workers=data_config['num_workers'],
        sequence_length=data_config['sequence_length'],
        mode=data_config['mode'],
        shuffle=True,
        pin_memory=data_config.get('pin_memory', True)
    )
    
    eval_loader, eval_info = create_dataloader(
        data_dir=Path(data_config['data_dir']),
        split="valid",
        batch_size=config['training']['batch_size'],
        num_workers=data_config['num_workers'],
        sequence_length=data_config['sequence_length'],
        mode=data_config['mode'],
        shuffle=False,
        pin_memory=data_config.get('pin_memory', True)
    )
    
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Valid: {len(eval_loader)} batches")
    
    # Training loop
    print("🎵 Starting training...")
    
    try:
        for epoch in range(config['training']['max_epochs']):
            avg_loss = trainer.train_epoch(train_loader, eval_loader, epoch)
            
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Log epoch metrics
            wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch}, step=trainer.step)
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise
    finally:
        # Save final checkpoint
        trainer.save_checkpoint("final_model.pt")
        print("💾 Saved final checkpoint")
        
        wandb.finish()

if __name__ == "__main__":
    main() 