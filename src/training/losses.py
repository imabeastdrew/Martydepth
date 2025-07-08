#!/usr/bin/env python3
"""
Advanced loss functions and training utilities for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import math

class WeightedCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss with class weights to handle imbalance."""
    
    def __init__(self, weights: torch.Tensor, ignore_index: int = -1, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weights', weights)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, 
            targets, 
            weight=self.weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            focal_loss = focal_loss * mask
            return focal_loss.sum() / mask.sum()
        else:
            return focal_loss.mean()

class DiversityRegularizedLoss(nn.Module):
    """Loss with diversity regularization to encourage varied predictions."""
    
    def __init__(self, base_criterion: nn.Module, diversity_weight: float = 0.1, 
                 temporal_diversity_weight: float = 0.05):
        super().__init__()
        self.base_criterion = base_criterion
        self.diversity_weight = diversity_weight
        self.temporal_diversity_weight = temporal_diversity_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        # Base loss
        base_loss = self.base_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Diversity regularization
        probs = F.softmax(logits, dim=-1)
        
        # 1. Encourage uniform distribution over vocabulary (entropy regularization)
        entropy_per_position = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        diversity_loss = -entropy_per_position.mean()  # Negative because we want high entropy
        
        # 2. Temporal diversity - encourage different predictions across time steps
        if logits.size(1) > 1:
            # Compare adjacent time steps
            prob_diff = torch.abs(probs[:, 1:] - probs[:, :-1]).sum(dim=-1)
            temporal_loss = -prob_diff.mean()  # Negative because we want high differences
        else:
            temporal_loss = torch.tensor(0.0, device=logits.device)
        
        total_loss = (base_loss + 
                     self.diversity_weight * diversity_loss + 
                     self.temporal_diversity_weight * temporal_loss)
        
        metrics = {
            'base_loss': base_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'avg_entropy': entropy_per_position.mean().item()
        }
        
        return total_loss, metrics

class ContrastiveDiversityLoss(nn.Module):
    """Contrastive loss to push away from repetitive patterns."""
    
    def __init__(self, base_criterion: nn.Module, contrastive_weight: float = 0.1, 
                 temperature: float = 0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        base_loss = self.base_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        batch_size, seq_len, vocab_size = logits.shape
        
        if seq_len <= 1:
            return base_loss, {'base_loss': base_loss.item(), 'contrastive_loss': 0.0}
        
        # Get embeddings for each position
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Create contrastive pairs - positive pairs are distant in time, negative are close
        contrastive_loss = 0.0
        num_pairs = 0
        
        for i in range(seq_len - 2):
            # Positive: far apart in time (should be different)
            if i + 4 < seq_len:
                pos_sim = F.cosine_similarity(probs[:, i], probs[:, i + 4], dim=-1)
                # We want low similarity for positive pairs (different chords)
                contrastive_loss += pos_sim.mean()
                num_pairs += 1
            
            # Negative: close in time (often should be similar for musical coherence)
            neg_sim = F.cosine_similarity(probs[:, i], probs[:, i + 1], dim=-1)
            # We want high similarity for negative pairs (musical coherence)
            contrastive_loss -= neg_sim.mean()
            num_pairs += 1
        
        if num_pairs > 0:
            contrastive_loss /= num_pairs
        
        total_loss = base_loss + self.contrastive_weight * contrastive_loss
        
        metrics = {
            'base_loss': base_loss.item(),
            'contrastive_loss': contrastive_loss.item() if num_pairs > 0 else 0.0
        }
        
        return total_loss, metrics

def create_class_weights(token_counts: Dict[int, int], vocab_size: int, 
                        method: str = "sqrt_inverse", 
                        device: torch.device = None) -> torch.Tensor:
    """Create class weights tensor from token count statistics."""
    
    if not token_counts:
        raise ValueError("Empty token_counts provided")
    
    total_samples = sum(token_counts.values())
    unique_classes = len(token_counts)
    
    # Initialize weights tensor for full vocabulary
    weights = torch.ones(vocab_size, device=device)
    
    if method == "inverse_frequency":
        for token, count in token_counts.items():
            if token < vocab_size:  # Only set weights for valid tokens
                weights[token] = total_samples / (unique_classes * count)
    elif method == "sqrt_inverse":
        for token, count in token_counts.items():
            if token < vocab_size:  # Only set weights for valid tokens
                weights[token] = math.sqrt(total_samples / count)
    elif method == "log_inverse":
        for token, count in token_counts.items():
            if token < vocab_size:  # Only set weights for valid tokens
                weights[token] = math.log(total_samples / count)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to have mean of 1.0 for the classes that appear
    appearing_tokens = [token for token in token_counts.keys() if token < vocab_size]
    if appearing_tokens:
        weight_values = [weights[token].item() for token in appearing_tokens]
        mean_weight = sum(weight_values) / len(weight_values)
        
        # Only normalize the weights for tokens that actually appear
        for token in appearing_tokens:
            weights[token] /= mean_weight
    
    return weights

class MixedLossTrainer:
    """Trainer that combines multiple loss functions with adaptive weighting."""
    
    def __init__(self, base_criterion: nn.Module, 
                 diversity_weight: float = 0.1,
                 focal_weight: float = 0.0,
                 contrastive_weight: float = 0.0,
                 adaptive_weighting: bool = True):
        self.base_criterion = base_criterion
        self.diversity_weight = diversity_weight
        self.focal_weight = focal_weight
        self.contrastive_weight = contrastive_weight
        self.adaptive_weighting = adaptive_weighting
        
        # Initialize additional losses if needed
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0) if focal_weight > 0 else None
        self.diversity_loss = DiversityRegularizedLoss(base_criterion, diversity_weight) if diversity_weight > 0 else None
        self.contrastive_loss = ContrastiveDiversityLoss(base_criterion, contrastive_weight) if contrastive_weight > 0 else None
        
        # For adaptive weighting
        self.loss_history = {
            'base': [],
            'diversity': [],
            'focal': [],
            'contrastive': []
        }
        
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                    step: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss with multiple regularization terms."""
        
        total_loss = torch.tensor(0.0, device=logits.device)
        metrics = {}
        
        # Base loss
        base_loss = self.base_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += base_loss
        metrics['base_loss'] = base_loss.item()
        
        # Focal loss
        if self.focal_loss is not None and self.focal_weight > 0:
            focal_loss = self.focal_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
            weight = self.focal_weight
            if self.adaptive_weighting and step > 100:
                weight = self._get_adaptive_weight('focal', focal_loss.item(), step)
            total_loss += weight * focal_loss
            metrics['focal_loss'] = focal_loss.item()
            metrics['focal_weight'] = weight
        
        # Diversity loss
        if self.diversity_loss is not None and self.diversity_weight > 0:
            _, div_metrics = self.diversity_loss(logits, targets)
            diversity_loss = div_metrics['diversity_loss']
            weight = self.diversity_weight
            if self.adaptive_weighting and step > 100:
                weight = self._get_adaptive_weight('diversity', diversity_loss, step)
            total_loss += weight * diversity_loss
            metrics.update(div_metrics)
            metrics['diversity_weight'] = weight
        
        # Contrastive loss
        if self.contrastive_loss is not None and self.contrastive_weight > 0:
            _, cont_metrics = self.contrastive_loss(logits, targets)
            contrastive_loss = cont_metrics['contrastive_loss']
            weight = self.contrastive_weight
            if self.adaptive_weighting and step > 100:
                weight = self._get_adaptive_weight('contrastive', contrastive_loss, step)
            total_loss += weight * contrastive_loss
            metrics.update(cont_metrics)
            metrics['contrastive_weight'] = weight
        
        metrics['total_loss'] = total_loss.item()
        return total_loss, metrics
    
    def _get_adaptive_weight(self, loss_type: str, current_loss: float, step: int) -> float:
        """Adaptively adjust loss weights based on training progress."""
        history = self.loss_history[loss_type]
        history.append(current_loss)
        
        # Keep only recent history
        if len(history) > 100:
            history.pop(0)
        
        if len(history) < 10:
            return getattr(self, f'{loss_type}_weight')
        
        # If loss is not decreasing, increase weight
        recent_avg = sum(history[-10:]) / 10
        older_avg = sum(history[-20:-10]) / 10 if len(history) >= 20 else recent_avg
        
        base_weight = getattr(self, f'{loss_type}_weight')
        
        if recent_avg > older_avg * 1.1:  # Loss increasing
            return min(base_weight * 1.5, base_weight * 3.0)
        elif recent_avg < older_avg * 0.9:  # Loss decreasing well
            return max(base_weight * 0.8, base_weight * 0.3)
        else:
            return base_weight

def create_loss_function(loss_config: Dict, vocab_size: int, 
                        token_counts: Optional[Dict[int, int]] = None,
                        ignore_index: int = -1,
                        device: torch.device = None) -> nn.Module:
    """Factory function to create loss functions based on configuration."""
    
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
    
    elif loss_type == 'weighted_cross_entropy':
        if token_counts is None:
            raise ValueError("token_counts required for weighted_cross_entropy")
        
        weights = create_class_weights(
            token_counts, 
            vocab_size,
            method=loss_config.get('weight_method', 'sqrt_inverse'),
            device=device
        )
        
        return WeightedCrossEntropyLoss(
            weights=weights,
            ignore_index=ignore_index,
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('focal_alpha', 1.0),
            gamma=loss_config.get('focal_gamma', 2.0),
            ignore_index=ignore_index
        )
    
    elif loss_type == 'mixed':
        # Create base criterion
        if token_counts is not None and loss_config.get('use_weighted_base', True):
            weights = create_class_weights(
                token_counts,
                vocab_size,
                method=loss_config.get('weight_method', 'sqrt_inverse'),
                device=device
            )
            base_criterion = WeightedCrossEntropyLoss(
                weights=weights,
                ignore_index=ignore_index,
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        else:
            base_criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                label_smoothing=loss_config.get('label_smoothing', 0.0)
            )
        
        return MixedLossTrainer(
            base_criterion=base_criterion,
            diversity_weight=loss_config.get('diversity_weight', 0.1),
            focal_weight=loss_config.get('focal_weight', 0.0),
            contrastive_weight=loss_config.get('contrastive_weight', 0.0),
            adaptive_weighting=loss_config.get('adaptive_weighting', True)
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def analyze_prediction_diversity(logits: torch.Tensor, targets: torch.Tensor, 
                               vocabulary_size: int) -> Dict[str, float]:
    """Analyze diversity of model predictions during training."""
    
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        # Calculate metrics
        batch_size, seq_len = predictions.shape
        
        # Unique predictions per sequence
        unique_per_seq = []
        for i in range(batch_size):
            unique_count = len(torch.unique(predictions[i]))
            unique_per_seq.append(unique_count)
        
        avg_unique = sum(unique_per_seq) / len(unique_per_seq)
        
        # Overall unique predictions
        all_predictions = predictions.flatten()
        total_unique = len(torch.unique(all_predictions))
        
        # Entropy of prediction distribution
        pred_counts = torch.bincount(all_predictions, minlength=vocabulary_size)
        pred_probs = pred_counts.float() / pred_counts.sum()
        pred_probs = pred_probs[pred_probs > 0]  # Remove zero probabilities
        entropy = -torch.sum(pred_probs * torch.log(pred_probs)).item()
        
        # Most common prediction ratio
        most_common_count = pred_counts.max().item()
        dominance_ratio = most_common_count / len(all_predictions)
        
        # Average confidence
        max_probs = torch.max(probs, dim=-1)[0]
        avg_confidence = max_probs.mean().item()
        
        return {
            'avg_unique_per_seq': avg_unique,
            'total_unique': total_unique,
            'vocabulary_coverage': total_unique / vocabulary_size,
            'prediction_entropy': entropy,
            'dominance_ratio': dominance_ratio,
            'avg_confidence': avg_confidence
        } 