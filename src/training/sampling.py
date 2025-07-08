#!/usr/bin/env python3
"""
Advanced sampling functions for diverse text generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict
import math

def temperature_sample(logits: torch.Tensor, temperature: float = 1.0, 
                      return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Sample with temperature scaling."""
    if temperature <= 0:
        # Greedy sampling
        tokens = torch.argmax(logits, dim=-1)
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            return tokens, probs
        return tokens
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    if return_probs:
        return tokens, probs
    return tokens

def nucleus_sample(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0,
                  return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Nucleus (top-p) sampling."""
    # Apply temperature first
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for nucleus
    nucleus_mask = cumulative_probs <= p
    
    # Include at least one token (the most probable one)
    nucleus_mask[..., 0] = True
    
    # Zero out probabilities outside nucleus
    filtered_probs = sorted_probs.clone()
    filtered_probs[~nucleus_mask] = 0.0
    
    # Renormalize
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Sample from filtered distribution
    sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    # Map back to original vocabulary indices
    tokens = sorted_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    if return_probs:
        return tokens, probs
    return tokens

def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0,
                return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Top-k sampling."""
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Get top-k values and indices
    top_k_logits, top_k_indices = torch.topk(scaled_logits, k, dim=-1)
    
    # Convert to probabilities
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from top-k
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
    
    # Map back to original vocabulary
    tokens = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    if return_probs:
        full_probs = F.softmax(scaled_logits, dim=-1)
        return tokens, full_probs
    return tokens

def typical_sample(logits: torch.Tensor, tau: float = 0.95, temperature: float = 1.0,
                  return_probs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Typical sampling based on conditional entropy."""
    # Apply temperature
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1, keepdim=True)
    
    # Calculate log probabilities
    log_probs = torch.log(probs + 1e-10)
    
    # Calculate absolute difference from entropy
    abs_diff = torch.abs(log_probs + entropy)
    
    # Sort by absolute difference (most typical first)
    sorted_diffs, sorted_indices = torch.sort(abs_diff, dim=-1)
    sorted_probs = probs.gather(-1, sorted_indices)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Create mask for typical tokens
    typical_mask = cumulative_probs <= tau
    typical_mask[..., 0] = True  # Include at least one token
    
    # Filter probabilities
    filtered_probs = sorted_probs.clone()
    filtered_probs[~typical_mask] = 0.0
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Sample
    sampled_indices = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    tokens = sorted_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    if return_probs:
        return tokens, probs
    return tokens

def repetition_penalty_logits(logits: torch.Tensor, input_ids: torch.Tensor, 
                             penalty: float = 1.1) -> torch.Tensor:
    """Apply repetition penalty to logits based on input history."""
    if penalty == 1.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Get unique tokens in the input sequence
        unique_tokens = torch.unique(input_ids[i])
        
        # Apply penalty
        for token in unique_tokens:
            if token < vocab_size:
                if penalized_logits[i, token] > 0:
                    penalized_logits[i, token] /= penalty
                else:
                    penalized_logits[i, token] *= penalty
    
    return penalized_logits

def frequency_penalty_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                           penalty: float = 0.1) -> torch.Tensor:
    """Apply frequency-based penalty to reduce repetition."""
    if penalty == 0.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Count token frequencies
        token_counts = torch.bincount(input_ids[i], minlength=vocab_size)
        
        # Apply frequency penalty
        frequency_penalty = penalty * token_counts.float()
        penalized_logits[i] -= frequency_penalty
    
    return penalized_logits

def presence_penalty_logits(logits: torch.Tensor, input_ids: torch.Tensor,
                          penalty: float = 0.1) -> torch.Tensor:
    """Apply presence-based penalty (binary version of frequency penalty)."""
    if penalty == 0.0:
        return logits
    
    batch_size, vocab_size = logits.shape
    penalized_logits = logits.clone()
    
    for i in range(batch_size):
        # Get unique tokens (presence mask)
        unique_tokens = torch.unique(input_ids[i])
        
        # Apply presence penalty
        for token in unique_tokens:
            if token < vocab_size:
                penalized_logits[i, token] -= penalty
    
    return penalized_logits

def contrastive_search(logits: torch.Tensor, input_ids: torch.Tensor, 
                      alpha: float = 0.6, k: int = 4,
                      hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Contrastive search for diverse and coherent generation."""
    batch_size, vocab_size = logits.shape
    
    if k >= vocab_size:
        k = vocab_size - 1
    
    # Get top-k candidates
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    if hidden_states is None or input_ids.size(1) <= 1:
        # Fallback to regular top-k sampling
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        return top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    # Calculate contrastive scores
    scores = []
    
    for i in range(k):
        # Model confidence (probability)
        model_score = top_k_probs[:, i]
        
        # Degeneration penalty (similarity to past hidden states)
        if hidden_states.size(1) > 1:
            current_hidden = hidden_states[:, -1:]  # Last hidden state
            past_hidden = hidden_states[:, :-1]     # Previous hidden states
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                current_hidden.unsqueeze(2), 
                past_hidden.unsqueeze(1), 
                dim=-1
            )
            max_similarity = similarity.max(dim=-1)[0].squeeze(1)
            degeneration_penalty = max_similarity
        else:
            degeneration_penalty = torch.zeros_like(model_score)
        
        # Combine scores
        combined_score = alpha * model_score - (1 - alpha) * degeneration_penalty
        scores.append(combined_score)
    
    # Stack scores and select best
    all_scores = torch.stack(scores, dim=1)
    best_indices = torch.argmax(all_scores, dim=1)
    
    return top_k_indices.gather(-1, best_indices.unsqueeze(-1)).squeeze(-1)

def diverse_beam_sample(logits: torch.Tensor, num_groups: int = 4, 
                       diversity_penalty: float = 0.5, group_size: int = 1) -> torch.Tensor:
    """Diverse beam sampling to encourage different token choices."""
    batch_size, vocab_size = logits.shape
    
    # Simplified version for single token generation
    # In practice, this would maintain beam states across multiple steps
    
    # Split vocabulary into groups
    group_size_actual = vocab_size // num_groups
    
    selected_tokens = []
    
    for batch_idx in range(batch_size):
        batch_logits = logits[batch_idx]
        group_scores = []
        
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size_actual
            end_idx = min((group_idx + 1) * group_size_actual, vocab_size)
            
            # Get group logits
            group_logits = batch_logits[start_idx:end_idx]
            
            # Apply diversity penalty based on previously selected groups
            if group_idx > 0:
                # Simple penalty: reduce scores for tokens in similar ranges
                penalty = diversity_penalty * group_idx
                group_logits = group_logits - penalty
            
            # Find best token in this group
            best_local_idx = torch.argmax(group_logits)
            best_global_idx = start_idx + best_local_idx
            group_scores.append((group_logits[best_local_idx].item(), best_global_idx))
        
        # Select the group with highest score
        best_score, best_token = max(group_scores, key=lambda x: x[0])
        selected_tokens.append(best_token)
    
    return torch.tensor(selected_tokens, device=logits.device)

def mirostat_sample(logits: torch.Tensor, tau: float = 5.0, eta: float = 0.1,
                   state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
    """Mirostat sampling for controlling perplexity."""
    if state is None:
        state = {'surprise': 0.0}
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate surprisal for each token
    surprisal = -torch.log(probs + 1e-10)
    
    # Target surprisal
    target_surprisal = tau
    
    # Adaptive threshold based on recent surprise
    current_surprise = state.get('surprise', 0.0)
    threshold = target_surprisal + eta * (current_surprise - target_surprisal)
    
    # Filter tokens based on surprisal threshold
    mask = surprisal <= threshold
    
    # Ensure at least one token is available
    if not mask.any(dim=-1).all():
        # If no tokens meet threshold, use top-1
        mask = torch.zeros_like(mask, dtype=torch.bool)
        mask.scatter_(-1, torch.argmax(probs, dim=-1, keepdim=True), True)
    
    # Sample from filtered distribution
    filtered_probs = probs * mask.float()
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
    
    # Update surprise estimate
    token_surprisal = surprisal.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    new_surprise = current_surprise + eta * (token_surprisal.mean().item() - current_surprise)
    
    new_state = {'surprise': new_surprise}
    
    return tokens, new_state

def dynamic_temperature_sample(logits: torch.Tensor, base_temperature: float = 1.0,
                             entropy_threshold: float = 5.0, temp_range: Tuple[float, float] = (0.5, 1.5)) -> torch.Tensor:
    """Dynamic temperature sampling based on logits entropy."""
    # Calculate entropy of logits
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Normalize entropy to [0, 1] range
    max_entropy = math.log(logits.size(-1))
    normalized_entropy = entropy / max_entropy
    
    # Adjust temperature based on entropy
    # High entropy (uncertain) -> higher temperature (more random)
    # Low entropy (confident) -> lower temperature (more deterministic)
    min_temp, max_temp = temp_range
    
    # Dynamic temperature calculation
    dynamic_temp = min_temp + (max_temp - min_temp) * normalized_entropy
    
    # Apply temperature and sample
    tokens = []
    for i in range(logits.size(0)):
        temp = dynamic_temp[i].item()
        scaled_logits = logits[i] / temp
        token_probs = F.softmax(scaled_logits, dim=-1)
        token = torch.multinomial(token_probs, num_samples=1)
        tokens.append(token)
    
    return torch.cat(tokens)

def musical_coherence_sample(logits: torch.Tensor, input_ids: torch.Tensor,
                           chord_transition_weights: Optional[torch.Tensor] = None,
                           coherence_weight: float = 0.3) -> torch.Tensor:
    """Musical coherence-aware sampling for chord progressions."""
    if chord_transition_weights is None or input_ids.size(1) == 0:
        # Fallback to nucleus sampling
        return nucleus_sample(logits, p=0.9)
    
    batch_size, vocab_size = logits.shape
    
    # Get the last chord for each sequence
    last_chords = input_ids[:, -1]
    
    # Apply musical transition weights
    coherent_logits = logits.clone()
    
    for i in range(batch_size):
        last_chord = last_chords[i].item()
        if last_chord < chord_transition_weights.size(0):
            # Add transition weights to logits
            transition_bias = chord_transition_weights[last_chord] * coherence_weight
            coherent_logits[i] += transition_bias[:vocab_size]
    
    # Sample using nucleus sampling on modified logits
    return nucleus_sample(coherent_logits, p=0.9)

class AdvancedSampler:
    """Advanced sampler with multiple strategies and adaptive selection."""
    
    def __init__(self, strategies: Dict[str, Dict] = None):
        self.strategies = strategies or {
            'nucleus': {'p': 0.9, 'temperature': 1.0},
            'top_k': {'k': 50, 'temperature': 1.0},
            'typical': {'tau': 0.95, 'temperature': 1.0},
            'mirostat': {'tau': 5.0, 'eta': 0.1}
        }
        self.mirostat_states = {}
        
    def sample(self, logits: torch.Tensor, strategy: str = 'nucleus',
              input_ids: Optional[torch.Tensor] = None,
              batch_idx: Optional[int] = None, **kwargs) -> torch.Tensor:
        """Sample using specified strategy."""
        
        # Apply penalties if input_ids provided
        if input_ids is not None:
            rep_penalty = kwargs.get('repetition_penalty', 1.1)
            freq_penalty = kwargs.get('frequency_penalty', 0.1)
            pres_penalty = kwargs.get('presence_penalty', 0.1)
            
            if rep_penalty != 1.0:
                logits = repetition_penalty_logits(logits, input_ids, rep_penalty)
            if freq_penalty != 0.0:
                logits = frequency_penalty_logits(logits, input_ids, freq_penalty)
            if pres_penalty != 0.0:
                logits = presence_penalty_logits(logits, input_ids, pres_penalty)
        
        # Sample based on strategy
        if strategy == 'nucleus':
            params = self.strategies['nucleus']
            return nucleus_sample(logits, p=params['p'], temperature=params['temperature'])
        
        elif strategy == 'top_k':
            params = self.strategies['top_k']
            return top_k_sample(logits, k=params['k'], temperature=params['temperature'])
        
        elif strategy == 'typical':
            params = self.strategies['typical']
            return typical_sample(logits, tau=params['tau'], temperature=params['temperature'])
        
        elif strategy == 'mirostat':
            params = self.strategies['mirostat']
            state_key = batch_idx if batch_idx is not None else 0
            current_state = self.mirostat_states.get(state_key, None)
            tokens, new_state = mirostat_sample(logits, tau=params['tau'], eta=params['eta'], state=current_state)
            self.mirostat_states[state_key] = new_state
            return tokens
        
        elif strategy == 'dynamic_temp':
            base_temp = kwargs.get('base_temperature', 1.0)
            temp_range = kwargs.get('temp_range', (0.5, 1.5))
            return dynamic_temperature_sample(logits, base_temp, temp_range=temp_range)
        
        elif strategy == 'greedy':
            return torch.argmax(logits, dim=-1)
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def reset_states(self):
        """Reset internal states (useful between sequences)."""
        self.mirostat_states.clear()

def create_sampler(config: Dict) -> AdvancedSampler:
    """Factory function to create samplers from configuration."""
    return AdvancedSampler(strategies=config.get('strategies', None)) 