#!/usr/bin/env python3
"""
Debug script for temporal evaluation issues.
This script will help identify problems with the online model and baseline calculation.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
import sys
sys.path.append('.')

from src.data.dataset import create_dataloader
from src.evaluation.evaluate import load_model_from_wandb as load_online_model
from src.evaluation.evaluate_offline import load_model_from_wandb as load_offline_model
from src.evaluation.temporal_evaluation import (
    calculate_harmony_at_timestep,
    calculate_test_set_baseline_temporal,
    extract_scenario_sequences,
    transpose_melody_token
)

def debug_baseline_calculation():
    """Debug the baseline calculation to see what's going wrong."""
    print("=== DEBUGGING BASELINE CALCULATION ===")
    
    # Load config
    with open('src/evaluation/configs/temporal_evaluation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load online model for tokenizer info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    online_model, tokenizer_info, online_config = load_online_model(config['online_artifact'], device)
    
    # Create dataloader
    online_dataloader, _ = create_dataloader(
        data_dir=Path(config['data_dir']),
        split=config['split'],
        batch_size=1,
        num_workers=0,
        sequence_length=512,
        mode='online',
        shuffle=False
    )
    
    print(f"Tokenizer info keys: {list(tokenizer_info.keys())}")
    print(f"Chord token start: {tokenizer_info.get('chord_token_start')}")
    print(f"Total vocab size: {tokenizer_info.get('total_vocab_size')}")
    
    # Test a few batches manually
    harmony_scores = []
    
    for batch_idx, batch in enumerate(online_dataloader):
        if batch_idx >= 3:  # Just test first 3 batches
            break
            
        print(f"\n--- Batch {batch_idx} ---")
        input_tokens = batch['input_tokens']
        print(f"Input tokens shape: {input_tokens.shape}")
        print(f"First 10 tokens: {input_tokens[0, :10].tolist()}")
        
        batch_size, seq_len = input_tokens.shape
        eval_beats = min(32, seq_len // 2)
        print(f"Evaluating {eval_beats} beats from {seq_len} tokens")
        
        batch_harmony_scores = []
        for beat in range(min(5, eval_beats)):  # Just test first 5 beats
            chord_idx = beat * 2
            melody_idx = beat * 2 + 1
            
            if melody_idx < seq_len:
                chord_token = input_tokens[0, chord_idx].item()
                melody_token = input_tokens[0, melody_idx].item()
                
                print(f"  Beat {beat}: chord_token={chord_token}, melody_token={melody_token}")
                
                harmony_score = calculate_harmony_at_timestep(
                    melody_token, chord_token, tokenizer_info
                )
                
                print(f"    Harmony score: {harmony_score}")
                
                if harmony_score is not None:
                    batch_harmony_scores.append(harmony_score)
                else:
                    batch_harmony_scores.append(0.0)
            else:
                batch_harmony_scores.append(0.0)
        
        harmony_scores.append(batch_harmony_scores)
        print(f"  Batch harmony scores: {batch_harmony_scores}")
        print(f"  Mean harmony: {np.mean(batch_harmony_scores):.3f}")
    
    # Calculate overall baseline
    if harmony_scores:
        print(f"\n--- OVERALL BASELINE ---")
        max_length = max(len(seq) for seq in harmony_scores)
        padded_scores = []
        for seq in harmony_scores:
            padded_seq = seq + [0.0] * (max_length - len(seq))
            padded_scores.append(padded_seq)
        
        scores_array = np.array(padded_scores)
        mean_harmony = np.mean(scores_array, axis=0)
        print(f"Mean harmony per beat: {mean_harmony}")
        print(f"Overall mean: {np.mean(mean_harmony):.3f}")
    
    return harmony_scores

def debug_online_model_generation():
    """Debug the online model generation to see what's failing."""
    print("\n=== DEBUGGING ONLINE MODEL GENERATION ===")
    
    # Load config
    with open('src/evaluation/configs/temporal_evaluation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_model, tokenizer_info, online_config = load_online_model(config['online_artifact'], device)
    online_model.eval()
    
    # Create dataloader
    online_dataloader, _ = create_dataloader(
        data_dir=Path(config['data_dir']),
        split=config['split'],
        batch_size=1,
        num_workers=0,
        sequence_length=512,
        mode='online',
        shuffle=False
    )
    
    # Test one batch
    batch = next(iter(online_dataloader))
    input_tokens = batch['input_tokens'].to(device)
    target_tokens = batch['target_tokens'].to(device)
    
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Target tokens shape: {target_tokens.shape}")
    
    # Test scenario extraction
    for scenario in ['primed', 'cold_start', 'perturbed']:
        print(f"\n--- Testing {scenario} scenario ---")
        
        try:
            context_tokens, reference_tokens = extract_scenario_sequences(
                input_tokens, target_tokens, scenario, perturbation_beat=17
            )
            print(f"Context tokens shape: {context_tokens.shape}")
            print(f"Reference tokens shape: {reference_tokens.shape}")
            
            # Test model inference
            with torch.no_grad():
                logits = online_model(context_tokens)
                print(f"Model output shape: {logits.shape}")
                
                # Test chord token filtering
                chord_token_start = tokenizer_info['chord_token_start']
                total_vocab_size = tokenizer_info['total_vocab_size']
                
                next_token_logits = logits[:, -1, :]
                chord_logits = next_token_logits[:, chord_token_start:total_vocab_size]
                print(f"Chord logits shape: {chord_logits.shape}")
                
                # Test sampling
                chord_probs = torch.softmax(chord_logits, dim=-1)
                sampled_indices = torch.multinomial(chord_probs, 1).squeeze(-1)
                sampled_chord_tokens = sampled_indices + chord_token_start
                print(f"Sampled chord token: {sampled_chord_tokens.item()}")
                
        except Exception as e:
            print(f"ERROR in {scenario} scenario: {e}")
            import traceback
            traceback.print_exc()

def debug_transpose_melody_token():
    """Debug the melody transposition function."""
    print("\n=== DEBUGGING MELODY TRANSPOSITION ===")
    
    # Test with some example tokens
    test_tokens = [100, 150, 200, 250, 300]  # Example melody tokens
    
    for token in test_tokens:
        transposed = transpose_melody_token(token, semitones=6)
        print(f"Token {token} -> {transposed} (transposed by +6 semitones)")

def main():
    """Run all debug functions."""
    print("Starting temporal evaluation debugging...")
    
    try:
        debug_baseline_calculation()
    except Exception as e:
        print(f"ERROR in baseline calculation: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        debug_online_model_generation()
    except Exception as e:
        print(f"ERROR in online model generation: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        debug_transpose_melody_token()
    except Exception as e:
        print(f"ERROR in melody transposition: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDebugging complete!")

if __name__ == "__main__":
    main() 