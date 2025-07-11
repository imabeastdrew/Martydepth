#!/usr/bin/env python3
"""
Calculate baseline metrics on test set ground truth data.
This gives us the benchmarks to compare model performance against.
"""

import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from src.data.dataset import create_dataloader
from src.evaluation.metrics import (
    calculate_harmony_metrics,
    calculate_emd_metrics,
    parse_sequences,
)

def get_ground_truth_sequences(dataloader, mode='offline'):
    """Extract ground truth interleaved sequences from dataloader."""
    sequences = []
    
    print(f"Extracting ground truth sequences from {mode} dataloader...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if mode == 'offline':
            # Offline mode has separate melody and chord arrays - this is the CORRECT format
            melody_tokens = batch['melody_tokens'].numpy()
            chord_tokens = batch['chord_target'].numpy()
            
            # DEBUG: Check first batch for PAD tokens
            if batch_idx == 0:
                print(f"\n=== DEBUGGING RAW DATA (first batch) ===")
                for i in range(min(3, len(chord_tokens))):
                    chord_seq = chord_tokens[i]
                    melody_seq = melody_tokens[i]
                    
                    # Count PAD tokens
                    pad_count_chords = np.sum(chord_seq == 178)
                    pad_count_melody = np.sum(melody_seq == 178)
                    
                    # Find non-pad regions
                    non_pad_chord_mask = chord_seq != 178
                    non_pad_melody_mask = melody_seq != 178
                    
                    print(f"\nRaw sequence {i}:")
                    print(f"  Chord length: {len(chord_seq)}, PAD tokens: {pad_count_chords}")
                    print(f"  Melody length: {len(melody_seq)}, PAD tokens: {pad_count_melody}")
                    print(f"  First 10 chords: {chord_seq[:10]}")
                    print(f"  First 10 melody: {melody_seq[:10]}")
                    
                    if pad_count_chords > 0:
                        # Show where PAD tokens appear
                        pad_positions = np.where(chord_seq == 178)[0]
                        print(f"  PAD positions in chords: {pad_positions[:10]}...")
                        
                        # Check if PAD tokens are only at the end (normal) or in the middle (problem)
                        first_pad = pad_positions[0] if len(pad_positions) > 0 else len(chord_seq)
                        continuous_pads = np.all(chord_seq[first_pad:] == 178) if first_pad < len(chord_seq) else True
                        print(f"  PADs only at end: {continuous_pads}")
            
            # Create correctly aligned interleaved sequences: [chord_0, melody_0, chord_1, melody_1, ...]
            for i in range(len(melody_tokens)):
                melody_seq = melody_tokens[i]
                chord_seq = chord_tokens[i]
                
                # CRITICAL FIX: Remove PAD tokens before processing (matching metrics.py fix)
                pad_token_id = 178  # PAD_TOKEN from config
                
                # Find effective sequence length (before padding)
                melody_end = len(melody_seq)
                chord_end = len(chord_seq)
                
                for j in range(len(melody_seq)):
                    if melody_seq[j] == pad_token_id:
                        melody_end = j
                        break
                        
                for j in range(len(chord_seq)):
                    if chord_seq[j] == pad_token_id:
                        chord_end = j
                        break
                
                # Use the shorter of the two non-padded lengths
                effective_len = min(melody_end, chord_end)
                if effective_len == 0:
                    continue  # Skip empty sequences
                    
                melody_seq = melody_seq[:effective_len]
                chord_seq = chord_seq[:effective_len]
                
                # Create interleaved sequence
                interleaved = np.empty(effective_len * 2, dtype=np.int64)
                interleaved[0::2] = chord_seq   # Even indices: chords
                interleaved[1::2] = melody_seq  # Odd indices: melody
                
                sequences.append(interleaved)
        
        elif mode == 'online':
            # Online mode requires reconstruction to proper interleaved format
            input_tokens = batch['input_tokens'].numpy()   # [chord_0, melody_0, chord_1, ..., chord_255]
            target_tokens = batch['target_tokens'].numpy() # [melody_0, chord_1, melody_1, ..., melody_255]
            
            for i in range(len(input_tokens)):
                input_seq = input_tokens[i]    # 511 tokens
                target_seq = target_tokens[i]  # 511 tokens
                
                # Reconstruct proper interleaved format: [chord_0, melody_0, chord_1, melody_1, ...]
                # input_seq[0] = chord_0, target_seq[0] = melody_0
                reconstructed = np.empty(512, dtype=np.int64)
                
                # Start with chord_0, melody_0
                reconstructed[0] = input_seq[0]    # chord_0
                reconstructed[1] = target_seq[0]   # melody_0
                
                # Continue with alternating pattern
                for j in range(1, 256):  # 255 more pairs
                    if j < len(input_seq) and (j*2 - 1) < len(target_seq):
                        reconstructed[j*2] = target_seq[j*2 - 1]     # chord_j from target (shifted back)
                        reconstructed[j*2 + 1] = input_seq[j]        # melody_j from input
                
                sequences.append(reconstructed)
    
    print(f"Extracted {len(sequences)} sequences")
    return sequences

def debug_parse_sequences(sequences, tokenizer_info, max_debug=3):
    """Debug the parse_sequences function to identify issues."""
    print(f"\n=== DEBUGGING PARSE_SEQUENCES (showing first {max_debug} sequences) ===")
    
    parsed_data = parse_sequences(sequences[:max_debug], tokenizer_info)
    
    for i, (seq, data) in enumerate(zip(sequences[:max_debug], parsed_data)):
        print(f"\nSequence {i}:")
        print(f"  Raw length: {len(seq)}")
        print(f"  Parsed notes: {len(data['notes'])}")
        print(f"  Parsed chords: {len(data['chords'])}")
        
        # Show first few tokens
        print(f"  First 10 tokens: {seq[:10]}")
        
        # Check for reasonable token ranges
        chord_tokens = seq[0::2]  # Even indices should be chords
        melody_tokens = seq[1::2]  # Odd indices should be melody
        
        chord_range = (chord_tokens.min(), chord_tokens.max())
        melody_range = (melody_tokens.min(), melody_tokens.max())
        
        print(f"  Chord token range: {chord_range}")
        print(f"  Melody token range: {melody_range}")
        
        # Check if tokens are in expected ranges
        chord_token_start = tokenizer_info['chord_token_start']
        melody_vocab_size = tokenizer_info['melody_vocab_size']
        
        valid_chords = np.sum((chord_tokens >= chord_token_start) | (chord_tokens == 0))  # 0 might be silence
        valid_melody = np.sum(melody_tokens < chord_token_start)
        
        print(f"  Valid chord tokens: {valid_chords}/{len(chord_tokens)}")
        print(f"  Valid melody tokens: {valid_melody}/{len(melody_tokens)}")

def calculate_test_set_baselines():
    """Calculate baseline metrics on test set ground truth data."""
    
    # Configuration
    data_dir = Path("data/interim")
    tokenizer_path = data_dir / "test" / "tokenizer_info.json"
    
    # Load tokenizer info
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    print(f"Loaded tokenizer info: {tokenizer_info['total_vocab_size']} total tokens")
    print(f"Chord token start: {tokenizer_info['chord_token_start']}")
    print(f"Melody vocab size: {tokenizer_info['melody_vocab_size']}")
    
    print("\n=== CALCULATING TEST SET BASELINES ===")
    
    # Use offline dataloader to get correctly aligned ground truth
    print("\nUsing offline dataloader for correct token alignment...")
    offline_loader, _ = create_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=64,
        num_workers=0,
        sequence_length=256,
        mode='offline',
        shuffle=False
    )
    
    # Extract ground truth sequences
    ground_truth_sequences = get_ground_truth_sequences(offline_loader, mode='offline')
    
    # Debug the parsing to check for issues
    debug_parse_sequences(ground_truth_sequences, tokenizer_info, max_debug=3)
    
    # Calculate harmony metrics on ground truth
    print(f"\nCalculating harmony metrics on {len(ground_truth_sequences)} sequences...")
    harmony_metrics = calculate_harmony_metrics(ground_truth_sequences, tokenizer_info)
    
    print("Harmony metrics:")
    for metric, value in harmony_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # For EMD baseline, compare subsets of the test set
    print(f"\nCalculating EMD metrics...")
    mid_point = len(ground_truth_sequences) // 2
    set1 = ground_truth_sequences[:mid_point]
    set2 = ground_truth_sequences[mid_point:mid_point*2]  # Same size
    
    print(f"Comparing {len(set1)} vs {len(set2)} sequences for EMD...")
    emd_metrics = calculate_emd_metrics(set1, set2, tokenizer_info)
    
    print("EMD metrics (internal test set variation):")
    for metric, value in emd_metrics.items():
        if np.isnan(value):
            print(f"  {metric}: NaN (investigate!)")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Perfect self-comparison (should give EMD ≈ 0)
    print(f"\nCalculating self-comparison EMD (should be ~0)...")
    sample_size = min(500, len(ground_truth_sequences))
    sample_seqs = ground_truth_sequences[:sample_size]
    
    perfect_emd = calculate_emd_metrics(sample_seqs, sample_seqs, tokenizer_info)
    
    print("Perfect baseline (self-comparison):")
    for metric, value in perfect_emd.items():
        if np.isnan(value):
            print(f"  {metric}: NaN (this indicates a bug in EMD calculation!)")
        else:
            print(f"  {metric}: {value:.4f}")
    
    # Summary
    print("\n=== TEST SET BASELINE SUMMARY ===")
    print("Ground truth metrics from test set:")
    print(f"  Harmony ratio: {harmony_metrics['melody_note_in_chord_ratio']:.2f}%")
    print(f"  Total frames analyzed: {harmony_metrics['total_frames_analyzed']}")
    print(f"  Chord length entropy: {emd_metrics['chord_length_entropy']:.4f}")
    print(f"  Onset interval EMD (internal variation): {emd_metrics['onset_interval_emd']:.4f}")
    
    if np.isnan(perfect_emd['onset_interval_emd']):
        print(f"  ⚠️  Self-comparison EMD is NaN - this indicates a bug in the EMD calculation!")
    else:
        print(f"  Self-comparison EMD: {perfect_emd['onset_interval_emd']:.4f}")
    
    return {
        'ground_truth_harmony': harmony_metrics,
        'internal_variation_emd': emd_metrics,
        'perfect_baseline_emd': perfect_emd
    }

if __name__ == "__main__":
    baselines = calculate_test_set_baselines() 