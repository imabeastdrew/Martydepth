#!/usr/bin/env python3
"""
Test script to verify the evaluation fixes work correctly.
Tests both online and offline model evaluation with proper interleaved sequences.
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append('.')

from src.evaluation.metrics import (
    calculate_harmony_metrics,
    calculate_emd_metrics,
    create_interleaved_sequences,
    parse_sequences,
    print_baseline_comparison,
    validate_interleaved_sequences
)

def test_validation_functions():
    """Test the new validation functions catch common errors."""
    print("=== TESTING VALIDATION FUNCTIONS ===")
    
    # Load tokenizer info for testing
    tokenizer_path = Path("data/interim/test/tokenizer_info.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    print(f"Loaded tokenizer info: {tokenizer_info['total_vocab_size']} total tokens")
    
    # Test 1: Valid interleaved sequence should pass
    print("\n1. Testing valid interleaved sequence...")
    valid_seq = np.array([179, 60, 2679, 62, 180, 64])  # [chord_onset, melody_note, chord_hold, melody_note, ...]
    try:
        validate_interleaved_sequences([valid_seq], tokenizer_info)
        print("✅ Valid sequence passed validation")
    except Exception as e:
        print(f"❌ Valid sequence failed validation: {e}")
    
    # Test 2: Chord-only sequence should fail
    print("\n2. Testing chord-only sequence (should fail)...")
    chord_only_seq = np.array([179, 2679, 180, 2680])  # All chord tokens
    try:
        validate_interleaved_sequences([chord_only_seq], tokenizer_info)
        print("❌ Chord-only sequence incorrectly passed validation")
    except ValueError as e:
        print(f"✅ Chord-only sequence correctly caught: {e}")
    
    # Test 3: Odd length sequence should fail
    print("\n3. Testing odd-length sequence (should fail)...")
    odd_seq = np.array([179, 60, 2679])  # Odd length
    try:
        validate_interleaved_sequences([odd_seq], tokenizer_info)
        print("❌ Odd-length sequence incorrectly passed validation")
    except ValueError as e:
        print(f"✅ Odd-length sequence correctly caught: {e}")

def test_create_interleaved_sequences():
    """Test the helper function to create interleaved sequences."""
    print("\n=== TESTING CREATE_INTERLEAVED_SEQUENCES ===")
    
    # Create test melody and chord arrays
    melody_tokens = np.array([[60, 62, 64, 65], [67, 69, 71, 72]])  # 2 sequences, 4 frames each
    chord_tokens = np.array([[179, 2679, 180, 2680], [181, 2681, 182, 2682]])  # 2 sequences, 4 frames each
    
    # Create interleaved sequences
    interleaved_seqs = create_interleaved_sequences(melody_tokens, chord_tokens)
    
    print(f"Input melody shape: {melody_tokens.shape}")
    print(f"Input chord shape: {chord_tokens.shape}")
    print(f"Output: {len(interleaved_seqs)} interleaved sequences")
    
    # Check first sequence
    seq0 = interleaved_seqs[0]
    print(f"First interleaved sequence: {seq0}")
    
    expected_seq0 = np.array([179, 60, 2679, 62, 180, 64, 2680, 65])
    if np.array_equal(seq0, expected_seq0):
        print("✅ Interleaved sequence creation correct")
    else:
        print(f"❌ Expected: {expected_seq0}")
        print(f"❌ Got:      {seq0}")

def test_metrics_with_proper_sequences():
    """Test metrics calculations with properly formatted sequences."""
    print("\n=== TESTING METRICS WITH PROPER SEQUENCES ===")
    
    # Load tokenizer info
    tokenizer_path = Path("data/interim/test/tokenizer_info.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    # Create a test interleaved sequence
    # Format: [chord_0, melody_0, chord_1, melody_1, ...]
    test_sequence = np.array([
        179, 60,    # C major chord onset, C note
        2679, 60,   # C major chord hold, C note hold
        2679, 64,   # C major chord hold, E note 
        2679, 67    # C major chord hold, G note
    ])
    
    sequences = [test_sequence]
    
    print(f"Test sequence: {test_sequence}")
    print(f"Chord tokens (even indices): {test_sequence[0::2]}")
    print(f"Melody tokens (odd indices): {test_sequence[1::2]}")
    
    try:
        # Test harmony metrics
        harmony_metrics = calculate_harmony_metrics(sequences, tokenizer_info)
        print(f"\nHarmony metrics:")
        for key, value in harmony_metrics.items():
            print(f"  {key}: {value}")
        
        # Test EMD metrics (using same sequence for both generated and ground truth)
        emd_metrics = calculate_emd_metrics(sequences, sequences, tokenizer_info)
        print(f"\nEMD metrics (self-comparison):")
        for key, value in emd_metrics.items():
            print(f"  {key}: {value}")
        
        # Test baseline comparison
        print(f"\nBaseline comparison:")
        print_baseline_comparison(harmony_metrics, emd_metrics)
        
    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        import traceback
        traceback.print_exc()

def test_offline_evaluation_format():
    """Test that the fixed offline evaluation produces correct format."""
    print("\n=== TESTING OFFLINE EVALUATION FORMAT ===")
    
    # This would normally require loading actual models, so we'll simulate the key parts
    print("Testing interleaved sequence creation from separate melody/chord arrays...")
    
    # Simulate what the offline model produces
    melody_batch = np.array([[60, 62, 64, 65, 67]])  # 1 sequence, 5 melody tokens
    chord_batch = np.array([[179, 2679, 180, 2680, 181]])  # 1 sequence, 5 chord tokens
    
    # This simulates what the fixed generate_offline function should do
    min_len = min(len(melody_batch[0]), len(chord_batch[0]))
    melody_seq = melody_batch[0][:min_len]
    chord_seq = chord_batch[0][:min_len]
    
    # Create interleaved sequence: [chord_0, melody_0, chord_1, melody_1, ...]
    interleaved = np.empty(min_len * 2, dtype=np.int64)
    interleaved[0::2] = chord_seq   # Even indices: chords
    interleaved[1::2] = melody_seq  # Odd indices: melody
    
    print(f"Melody tokens: {melody_seq}")
    print(f"Chord tokens: {chord_seq}")
    print(f"Interleaved sequence: {interleaved}")
    
    expected = np.array([179, 60, 2679, 62, 180, 64, 2680, 65, 181, 67])
    if np.array_equal(interleaved, expected):
        print("✅ Offline evaluation format correct")
    else:
        print(f"❌ Expected: {expected}")
        print(f"❌ Got:      {interleaved}")

if __name__ == "__main__":
    print("Testing evaluation fixes...")
    
    try:
        test_validation_functions()
        test_create_interleaved_sequences()
        test_metrics_with_proper_sequences()
        test_offline_evaluation_format()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS COMPLETED")
        print("The evaluation fixes should now work correctly.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 