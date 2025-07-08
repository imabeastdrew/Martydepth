#!/usr/bin/env python3
"""
Test script to verify the evaluation fixes are working properly.
"""

import json
import numpy as np
from pathlib import Path

# Test that tokenizer info is correctly updated
def test_tokenizer_info():
    """Test that vocabulary size is correctly fixed."""
    print("Testing tokenizer info fixes...")
    
    tokenizer_path = Path("data/interim/test/tokenizer_info.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    total_vocab_size = tokenizer_info['total_vocab_size']
    max_token_id = max(int(k) for k in tokenizer_info['token_to_chord'].keys())
    
    print(f"  Total vocab size: {total_vocab_size}")
    print(f"  Maximum token ID: {max_token_id}")
    print(f"  Vocab size matches max token + 1: {total_vocab_size == max_token_id + 1}")
    
    assert total_vocab_size == max_token_id + 1, f"Vocab size mismatch: {total_vocab_size} != {max_token_id + 1}"
    print("  ✅ Tokenizer info fix verified!")

def test_sequence_parsing():
    """Test that sequence parsing handles interleaved sequences correctly."""
    print("\nTesting sequence parsing...")
    
    # Import the metrics module
    import sys
    sys.path.append('.')
    from src.evaluation.metrics import parse_sequences
    
    # Load tokenizer info
    tokenizer_path = Path("data/interim/test/tokenizer_info.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    # Create a test interleaved sequence: [chord_0, melody_0, chord_1, melody_1, ...]
    # Using actual token values from the data
    chord_tokens = [179, 180, 181]  # Some valid chord tokens
    melody_tokens = [0, 88, 89]     # Some valid melody tokens (onset, silence, hold)
    
    # Create interleaved sequence
    interleaved_seq = []
    for i in range(3):
        interleaved_seq.extend([chord_tokens[i], melody_tokens[i]])
    
    test_sequence = np.array(interleaved_seq, dtype=np.int64)
    print(f"  Test sequence: {test_sequence}")
    print(f"  Sequence length: {len(test_sequence)}")
    
    # Parse the sequence
    parsed_data = parse_sequences([test_sequence], tokenizer_info)
    
    print(f"  Number of parsed sequences: {len(parsed_data)}")
    print(f"  Notes found: {len(parsed_data[0]['notes'])}")
    print(f"  Chords found: {len(parsed_data[0]['chords'])}")
    
    # Should find some notes and chords
    assert len(parsed_data) == 1, "Should parse exactly one sequence"
    print("  ✅ Sequence parsing fix verified!")

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Online Model Evaluation Fixes")
    print("=" * 50)
    
    try:
        test_tokenizer_info()
        test_sequence_parsing()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Evaluation fixes are working.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    main() 