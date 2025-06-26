#!/usr/bin/env python3
"""
Test script to verify preprocessing fixes
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocess_frames import FramePreprocessor, validate_sequence
from src.config.tokenization_config import *

def test_vocabulary_ranges():
    """Test that vocabulary ranges don't overlap"""
    print("Testing vocabulary ranges...")
    
    print(f"Melody tokens: [0, {MELODY_VOCAB_SIZE-1}]")
    print(f"PAD token: {PAD_TOKEN}")
    print(f"Chord tokens: [{CHORD_TOKEN_START}, {CHORD_ONSET_HOLD_START-1}] (silence)")
    print(f"Chord onset/hold: [{CHORD_ONSET_HOLD_START}, ...] (dynamic)")
    
    # Check no overlaps
    assert PAD_TOKEN >= MELODY_VOCAB_SIZE, f"PAD_TOKEN ({PAD_TOKEN}) overlaps with melody range [0, {MELODY_VOCAB_SIZE-1}]"
    assert CHORD_TOKEN_START > PAD_TOKEN, f"Chord tokens ({CHORD_TOKEN_START}) overlap with PAD token ({PAD_TOKEN})"
    print("✅ No vocabulary overlaps")

def test_midi_conversion():
    """Test MIDI number conversion functions"""
    print("\nTesting MIDI conversion...")
    
    # Test edge cases
    assert midi_to_token_index(-27) == 0
    assert midi_to_token_index(60) == 87
    assert token_index_to_midi(0) == -27
    assert token_index_to_midi(87) == 60
    
    # Test round-trip conversion
    for midi in [-27, 0, 30, 60]:
        idx = midi_to_token_index(midi)
        back = token_index_to_midi(idx)
        assert back == midi, f"Round-trip failed: {midi} -> {idx} -> {back}"
    
    print("✅ MIDI conversion functions work correctly")

def test_small_sample():
    """Test preprocessing on a small sample"""
    print("\nTesting preprocessing on sample data...")
    
    # Load a small sample of raw data
    raw_data_path = project_root / "data" / "raw" / "Hooktheory copy.json"
    
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Test on first 5 valid songs
    preprocessor = FramePreprocessor(sequence_length=256)
    test_count = 0
    validation_stats = []
    
    for song_id, song_data in raw_data.items():
        if test_count >= 5:
            break
            
        if 'MELODY' not in song_data.get('tags', []) or 'HARMONY' not in song_data.get('tags', []):
            continue
            
        sequences = preprocessor.process_song(song_id, song_data)
        if not sequences:
            continue
            
        test_count += 1
        print(f"  Testing song {song_id}: {len(sequences)} sequences")
        
        for i, seq in enumerate(sequences):
            validation = validate_sequence(seq)
            validation_stats.append(validation["stats"])
            
            if not validation["valid"]:
                print(f"    ❌ Sequence {i} invalid: {validation['issues']}")
                return False, validation_stats
            else:
                stats = validation["stats"]
                print(f"    ✅ Sequence {i}: {stats['content_length']} content, {stats['padding_ratio']:.1%} padding")
    
    print(f"✅ All {test_count} test songs processed successfully")
    return True, validation_stats

def test_tokenizer_ranges():
    """Test that tokenizers produce tokens in expected ranges"""
    print("\nTesting tokenizer ranges...")
    
    preprocessor = FramePreprocessor(sequence_length=256)
    
    # Test melody tokenizer
    for midi in [-27, 0, 30, 60]:
        onset_token, hold_token = preprocessor.melody_tokenizer.encode_midi_note(midi)
        print(f"  MIDI {midi}: onset={onset_token}, hold={hold_token}")
        
        # Verify ranges
        assert 1 <= onset_token < MIDI_HOLD_START, f"Onset token {onset_token} out of range"
        assert MIDI_HOLD_START <= hold_token < MELODY_VOCAB_SIZE, f"Hold token {hold_token} out of range"
    
    # Test out-of-range MIDI
    silence_onset, silence_hold = preprocessor.melody_tokenizer.encode_midi_note(-100)
    assert silence_onset == SILENCE_TOKEN and silence_hold == SILENCE_TOKEN
    
    print("✅ Tokenizer ranges correct")

if __name__ == "__main__":
    try:
        test_vocabulary_ranges()
        test_midi_conversion()
        test_tokenizer_ranges()
        success, stats = test_small_sample()
        
        if success:
            # Print summary statistics
            if stats:
                avg_padding = sum(s["padding_ratio"] for s in stats) / len(stats)
                avg_content = sum(s["content_length"] for s in stats) / len(stats)
                print(f"\n📊 Summary Statistics:")
                print(f"  Average content length: {avg_content:.1f} frames")
                print(f"  Average padding ratio: {avg_padding:.1%}")
                print(f"  Total sequences tested: {len(stats)}")
            
            print("\n🎉 All tests passed! Ready to reprocess full dataset.")
            print(f"\n💡 Expected improvements:")
            print(f"  - Melody vocab: 257 → {MELODY_VOCAB_SIZE} tokens ({(1-MELODY_VOCAB_SIZE/257):.1%} reduction)")
            print(f"  - PAD token: {PAD_TOKEN} (outside melody range)")
            print(f"  - No more PAD tokens in chord sequences")
        else:
            print("\n❌ Tests failed. Fix issues before proceeding.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1) 