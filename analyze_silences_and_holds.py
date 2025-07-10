#!/usr/bin/env python3
"""
Detailed analysis of silence sections and hold token behavior in preprocessed data.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from collections import Counter
from src.data.dataset import create_dataloader
from src.data.preprocess_frames import MIDITokenizer

def midi_note_to_name(midi_note):
    """Convert MIDI note number to note name (C4 = 60)."""
    if midi_note is None:
        return "UNKNOWN"
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note + 9) // 12 - 1  # C4 = 60, so offset by 9
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def analyze_silence_patterns():
    """Analyze silence patterns in melody and chord sequences."""
    print("=== ANALYZING SILENCE PATTERNS ===")
    
    data_dir = Path("data/interim")
    
    # Load tokenizer info
    tokenizer_path = data_dir / "test" / "tokenizer_info.json"
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    # Create dataloader
    offline_loader, _ = create_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=32,
        num_workers=0,
        sequence_length=128,
        mode='offline',
        shuffle=False
    )
    
    melody_tokenizer = MIDITokenizer()
    
    # Analyze patterns
    silence_stats = {
        'chord_silence_sequences': 0,
        'melody_silence_frames': 0,
        'chord_silence_frames': 0,
        'total_sequences': 0,
        'total_frames': 0,
        'melody_token_counter': Counter(),
        'chord_token_counter': Counter(),
        'unknown_tokens': set()
    }
    
    print("Analyzing batches...")
    for batch_idx, batch in enumerate(offline_loader):
        if batch_idx >= 5:  # Analyze first 5 batches
            break
            
        melody_tokens = batch['melody_tokens'].numpy()
        chord_tokens = batch['chord_target'].numpy()
        
        for seq_idx in range(len(melody_tokens)):
            melody_seq = melody_tokens[seq_idx]
            chord_seq = chord_tokens[seq_idx]
            
            silence_stats['total_sequences'] += 1
            sequence_length = len(melody_seq)
            silence_stats['total_frames'] += sequence_length
            
            # Count tokens
            silence_stats['melody_token_counter'].update(melody_seq)
            silence_stats['chord_token_counter'].update(chord_seq)
            
            # Check for chord silence sequences (all silence using unified token 88)
            if np.all(chord_seq == 88):
                silence_stats['chord_silence_sequences'] += 1
            
            # Count silence frames using unified silence token (88)
            melody_silence_frames = np.sum(melody_seq == 88)
            chord_silence_frames = np.sum(chord_seq == 88)
            
            silence_stats['melody_silence_frames'] += melody_silence_frames
            silence_stats['chord_silence_frames'] += chord_silence_frames
            
            # Check for unknown tokens
            for token in melody_seq:
                if token >= 179:  # Should be melody tokens < 179
                    silence_stats['unknown_tokens'].add(int(token))
            
            # Example sequence with interesting patterns
            if seq_idx < 3 and batch_idx == 0:
                print(f"\n--- Example Sequence {seq_idx} ---")
                print(f"Melody tokens (first 20): {melody_seq[:20]}")
                print(f"Chord tokens (first 20): {chord_seq[:20]}")
                
                # Decode some tokens
                melody_analysis = []
                for i, token in enumerate(melody_seq[:10]):
                    midi_note, is_onset = melody_tokenizer.decode_token(token)
                    if midi_note is not None:
                        note_name = midi_note_to_name(midi_note)
                        melody_analysis.append(f"Frame {i}: {'ONSET' if is_onset else 'HOLD'} {note_name} [token {token}]")
                    elif token == 88:
                        melody_analysis.append(f"Frame {i}: SILENCE [token {token}]")
                    else:
                        melody_analysis.append(f"Frame {i}: UNKNOWN [token {token}]")
                
                print("Melody analysis:")
                for line in melody_analysis:
                    print(f"  {line}")
    
    # Print statistics
    print(f"\n=== SILENCE STATISTICS ===")
    print(f"Total sequences analyzed: {silence_stats['total_sequences']}")
    print(f"Total frames analyzed: {silence_stats['total_frames']}")
    print(f"Sequences with all-silence chords: {silence_stats['chord_silence_sequences']} ({silence_stats['chord_silence_sequences']/silence_stats['total_sequences']*100:.1f}%)")
    print(f"Melody silence frames: {silence_stats['melody_silence_frames']} ({silence_stats['melody_silence_frames']/silence_stats['total_frames']*100:.1f}%)")
    print(f"Chord silence frames: {silence_stats['chord_silence_frames']} ({silence_stats['chord_silence_frames']/silence_stats['total_frames']*100:.1f}%)")
    
    # Most common melody tokens
    print(f"\nMost common melody tokens:")
    for token, count in silence_stats['melody_token_counter'].most_common(10):
        if token == 88:
            print(f"  Token {token}: {count} times (SILENCE)")
        elif token == 178:
            print(f"  Token {token}: {count} times (PAD)")
        elif token < 179:
            midi_note, is_onset = melody_tokenizer.decode_token(token)
            if midi_note is not None:
                note_name = midi_note_to_name(midi_note)
                print(f"  Token {token}: {count} times ({'ONSET' if is_onset else 'HOLD'} {note_name})")
            else:
                print(f"  Token {token}: {count} times (UNKNOWN)")
        else:
            print(f"  Token {token}: {count} times (INVALID - should be chord token)")
    
    # Most common chord tokens
    print(f"\nMost common chord tokens:")
    for token, count in silence_stats['chord_token_counter'].most_common(10):
        if token == 88:
            print(f"  Token {token}: {count} times (SILENCE)")
        elif token == 178:
            print(f"  Token {token}: {count} times (PAD)")
        elif token >= 179:
            if str(token) in tokenizer_info['token_to_chord']:
                chord_info = tokenizer_info['token_to_chord'][str(token)]
                is_hold = chord_info.get('is_hold', False)
                print(f"  Token {token}: {count} times ({'HOLD' if is_hold else 'ONSET'} chord)")
            else:
                print(f"  Token {token}: {count} times (UNKNOWN chord)")
        elif token == 88:
            print(f"  Token {token}: {count} times (CHORD SILENCE - unified silence token)")
        else:
            print(f"  Token {token}: {count} times (INVALID - should be melody token)")
    
    if silence_stats['unknown_tokens']:
        print(f"\nUnknown melody tokens found: {sorted(silence_stats['unknown_tokens'])}")
    
    return silence_stats

def analyze_hold_token_behavior():
    """Analyze hold token onset/hold patterns in detail."""
    print(f"\n=== ANALYZING HOLD TOKEN BEHAVIOR ===")
    
    data_dir = Path("data/interim")
    
    # Load tokenizer info
    tokenizer_path = data_dir / "test" / "tokenizer_info.json"
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    # Create dataloader
    offline_loader, _ = create_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=16,
        num_workers=0,
        sequence_length=64,
        mode='offline',
        shuffle=False
    )
    
    melody_tokenizer = MIDITokenizer()
    
    # Find sequences with clear onset→hold patterns
    for batch_idx, batch in enumerate(offline_loader):
        if batch_idx >= 2:  # Analyze first 2 batches
            break
            
        melody_tokens = batch['melody_tokens'].numpy()
        chord_tokens = batch['chord_target'].numpy()
        
        for seq_idx in range(min(3, len(melody_tokens))):  # First 3 sequences per batch
            melody_seq = melody_tokens[seq_idx]
            chord_seq = chord_tokens[seq_idx]
            
            print(f"\n--- Hold Pattern Analysis: Batch {batch_idx}, Sequence {seq_idx} ---")
            
            # Analyze melody onset/hold patterns
            melody_patterns = []
            current_note = None
            hold_count = 0
            
            for i, token in enumerate(melody_seq[:20]):  # First 20 frames
                if token == 178:  # PAD token
                    break
                    
                midi_note, is_onset = melody_tokenizer.decode_token(token)
                
                if midi_note is not None:
                    note_name = midi_note_to_name(midi_note)
                    
                    if is_onset:
                        if current_note and hold_count > 0:
                            melody_patterns.append(f"  → {current_note} held for {hold_count} frames")
                        current_note = note_name
                        hold_count = 0
                        melody_patterns.append(f"Frame {i:2d}: ONSET {note_name} [token {token}]")
                    else:
                        hold_count += 1
                        melody_patterns.append(f"Frame {i:2d}: HOLD  {note_name} [token {token}]")
                elif token == 88:
                    if current_note and hold_count > 0:
                        melody_patterns.append(f"  → {current_note} held for {hold_count} frames")
                        current_note = None
                        hold_count = 0
                    melody_patterns.append(f"Frame {i:2d}: SILENCE [token {token}]")
                else:
                    melody_patterns.append(f"Frame {i:2d}: UNKNOWN [token {token}]")
            
            print("Melody patterns:")
            for pattern in melody_patterns:
                print(pattern)
            
            # Analyze chord onset/hold patterns
            print("\nChord patterns:")
            chord_patterns = []
            current_chord = None
            hold_count = 0
            
            for i, token in enumerate(chord_seq[:20]):  # First 20 frames
                if token == 178:  # PAD token
                    break
                    
                if token == 88:
                    if current_chord and hold_count > 0:
                        chord_patterns.append(f"  → Chord {current_chord} held for {hold_count} frames")
                        current_chord = None
                        hold_count = 0
                    chord_patterns.append(f"Frame {i:2d}: SILENCE [token {token}]")
                elif token >= 179:
                    if str(token) in tokenizer_info['token_to_chord']:
                        chord_info = tokenizer_info['token_to_chord'][str(token)]
                        is_hold = chord_info.get('is_hold', False)
                        
                        if not is_hold:  # Onset token
                            if current_chord and hold_count > 0:
                                chord_patterns.append(f"  → Chord {current_chord} held for {hold_count} frames")
                            current_chord = token
                            hold_count = 0
                            chord_patterns.append(f"Frame {i:2d}: ONSET Chord {token} [is_hold: {is_hold}]")
                        else:  # Hold token
                            hold_count += 1
                            chord_patterns.append(f"Frame {i:2d}: HOLD  Chord {token} [is_hold: {is_hold}]")
                    else:
                        chord_patterns.append(f"Frame {i:2d}: UNKNOWN chord [token {token}]")
                else:
                    chord_patterns.append(f"Frame {i:2d}: INVALID chord token [token {token}]")
            
            for pattern in chord_patterns:
                print(pattern)

def main():
    """Main analysis function."""
    silence_stats = analyze_silence_patterns()
    analyze_hold_token_behavior()
    
    print(f"\n{'='*60}")
    print("✅ SILENCE AND HOLD TOKEN ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 