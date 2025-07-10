#!/usr/bin/env python3
"""
Examine preprocessed sequences to understand the data format and tokenization.
"""

import numpy as np
import json
import pickle
from pathlib import Path
import random
from src.data.preprocess_frames import MIDITokenizer
from src.data.dataset import create_dataloader

def examine_raw_sequence_files():
    """Look at the raw .pkl files to understand their structure."""
    print("=== EXAMINING RAW SEQUENCE FILES ===")
    
    data_dir = Path("data/interim")
    
    # Check what's in the directories
    for split in ["train", "test", "valid"]:
        split_dir = data_dir / split
        if split_dir.exists():
            pkl_files = list(split_dir.glob("*.pkl"))
            json_files = list(split_dir.glob("*.json"))
            print(f"\n{split.upper()} split:")
            print(f"  PKL files: {len(pkl_files)}")
            print(f"  JSON files: {len(json_files)}")
            
            # Load and examine a few sequence files
            if pkl_files:
                for i, pkl_file in enumerate(pkl_files[:3]):
                    print(f"\n  Examining {pkl_file.name}:")
                    try:
                        with open(pkl_file, 'rb') as f:
                            data = pickle.load(f)
                        
                        print(f"    Data type: {type(data)}")
                        if isinstance(data, dict):
                            print(f"    Keys: {list(data.keys())}")
                            for key, value in data.items():
                                if isinstance(value, np.ndarray):
                                    print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                                    print(f"      Range: [{value.min()}, {value.max()}]")
                                    print(f"      First 10: {value[:10]}")
                                else:
                                    print(f"    {key}: {type(value)} - {value}")
                        elif isinstance(data, (list, np.ndarray)):
                            print(f"    Length: {len(data)}")
                            print(f"    Type of elements: {type(data[0]) if len(data) > 0 else 'empty'}")
                            
                    except Exception as e:
                        print(f"    Error loading: {e}")

def examine_dataloader_output():
    """Examine what the dataloader provides."""
    print("\n\n=== EXAMINING DATALOADER OUTPUT ===")
    
    data_dir = Path("data/interim")
    
    # Load tokenizer info
    tokenizer_path = data_dir / "test" / "tokenizer_info.json"
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    print(f"Tokenizer info:")
    for key, value in tokenizer_info.items():
        print(f"  {key}: {value}")
    
    # Test both online and offline dataloaders
    for mode in ["online", "offline"]:
        print(f"\n--- {mode.upper()} DATALOADER ---")
        try:
            dataloader, _ = create_dataloader(
                data_dir=data_dir,
                split="test",
                batch_size=2,
                num_workers=0,
                sequence_length=64,  # Smaller for easier viewing
                mode=mode,
                shuffle=False
            )
            
            # Get one batch
            batch = next(iter(dataloader))
            print(f"Batch keys: {list(batch.keys())}")
            
            for key, value in batch.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                    print(f"    Range: [{value.min()}, {value.max()}]")
                    print(f"    First sequence (first 20 tokens): {value[0][:20]}")
                else:
                    print(f"  {key}: {type(value)} - {value}")
                    
        except Exception as e:
            print(f"  Error with {mode} dataloader: {e}")

def decode_sequences():
    """Decode sequences to understand what they represent musically."""
    print("\n\n=== DECODING SEQUENCES TO MUSICAL INFORMATION ===")
    
    data_dir = Path("data/interim")
    
    # Load tokenizer info
    tokenizer_path = data_dir / "test" / "tokenizer_info.json"
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    # Create MIDI tokenizer
    midi_tokenizer = MIDITokenizer()
    
    # Get some sequences from offline dataloader (easier to understand)
    dataloader, _ = create_dataloader(
        data_dir=data_dir,
        split="test",
        batch_size=3,
        num_workers=0,
        sequence_length=32,  # Short sequences for readability
        mode="offline",
        shuffle=False
    )
    
    batch = next(iter(dataloader))
    melody_tokens = batch['melody_tokens'].numpy()
    chord_tokens = batch['chord_target'].numpy()
    
    print(f"Examining {len(melody_tokens)} sequences...")
    
    for seq_idx in range(len(melody_tokens)):
        print(f"\n--- SEQUENCE {seq_idx} ---")
        melody_seq = melody_tokens[seq_idx]
        chord_seq = chord_tokens[seq_idx]
        
        print(f"Melody tokens: {melody_seq}")
        print(f"Chord tokens:  {chord_seq}")
        
        # Decode melody
        print(f"\nMelody decoding:")
        for i, token in enumerate(melody_seq[:15]):  # First 15 tokens
            if token == tokenizer_info.get('pad_token_id', 178):
                print(f"  Frame {i:2d}: PAD")
                break
            
            midi_note, is_onset = midi_tokenizer.decode_token(int(token))
            if midi_note is not None:
                note_name = midi_note_to_name(midi_note)
                onset_str = "ONSET" if is_onset else "HOLD "
                print(f"  Frame {i:2d}: {onset_str} {note_name} (MIDI {midi_note}) [token {token}]")
            elif token == 0:
                print(f"  Frame {i:2d}: SILENCE [token {token}]")
            else:
                print(f"  Frame {i:2d}: UNKNOWN [token {token}]")
        
        # Decode chords
        print(f"\nChord decoding:")
        token_to_chord = tokenizer_info.get('token_to_chord', {})
        
        for i, token in enumerate(chord_seq[:15]):  # First 15 tokens
            if token == tokenizer_info.get('pad_token_id', 178):
                print(f"  Frame {i:2d}: PAD")
                break
            elif token == 0:
                print(f"  Frame {i:2d}: SILENCE [token {token}]")
            elif str(token) in token_to_chord:
                chord_info = token_to_chord[str(token)]
                chord_name = chord_info.get('chord_name', 'Unknown')
                is_hold = chord_info.get('is_hold', False)
                onset_str = "HOLD " if is_hold else "ONSET"
                print(f"  Frame {i:2d}: {onset_str} {chord_name} [token {token}]")
            else:
                print(f"  Frame {i:2d}: UNKNOWN CHORD [token {token}]")
        
        # Show sequence statistics
        print(f"\nSequence Statistics:")
        unique_melody, melody_counts = np.unique(melody_seq, return_counts=True)
        unique_chords, chord_counts = np.unique(chord_seq, return_counts=True)
        
        print(f"  Melody: {len(unique_melody)} unique tokens")
        print(f"  Chords: {len(unique_chords)} unique tokens")
        
        # Check for realistic patterns
        non_pad_melody = melody_seq[melody_seq != tokenizer_info.get('pad_token_id', 178)]
        non_pad_chords = chord_seq[chord_seq != tokenizer_info.get('pad_token_id', 178)]
        
        print(f"  Active melody frames: {len(non_pad_melody)}")
        print(f"  Active chord frames: {len(non_pad_chords)}")

def midi_note_to_name(midi_note):
    """Convert MIDI note number to note name."""
    if midi_note < 0 or midi_note > 127:
        return f"Invalid({midi_note})"
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_note // 12 - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def analyze_token_ranges():
    """Analyze the token ranges to understand the vocabulary structure."""
    print("\n\n=== ANALYZING TOKEN RANGES ===")
    
    data_dir = Path("data/interim")
    tokenizer_path = data_dir / "test" / "tokenizer_info.json"
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    print("Vocabulary Structure:")
    print(f"  Total vocab size: {tokenizer_info['total_vocab_size']}")
    print(f"  Melody vocab size: {tokenizer_info['melody_vocab_size']}")
    print(f"  Chord vocab size: {tokenizer_info['chord_vocab_size']}")
    print(f"  Chord token start: {tokenizer_info['chord_token_start']}")
    print(f"  PAD token: {tokenizer_info.get('pad_token_id', 'Not specified')}")
    
    # Analyze chord patterns
    token_to_chord = tokenizer_info.get('token_to_chord', {})
    print(f"\nChord Patterns: {len(token_to_chord)} total chord tokens")
    
    onset_tokens = []
    hold_tokens = []
    
    for token_str, chord_info in token_to_chord.items():
        token = int(token_str)
        if chord_info.get('is_hold', False):
            hold_tokens.append(token)
        else:
            onset_tokens.append(token)
    
    if onset_tokens:
        print(f"  Onset tokens: {len(onset_tokens)} (range: {min(onset_tokens)} - {max(onset_tokens)})")
    if hold_tokens:
        print(f"  Hold tokens: {len(hold_tokens)} (range: {min(hold_tokens)} - {max(hold_tokens)})")
    
    # Show some example chord mappings
    print(f"\nExample Chord Mappings (first 10):")
    for i, (token_str, chord_info) in enumerate(list(token_to_chord.items())[:10]):
        chord_name = chord_info.get('chord_name', 'Unknown')
        is_hold = chord_info.get('is_hold', False)
        token_type = "HOLD" if is_hold else "ONSET"
        print(f"  Token {token_str}: {token_type} {chord_name}")

if __name__ == "__main__":
    print("Examining preprocessed sequences...")
    
    examine_raw_sequence_files()
    examine_dataloader_output()
    decode_sequences()
    analyze_token_ranges()
    
    print("\n" + "="*60)
    print("✅ EXAMINATION COMPLETE")
    print("="*60) 