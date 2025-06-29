#!/usr/bin/env python3
"""
Analyze harmony in preprocessed data by measuring the percentage of melody notes
that are part of their corresponding chords.
"""

import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

from src.config.tokenization_config import (
    SILENCE_TOKEN,
    CHORD_TOKEN_START,
    CHORD_SILENCE_TOKEN,
    MELODY_ONSET_HOLD_START,
    PAD_TOKEN,
    token_to_midi_and_type,
)

def is_pitch_in_chord(pitch: int, chord_token: int, tokenizer_info: dict) -> bool:
    """
    Checks if a MIDI pitch is part of a given chord token.
    """
    token_to_chord = tokenizer_info.get("token_to_chord", {})
    
    # JSON saves integer keys as strings, so we must convert
    chord_token_str = str(chord_token)

    if chord_token_str not in token_to_chord:
        return False # Unknown chord token

    # The structure from tokenizer is [root, [intervals], inversion, is_onset]
    chord_info = token_to_chord[chord_token_str]
    root_pc = chord_info[0]
    intervals = chord_info[1]
    
    # A chord is defined by its root pitch class and intervals
    chord_pitch_classes = {(root_pc + interval) % 12 for interval in intervals}
    chord_pitch_classes.add(root_pc)

    # Convert the melody MIDI pitch to a pitch class
    melody_pitch_class = pitch % 12
    
    return melody_pitch_class in chord_pitch_classes

def analyze_sequence_harmony(sequence, tokenizer_info: dict) -> dict:
    """
    Analyzes harmony in a single sequence.
    """
    melody_tokens = sequence.melody_tokens
    chord_tokens = sequence.chord_tokens
    
    # Skip padding
    non_pad_mask = melody_tokens != PAD_TOKEN
    melody_tokens = melody_tokens[non_pad_mask]
    chord_tokens = chord_tokens[non_pad_mask]
    
    total_notes = 0
    in_harmony_count = 0
    
    for i, melody_token in enumerate(melody_tokens):
        # Skip if melody is silence
        if melody_token == SILENCE_TOKEN:
            continue
            
        # Only check onset tokens
        if melody_token < MELODY_ONSET_HOLD_START:
            continue
            
        # Check if it's an onset token (even offset from MELODY_ONSET_HOLD_START)
        token_offset = melody_token - MELODY_ONSET_HOLD_START
        if token_offset % 2 != 0:  # Skip hold tokens
            continue
            
        # Get the MIDI pitch from the melody token
        try:
            midi_pitch, is_onset = token_to_midi_and_type(melody_token)
            if not is_onset:
                continue
        except ValueError:
            print(f"Warning: Invalid melody token {melody_token}")
            continue
            
        # Skip if chord is silence
        chord_token = chord_tokens[i]
        if chord_token == CHORD_SILENCE_TOKEN:
            continue
            
        total_notes += 1
        if is_pitch_in_chord(midi_pitch, chord_token, tokenizer_info):
            in_harmony_count += 1
    
    return {
        "total_notes": total_notes,
        "in_harmony_notes": in_harmony_count,
        "harmony_ratio": in_harmony_count / total_notes if total_notes > 0 else 0
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze harmony in preprocessed data")
    parser.add_argument("--data_dir", type=str, default="data/interim",
                       help="Directory containing preprocessed data")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    splits = ["train", "valid", "test"]
    
    # Load tokenizer info
    tokenizer_info_path = data_dir / "train" / "tokenizer_info.json"
    with open(tokenizer_info_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    for split in splits:
        print(f"\nAnalyzing {split} split...")
        split_dir = data_dir / split
        sequence_files = sorted(list(split_dir.glob("sequence_*.pkl")))
        
        total_sequences = 0
        total_notes = 0
        total_in_harmony = 0
        harmony_ratios = []
        
        for seq_file in tqdm(sequence_files, desc=f"Processing {split}"):
            with open(seq_file, 'rb') as f:
                sequence = pickle.load(f)
            
            stats = analyze_sequence_harmony(sequence, tokenizer_info)
            total_sequences += 1
            total_notes += stats["total_notes"]
            total_in_harmony += stats["in_harmony_notes"]
            if stats["total_notes"] > 0:
                harmony_ratios.append(stats["harmony_ratio"])
        
        overall_ratio = total_in_harmony / total_notes if total_notes > 0 else 0
        harmony_ratios = np.array(harmony_ratios)
        
        print(f"\n{split.upper()} Harmony Analysis:")
        print(f"  Total sequences: {total_sequences}")
        print(f"  Total melody notes: {total_notes}")
        print(f"  Notes in harmony: {total_in_harmony}")
        print(f"  Overall harmony ratio: {overall_ratio:.2%}")
        print(f"  Per-sequence harmony stats:")
        print(f"    Mean: {np.mean(harmony_ratios):.2%}")
        print(f"    Median: {np.median(harmony_ratios):.2%}")
        print(f"    Std: {np.std(harmony_ratios):.2%}")
        print(f"    Min: {np.min(harmony_ratios):.2%}")
        print(f"    Max: {np.max(harmony_ratios):.2%}")

if __name__ == "__main__":
    main() 