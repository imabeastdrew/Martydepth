#!/usr/bin/env python3
"""
Chord Preprocessor: Convert raw chord data into tokenized encodings
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from collections import defaultdict

class ChordTokenizer:
    def __init__(self):
        self.chord_to_token = {}  # Maps (root, intervals, inversion) → token_id
        self.token_to_chord = {}  # Maps token_id → (root, intervals, inversion)
        self.next_token_id = 0
        
    def _get_chord_key(self, root: int, intervals: List[int], inversion: int) -> Tuple[int, Tuple[int, ...], int]:
        """Create a consistent key for a chord, preserving interval order"""
        if not intervals:
            raise ValueError("Chord must have at least one interval")
        if not all(isinstance(x, int) for x in intervals):
            raise ValueError("All intervals must be integers")
        if not isinstance(root, int) or not isinstance(inversion, int):
            raise ValueError("Root and inversion must be integers")
            
        return (root, tuple(intervals), inversion)  # Preserve original interval order
    
    def encode_chord(self, root: int, intervals: List[int], inversion: int) -> int:
        """Encode a chord into a token ID, creating new token if needed"""
        chord_key = self._get_chord_key(root, intervals, inversion)
        
        if chord_key not in self.chord_to_token:
            self.chord_to_token[chord_key] = self.next_token_id
            self.token_to_chord[self.next_token_id] = chord_key
            self.next_token_id += 1
            
        return self.chord_to_token[chord_key]
    
    def decode_chord(self, token: int) -> Tuple[int, List[int], int]:
        """Decode a token ID back into chord components"""
        if token not in self.token_to_chord:
            raise ValueError(f"Unknown token ID: {token}")
            
        root, intervals, inversion = self.token_to_chord[token]
        return root, list(intervals), inversion

def preprocess_dataset(data_path: str) -> Tuple[List[int], ChordTokenizer]:
    """Preprocess the dataset and return tokenized chords and tokenizer"""
    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Initialize tokenizer
    tokenizer = ChordTokenizer()
    all_chord_tokens = []
    skipped_songs = 0
    skipped_chords = 0
    
    # Process each song
    for song_id, song in data.items():
        try:
            if not isinstance(song, dict):
                print(f"Warning: Song {song_id} is not a dictionary, skipping")
                skipped_songs += 1
                continue
                
            annotations = song.get('annotations', {})
            if not isinstance(annotations, dict):
                print(f"Warning: Annotations for song {song_id} is not a dictionary, skipping")
                skipped_songs += 1
                continue
                
            harmony = annotations.get('harmony')
            if not harmony:
                skipped_songs += 1
                continue
                
            if not isinstance(harmony, list):
                print(f"Warning: Harmony for song {song_id} is not a list, skipping")
                skipped_songs += 1
                continue
            
            # Process each chord in the song
            for chord in harmony:
                try:
                    if not isinstance(chord, dict):
                        print(f"Warning: Chord in song {song_id} is not a dictionary, skipping")
                        skipped_chords += 1
                        continue
                        
                    root = chord.get('root_pitch_class')
                    intervals = chord.get('root_position_intervals')
                    inversion = chord.get('inversion', 0)
                    
                    # Skip if missing essential data
                    if root is None or intervals is None:
                        skipped_chords += 1
                        continue
                    
                    # Encode chord
                    token = tokenizer.encode_chord(root, intervals, inversion)
                    all_chord_tokens.append(token)
                    
                except ValueError as e:
                    print(f"Warning: Invalid chord in song {song_id}: {e}")
                    skipped_chords += 1
                    continue
                    
        except Exception as e:
            print(f"Error processing song {song_id}: {e}")
            skipped_songs += 1
            continue
    
    print(f"\nProcessing statistics:")
    print(f"Skipped songs: {skipped_songs}")
    print(f"Skipped chords: {skipped_chords}")
    
    return all_chord_tokens, tokenizer

def save_processed_data(tokens: List[int], tokenizer: ChordTokenizer, output_dir: Path):
    """Save processed data and tokenizer information"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokens
    np.save(output_dir / 'chord_tokens.npy', np.array(tokens))
    
    # Save tokenizer info
    tokenizer_info = {
        'chord_to_token': {str(k): v for k, v in tokenizer.chord_to_token.items()},
        'token_to_chord': {str(k): v for k, v in tokenizer.token_to_chord.items()},
        'vocab_size': tokenizer.next_token_id
    }
    
    with open(output_dir / 'tokenizer_info.json', 'w') as f:
        json.dump(tokenizer_info, f, indent=2)

def main():
    # Get the project root directory and file paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "Hooktheory copy.json"
    output_dir = project_root / "data" / "processed"
    
    # Preprocess the dataset
    print("Preprocessing dataset...")
    tokens, tokenizer = preprocess_dataset(str(data_path))
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(tokens, tokenizer, output_dir)
    
    # Print some statistics
    print(f"\nPreprocessing complete!")
    print(f"Total chords processed: {len(tokens)}")
    print(f"Unique chord types: {tokenizer.next_token_id}")
    
    # Show some example encodings
    print("\nExample chord encodings:")
    for i in range(min(5, len(tokens))):
        root, intervals, inversion = tokenizer.decode_chord(tokens[i])
        print(f"Token {tokens[i]}: Root={root}, Intervals={intervals}, Inversion={inversion}")

if __name__ == "__main__":
    main() 