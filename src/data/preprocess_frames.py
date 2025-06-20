#!/usr/bin/env python3
"""
Frame-based Preprocessing: Convert raw chord data into frame sequences
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import random

from src.config.tokenization_config import (
    SILENCE_TOKEN,
    CHORD_TOKEN_START,
    CHORD_SILENCE_TOKEN,
    CHORD_ONSET_HOLD_START,
    MIDI_ONSET_START,
    MIDI_HOLD_START,
    MAX_MIDI_NOTE,
    MELODY_VOCAB_SIZE,
)
from src.data.datastructures import FrameSequence

# Token ID ranges
MELODY_TOKEN_START = 0      # Melody tokens: 0-256 (257 tokens)
CHORD_TOKEN_START = 257     # Chord tokens: 257-4832 (4576 tokens)
SILENCE_TOKEN = 0           # Shared silence token for melody
CHORD_SILENCE_TOKEN = CHORD_TOKEN_START  # 257, first chord token

class ChordTokenizer:
    def __init__(self):
        self.chord_to_token = {}  # Maps (root, intervals, inversion) → (onset_token, hold_token)
        self.token_to_chord = {}  # Maps token_id → (root, intervals, inversion, is_onset)
        self.next_token_id = CHORD_ONSET_HOLD_START  # Start after chord silence
        self.silence_token = CHORD_SILENCE_TOKEN
        
    def _get_chord_key(self, root: int, intervals: List[int], inversion: int) -> Tuple[int, Tuple[int, ...], int]:
        """Create a consistent key for a chord, ensuring interval order doesn't matter"""
        if not intervals:
            raise ValueError("Chord must have at least one interval")
        if not all(isinstance(x, int) for x in intervals):
            raise ValueError("All intervals must be integers")
        if not isinstance(root, int) or not isinstance(inversion, int):
            raise ValueError("Root and inversion must be integers")
            
        return (root, tuple(sorted(intervals)), inversion)  # Sort intervals for consistency
    
    def encode_chord(self, root: int, intervals: List[int], inversion: int) -> Tuple[int, int]:
        """Encode a chord into onset and hold token IDs"""
        chord_key = self._get_chord_key(root, intervals, inversion)
        
        if chord_key not in self.chord_to_token:
            # Create new onset and hold tokens
            onset_token = self.next_token_id
            hold_token = self.next_token_id + 1
            self.next_token_id += 2
            
            # No hardcoded vocab limit
            self.chord_to_token[chord_key] = (onset_token, hold_token)
            self.token_to_chord[onset_token] = (*chord_key, True)   # Add is_onset=True
            self.token_to_chord[hold_token] = (*chord_key, False)   # Add is_onset=False
            
        return self.chord_to_token[chord_key]
    
    def decode_chord(self, token: int) -> Tuple[int, List[int], int, bool]:
        """Decode a token ID back into chord components and onset status"""
        if token == self.silence_token:
            return -1, [], -1, False
        if token not in self.token_to_chord:
            raise ValueError(f"Unknown token ID: {token}")
            
        root, intervals, inversion, is_onset = self.token_to_chord[token]
        return root, list(intervals), inversion, is_onset

class MIDITokenizer:
    def __init__(self):
        self.silence_token = SILENCE_TOKEN
        self.midi_onset_start = MIDI_ONSET_START
        self.midi_hold_start = MIDI_HOLD_START
        self.max_midi_note = MAX_MIDI_NOTE
        # Pre-populate the mappings
        self.note_to_token = {(-1, -1): (self.silence_token, self.silence_token)}
        self.token_to_note = {self.silence_token: (-1, -1, False)}
        for midi_num in range(self.max_midi_note + 1):
            onset_token = self.midi_onset_start + midi_num
            hold_token = self.midi_hold_start + midi_num
            self.note_to_token[midi_num] = (onset_token, hold_token)
            self.token_to_note[onset_token] = (midi_num, -1, True)
            self.token_to_note[hold_token] = (midi_num, -1, False)
    def encode_midi_note(self, midi_number: int) -> Tuple[int, int]:
        if not (0 <= midi_number <= self.max_midi_note):
            return self.silence_token, self.silence_token
        onset_token = self.midi_onset_start + midi_number
        hold_token = self.midi_hold_start + midi_number
        return onset_token, hold_token
    def decode_note(self, token: int) -> Tuple[int, int, bool]:
        if token == self.silence_token:
            return -1, -1, False
        if token not in self.token_to_note:
            raise ValueError(f"Unknown token ID: {token}")
        midi_num, unused, is_onset = self.token_to_note[token]
        return midi_num, unused, is_onset
    @property
    def next_token_id(self):
        return CHORD_TOKEN_START

class FramePreprocessor:
    def __init__(self, sequence_length: int = 256):
        self.sequence_length = sequence_length
        self.chord_tokenizer = ChordTokenizer()
        self.melody_tokenizer = MIDITokenizer()  # Changed from MelodyTokenizer
        
    def _get_song_contexts(self, song_data: Dict, total_frames: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-calculates key and meter contexts for the entire song."""
        # Key context
        key_context = np.full(total_frames, 0) # Default to C major
        key_changes = sorted(song_data.get('annotations', {}).get('keys', []), key=lambda x: x.get('beat', 0))
        if key_changes:
            for i, key_change in enumerate(key_changes):
                start_frame = int(key_change.get('beat', 0) * 4)
                end_frame = total_frames
                if i + 1 < len(key_changes):
                    end_frame = int(key_changes[i+1].get('beat', 0) * 4)
                key_context[start_frame:end_frame] = key_change.get('tonic_pitch_class', 0)

        # Meter context
        meter_context = np.full(total_frames, 4) # Default to 4/4
        meter_changes = sorted(song_data.get('annotations', {}).get('meters', []), key=lambda x: x.get('beat', 0))
        if meter_changes:
            for i, meter_change in enumerate(meter_changes):
                start_frame = int(meter_change.get('beat', 0) * 4)
                end_frame = total_frames
                if i + 1 < len(meter_changes):
                    end_frame = int(meter_changes[i+1].get('beat', 0) * 4)
                meter_context[start_frame:end_frame] = meter_change.get('beats_per_bar', 4)
                
        return key_context, meter_context

    def convert_song_to_frames(self, song_data: Dict) -> Tuple[List[int], List[int]]:
        """Convert a song to frame sequences using MIDI representation"""
        num_beats = int(song_data.get('annotations', {}).get('num_beats', 0))
        if num_beats == 0:
            return [], []
        total_frames = num_beats * 4
        
        # Initialize with silence tokens
        melody_tokens = [self.melody_tokenizer.silence_token] * total_frames
        chord_tokens = [self.chord_tokenizer.silence_token] * total_frames  # Use chord silence token
        
        # Convert melody notes to MIDI with range validation
        for note in song_data.get('annotations', {}).get('melody', []):
            onset_frame = int(note['onset'] * 4)
            offset_frame = int(note['offset'] * 4)
            pitch_class = note['pitch_class']
            octave = note['octave']
            midi_number = (octave + 1) * 12 + pitch_class
            if not (0 <= midi_number <= self.melody_tokenizer.max_midi_note):
                print(f"Warning: Note MIDI number {midi_number} out of range, skipping")
                continue

            onset_token, hold_token = self.melody_tokenizer.encode_midi_note(midi_number)
            if onset_frame < total_frames:
                melody_tokens[onset_frame] = onset_token
                for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                    melody_tokens[frame] = hold_token
        # Improved chord processing
        for chord in song_data.get('annotations', {}).get('harmony', []):
            onset_frame = int(chord['onset'] * 4)
            offset_frame = int(chord['offset'] * 4)
            try:
                root = chord.get('root_pitch_class')
                intervals = chord.get('root_position_intervals', [])
                inversion = chord.get('inversion', 0)
                if root is None:
                    continue
                if not intervals or len(intervals) == 0:
                    continue
                if not isinstance(intervals, list):
                    continue
                onset_token, hold_token = self.chord_tokenizer.encode_chord(root, intervals, inversion)
                if onset_frame < total_frames:
                    chord_tokens[onset_frame] = onset_token
                    for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                        chord_tokens[frame] = hold_token
            except (KeyError, ValueError, TypeError) as e:
                continue
        return melody_tokens, chord_tokens
    
    def process_song(self, song_id: str, song_data: Dict) -> List[FrameSequence]:
        """Process a single song into frame sequences"""
        sequences = []
        
        # Skip songs without required tags
        if 'MELODY' not in song_data.get('tags', []) or 'HARMONY' not in song_data.get('tags', []):
            return sequences
        
        # Convert song to frames
        melody_tokens, chord_tokens = self.convert_song_to_frames(song_data)
        
        if not melody_tokens or not chord_tokens:
            return sequences
        
        # Pre-calculate context for the whole song once
        total_frames = len(melody_tokens)
        song_key_context, song_meter_context = self._get_song_contexts(song_data, total_frames)
        
        # Create sequences of fixed length
        for start_idx in range(0, len(melody_tokens), self.sequence_length):
            end_idx = min(start_idx + self.sequence_length, len(melody_tokens))
            
            # Get subsequences
            melody_seq = melody_tokens[start_idx:end_idx]
            chord_seq = chord_tokens[start_idx:end_idx]
            
            # Get context for this specific chunk by slicing the pre-calculated array
            key_context = song_key_context[start_idx:end_idx]
            meter_context = song_meter_context[start_idx:end_idx]
            
            # Pad if needed
            if len(melody_seq) < self.sequence_length:
                melody_seq.extend([self.melody_tokenizer.silence_token] * (self.sequence_length - len(melody_seq)))
                chord_seq.extend([self.chord_tokenizer.silence_token] * (self.sequence_length - len(chord_seq)))
                key_context = np.pad(key_context, (0, self.sequence_length - len(key_context)), mode='edge')
                meter_context = np.pad(meter_context, (0, self.sequence_length - len(meter_context)), mode='edge')
            
            sequences.append(FrameSequence(
                melody_tokens=np.array(melody_seq),
                chord_tokens=np.array(chord_seq),
                key_context=key_context,
                meter_context=meter_context,
                song_id=song_id,
                start_frame=start_idx
            ))
        
        return sequences

def save_processed_data(sequences: List[FrameSequence], chord_tokenizer: ChordTokenizer, 
                       melody_tokenizer: MIDITokenizer, output_dir: Path):
    """Save processed sequences as individual pickle files and tokenizer info as json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each sequence as a separate pickle file for easy lazy loading
    print(f"Saving {len(sequences)} sequences to {output_dir}...")
    for i, seq in enumerate(sequences):
        # Use a zero-padded filename for correct sorting
        sequence_filename = f"sequence_{i:06d}.pkl"
        with open(output_dir / sequence_filename, 'wb') as f:
            pickle.dump(seq, f)

    # Save tokenizer info as json
    tokenizer_info = {
        "melody_vocab_size": melody_tokenizer.next_token_id,
        "chord_vocab_size": chord_tokenizer.next_token_id,
        "total_vocab_size": chord_tokenizer.next_token_id,
        "chord_token_start": CHORD_TOKEN_START,
        "token_to_chord": chord_tokenizer.token_to_chord,
        "token_to_note": melody_tokenizer.token_to_note,
    }

    with open(output_dir / 'tokenizer_info.json', 'w') as f:
        # Custom JSON encoder to handle tuple keys
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(tokenizer_info, f, cls=NpEncoder, indent=4)
    print("Tokenizer info saved.")

def main():
    # Get the project root directory and file paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    raw_data_path = data_dir / "raw" / "Hooktheory copy.json"
    output_dir = data_dir / "interim"
    
    # Load the raw data
    print("Loading raw data...")
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Initialize preprocessor
    preprocessor = FramePreprocessor(sequence_length=256)
    
    # Process each song
    print("Processing songs into frame sequences...")
    song_sequences = defaultdict(list)  # More robust collection
    
    for song_id, song_data in raw_data.items():
        # Basic validation of song_data structure
        if not isinstance(song_data, dict) or 'annotations' not in song_data:
            print(f"Warning: Skipping song {song_id} due to missing 'annotations'")
            continue
            
        sequences = preprocessor.process_song(song_id, song_data)
        if sequences:  # Only add songs that have valid sequences
            song_sequences[song_id] = sequences
    
    # Shuffle songs (not sequences) to avoid data leakage
    song_ids = list(song_sequences.keys())
    random.seed(42) # for reproducibility
    random.shuffle(song_ids)
    
    # Split songs into train/valid/test (80/10/10)
    n = len(song_ids)
    train_idx = int(0.8 * n)
    valid_idx = int(0.9 * n)
    
    train_songs = song_ids[:train_idx]
    valid_songs = song_ids[train_idx:valid_idx]
    test_songs = song_ids[valid_idx:]
    
    # Collect sequences for each split
    train_sequences = []
    valid_sequences = []
    test_sequences = []
    
    for song_id in train_songs:
        train_sequences.extend(song_sequences[song_id])
    for song_id in valid_songs:
        valid_sequences.extend(song_sequences[song_id])
    for song_id in test_songs:
        test_sequences.extend(song_sequences[song_id])
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(train_sequences, preprocessor.chord_tokenizer, 
                       preprocessor.melody_tokenizer, output_dir / 'train')
    save_processed_data(valid_sequences, preprocessor.chord_tokenizer,
                       preprocessor.melody_tokenizer, output_dir / 'valid')
    save_processed_data(test_sequences, preprocessor.chord_tokenizer,
                       preprocessor.melody_tokenizer, output_dir / 'test')
    
    # Print statistics
    print(f"\nPreprocessing complete!")
    print(f"Total songs: {len(song_sequences)}")
    print(f"  Train: {len(train_songs)}")
    print(f"  Valid: {len(valid_songs)}")
    print(f"  Test: {len(test_songs)}")
    print(f"Total sequences: {len(train_sequences) + len(valid_sequences) + len(test_sequences)}")
    print(f"  Train: {len(train_sequences)}")
    print(f"  Valid: {len(valid_sequences)}")
    print(f"  Test: {len(test_sequences)}")
    print(f"Sequence length: {preprocessor.sequence_length}")
    
    # --- Load the final training tokenizer info to print the definitive vocab sizes ---
    with open(output_dir / 'train' / 'tokenizer_info.json', 'r') as f:
        final_tokenizer_info = json.load(f)

    print(f"Vocabulary sizes:")
    print(f"  Melody tokens: {final_tokenizer_info['melody_vocab_size']}")
    print(f"  Chord tokens: {final_tokenizer_info['chord_vocab_size']}")
    print(f"  Total tokens: {final_tokenizer_info['total_vocab_size']}")

if __name__ == "__main__":
    main() 