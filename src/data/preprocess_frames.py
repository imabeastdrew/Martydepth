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

# Token ID ranges
MELODY_TOKEN_START = 0      # Melody tokens: 0-256 (257 tokens)
CHORD_TOKEN_START = 257     # Chord tokens: 257-4832 (4576 tokens)
SILENCE_TOKEN = 0           # Shared silence token for melody
CHORD_SILENCE_TOKEN = CHORD_TOKEN_START  # 257, first chord token

@dataclass
class FrameSequence:
    """Container for a sequence of chord frames"""
    melody_tokens: np.ndarray  # Shape: (sequence_length,)
    chord_tokens: np.ndarray   # Shape: (sequence_length,)
    key_context: np.ndarray    # Shape: (sequence_length,)
    meter_context: np.ndarray  # Shape: (sequence_length,)
    song_id: str
    start_frame: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'melody_tokens': self.melody_tokens,
            'chord_tokens': self.chord_tokens,
            'key_context': self.key_context,
            'meter_context': self.meter_context,
            'song_id': self.song_id,
            'start_frame': self.start_frame
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameSequence':
        """Create from dictionary after deserialization"""
        return cls(
            melody_tokens=data['melody_tokens'],
            chord_tokens=data['chord_tokens'],
            key_context=data['key_context'],
            meter_context=data['meter_context'],
            song_id=data['song_id'],
            start_frame=data['start_frame']
        )

class ChordTokenizer:
    def __init__(self):
        self.chord_to_token = {}  # Maps (root, intervals, inversion) → (onset_token, hold_token)
        self.token_to_chord = {}  # Maps token_id → (root, intervals, inversion, is_onset)
        self.next_token_id = CHORD_TOKEN_START + 1  # Start after chord silence (258)
        self.silence_token = CHORD_SILENCE_TOKEN  # 257
        
    def _get_chord_key(self, root: int, intervals: List[int], inversion: int) -> Tuple[int, Tuple[int, ...], int]:
        """Create a consistent key for a chord, preserving interval order"""
        if not intervals:
            raise ValueError("Chord must have at least one interval")
        if not all(isinstance(x, int) for x in intervals):
            raise ValueError("All intervals must be integers")
        if not isinstance(root, int) or not isinstance(inversion, int):
            raise ValueError("Root and inversion must be integers")
            
        return (root, tuple(intervals), inversion)  # Preserve original interval order
    
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
        self.silence_token = SILENCE_TOKEN  # 0
        self.midi_onset_start = 1           # MIDI 0-127 onset: tokens 1-128
        self.midi_hold_start = 129          # MIDI 0-127 hold: tokens 129-256
        self.max_midi_note = 127
        # Pre-populate the mappings
        self.note_to_token = {(-1, -1): (self.silence_token, self.silence_token)}
        self.token_to_note = {self.silence_token: (-1, -1, False)}
        for midi_num in range(128):
            onset_token = self.midi_onset_start + midi_num
            hold_token = self.midi_hold_start + midi_num
            self.note_to_token[midi_num] = (onset_token, hold_token)
            self.token_to_note[onset_token] = (midi_num, -1, True)
            self.token_to_note[hold_token] = (midi_num, -1, False)
    def encode_midi_note(self, midi_number: int) -> Tuple[int, int]:
        if midi_number < 0 or midi_number > 127:
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
        
    def _get_key_context(self, song_data: Dict, start_frame: int, end_frame: int) -> np.ndarray:
        """Extract key context for a specific frame range"""
        key_changes = song_data.get('annotations', {}).get('keys', [])
        if not key_changes:
            return np.full(end_frame - start_frame, 0)  # Default to C major
            
        # Create key context array for this chunk
        key_context = np.zeros(end_frame - start_frame)
        
        # Find the last key change before or at start_frame
        current_key = 0  # Default to C major
        for key_change in key_changes:
            beat = key_change.get('beat', 0)
            frame = int(beat * 4)  # Convert to 1/16th note frames
            if frame <= start_frame:
                current_key = key_change.get('tonic_pitch_class', current_key)
                
        # Fill the beginning of the context with the current key
        key_context[:] = current_key
        
        # Apply all key changes within this chunk
        for key_change in key_changes:
            beat = key_change.get('beat', 0)
            frame = int(beat * 4)
            if start_frame < frame < end_frame:
                current_key = key_change.get('tonic_pitch_class', current_key)
                key_context[frame - start_frame:] = current_key
                
        return key_context
    
    def _get_meter_context(self, song_data: Dict, start_frame: int, end_frame: int) -> np.ndarray:
        """Extract meter context for a specific frame range"""
        meter_changes = song_data.get('annotations', {}).get('meters', [])
        if not meter_changes:
            return np.full(end_frame - start_frame, 4)  # Default to 4/4
            
        # Create meter context array for this chunk
        meter_context = np.full(end_frame - start_frame, 4)  # Default to 4/4
        
        # Find the last meter change before or at start_frame
        current_meter = 4  # Default to 4/4
        for meter_change in meter_changes:
            beat = meter_change.get('beat', 0)
            frame = int(beat * 4)  # Convert to 1/16th note frames
            if frame <= start_frame:
                current_meter = meter_change.get('beats_per_bar', current_meter)
                
        # Fill the beginning of the context with the current meter
        meter_context[:] = current_meter
        
        # Apply all meter changes within this chunk
        for meter_change in meter_changes:
            beat = meter_change.get('beat', 0)
            frame = int(beat * 4)
            if start_frame < frame < end_frame:
                current_meter = meter_change.get('beats_per_bar', current_meter)
                meter_context[frame - start_frame:] = current_meter
                
        return meter_context
    
    def convert_song_to_frames(self, song_data: Dict) -> Tuple[List[int], List[int]]:
        """Convert a song to frame sequences using MIDI representation"""
        num_beats = int(song_data['annotations']['num_beats'])
        total_frames = num_beats * 4
        
        # Initialize with silence tokens
        melody_tokens = [self.melody_tokenizer.silence_token] * total_frames
        chord_tokens = [self.chord_tokenizer.silence_token] * total_frames  # Use chord silence token
        
        # Convert melody notes to MIDI with range validation
        for note in song_data['annotations']['melody']:
            onset_frame = int(note['onset'] * 4)
            offset_frame = int(note['offset'] * 4)
            pitch_class = note['pitch_class']
            octave = note['octave']
            midi_number = (octave + 1) * 12 + pitch_class
            if midi_number < 0:
                print(f"Warning: Note too low (pc={pitch_class}, oct={octave}, midi={midi_number}), skipping")
                continue
            if midi_number > 127:
                print(f"Warning: Note too high (pc={pitch_class}, oct={octave}, midi={midi_number}), skipping")
                continue
            onset_token, hold_token = self.melody_tokenizer.encode_midi_note(midi_number)
            if onset_frame < total_frames:
                melody_tokens[onset_frame] = onset_token
                for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                    melody_tokens[frame] = hold_token
        # Improved chord processing
        for chord in song_data['annotations']['harmony']:
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
        
        # Create sequences of fixed length
        for start_idx in range(0, len(melody_tokens), self.sequence_length):
            end_idx = min(start_idx + self.sequence_length, len(melody_tokens))
            
            # Get subsequences
            melody_seq = melody_tokens[start_idx:end_idx]
            chord_seq = chord_tokens[start_idx:end_idx]
            
            # Get context for this specific chunk
            key_context = self._get_key_context(song_data, start_idx, end_idx)
            meter_context = self._get_meter_context(song_data, start_idx, end_idx)
            
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
    """Save processed sequences and tokenizer info"""
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dicts = [seq.to_dict() for seq in sequences]
    split_name = output_dir.name
    with open(output_dir / f'frame_sequences_{split_name}.pkl', 'wb') as f:
        pickle.dump(sequence_dicts, f)
    tokenizer_info = {
        'chord_to_token': {str(k): v for k, v in chord_tokenizer.chord_to_token.items()},
        'token_to_chord': {str(k): v for k, v in chord_tokenizer.token_to_chord.items()},
        'chord_vocab_size': chord_tokenizer.next_token_id - CHORD_TOKEN_START,
        'chord_silence_token': CHORD_SILENCE_TOKEN,
        'note_to_token': {str(k): v for k, v in melody_tokenizer.note_to_token.items()},
        'token_to_note': {str(k): v for k, v in melody_tokenizer.token_to_note.items()},
        'melody_vocab_size': 257,  # Fixed size for MIDI
        'total_vocab_size': 257 + (chord_tokenizer.next_token_id - CHORD_TOKEN_START)
    }
    with open(output_dir / 'tokenizer_info.json', 'w') as f:
        json.dump(tokenizer_info, f, indent=2)

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
    song_sequences = {}  # Map song_id to its sequences
    
    for song_id, song_data in raw_data.items():
        sequences = preprocessor.process_song(song_id, song_data)
        if sequences:  # Only add songs that have valid sequences
            song_sequences[song_id] = sequences
    
    # Shuffle songs (not sequences) to avoid data leakage
    song_ids = list(song_sequences.keys())
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
    print(f"Vocabulary sizes:")
    print(f"  Melody tokens: {preprocessor.melody_tokenizer.next_token_id - MELODY_TOKEN_START}")
    print(f"  Chord tokens: {preprocessor.chord_tokenizer.next_token_id - CHORD_TOKEN_START}")
    print(f"  Total tokens: {preprocessor.melody_tokenizer.next_token_id - MELODY_TOKEN_START + preprocessor.chord_tokenizer.next_token_id - CHORD_TOKEN_START}")

if __name__ == "__main__":
    main() 