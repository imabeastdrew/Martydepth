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
MELODY_TOKEN_START = 0      # Melody tokens: 0-176 (177 tokens)
CHORD_TOKEN_START = 177     # Chord tokens: 177-4752 (4576 tokens)
SILENCE_TOKEN = 0           # Shared silence token

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
        self.next_token_id = CHORD_TOKEN_START
        self.silence_token = SILENCE_TOKEN
        
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
            
            # Ensure we don't exceed vocabulary size
            if hold_token >= 4753:  # Total vocabulary size
                raise ValueError("Chord vocabulary size exceeded")
            
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

class MelodyTokenizer:
    def __init__(self):
        self.note_to_token = {}  # Maps (pitch_class, octave) → (onset_token, hold_token)
        self.token_to_note = {}  # Maps token_id → (pitch_class, octave, is_onset)
        self.next_token_id = MELODY_TOKEN_START + 1  # Start after silence token
        self.silence_token = SILENCE_TOKEN
        self.note_to_token[(-1, -1)] = (self.silence_token, self.silence_token)
        self.token_to_note[self.silence_token] = (-1, -1, False)
        
    def encode_note(self, pitch_class: int, octave: int) -> Tuple[int, int]:
        """Encode a note into onset and hold token IDs"""
        note_key = (pitch_class, octave)
        
        if note_key not in self.note_to_token:
            # Create new onset and hold tokens
            onset_token = self.next_token_id
            hold_token = self.next_token_id + 1
            self.next_token_id += 2
            
            # Ensure we don't exceed melody vocabulary size
            if hold_token >= CHORD_TOKEN_START:  # Must stay within melody range
                raise ValueError("Melody vocabulary size exceeded")
            
            self.note_to_token[note_key] = (onset_token, hold_token)
            self.token_to_note[onset_token] = (pitch_class, octave, True)
            self.token_to_note[hold_token] = (pitch_class, octave, False)
            
        return self.note_to_token[note_key]
    
    def decode_note(self, token: int) -> Tuple[int, int, bool]:
        """Decode a token ID back into note components and onset status"""
        if token == self.silence_token:
            return -1, -1, False
        if token not in self.token_to_note:
            raise ValueError(f"Unknown token ID: {token}")
            
        pitch_class, octave, is_onset = self.token_to_note[token]
        return pitch_class, octave, is_onset

class FramePreprocessor:
    def __init__(self, sequence_length: int = 256):
        self.sequence_length = sequence_length
        self.chord_tokenizer = ChordTokenizer()
        self.melody_tokenizer = MelodyTokenizer()
        
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
        """Convert a song to frame sequences using the existing pipeline"""
        num_beats = int(song_data['annotations']['num_beats'])  # Ensure integer
        total_frames = num_beats * 4  # 4 frames per beat
        
        # Initialize with silence tokens
        melody_tokens = [self.melody_tokenizer.silence_token] * total_frames
        chord_tokens = [self.chord_tokenizer.silence_token] * total_frames
        
        # Convert melody notes
        for note in song_data['annotations']['melody']:
            onset_frame = int(note['onset'] * 4)
            offset_frame = int(note['offset'] * 4)
            
            # Create note tokens (onset and hold)
            pitch_class = note['pitch_class']
            octave = note['octave']
            onset_token, hold_token = self.melody_tokenizer.encode_note(pitch_class, octave)
            
            # Fill frames
            if onset_frame < total_frames:
                melody_tokens[onset_frame] = onset_token
                for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                    melody_tokens[frame] = hold_token
        
        # Convert chords using the tokenizer
        for chord in song_data['annotations']['harmony']:
            onset_frame = int(chord['onset'] * 4)
            offset_frame = int(chord['offset'] * 4)
            
            try:
                # Get chord components
                root = chord['root_pitch_class']
                intervals = chord['root_position_intervals']
                inversion = chord.get('inversion', 0)
                
                # Encode chord using tokenizer (now returns onset and hold tokens)
                onset_token, hold_token = self.chord_tokenizer.encode_chord(root, intervals, inversion)
                
                # Fill frames
                if onset_frame < total_frames:
                    chord_tokens[onset_frame] = onset_token
                    for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                        chord_tokens[frame] = hold_token
                        
            except (KeyError, ValueError) as e:
                print(f"Warning: Invalid chord in song: {e}")
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
                       melody_tokenizer: MelodyTokenizer, output_dir: Path):
    """Save processed sequences and tokenizer info"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert sequences to dictionaries for serialization
    sequence_dicts = [seq.to_dict() for seq in sequences]
    
    # Save sequences
    split_name = output_dir.name  # Get 'train', 'valid', or 'test' from path
    with open(output_dir / f'frame_sequences_{split_name}.pkl', 'wb') as f:
        pickle.dump(sequence_dicts, f)
    
    # Save tokenizer info
    tokenizer_info = {
        'chord_to_token': {str(k): v for k, v in chord_tokenizer.chord_to_token.items()},
        'token_to_chord': {str(k): v for k, v in chord_tokenizer.token_to_chord.items()},
        'chord_vocab_size': chord_tokenizer.next_token_id - CHORD_TOKEN_START,
        'note_to_token': {str(k): v for k, v in melody_tokenizer.note_to_token.items()},
        'token_to_note': {str(k): v for k, v in melody_tokenizer.token_to_note.items()},
        'melody_vocab_size': melody_tokenizer.next_token_id - MELODY_TOKEN_START,
        'total_vocab_size': (chord_tokenizer.next_token_id - CHORD_TOKEN_START + 
                           melody_tokenizer.next_token_id - MELODY_TOKEN_START)
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