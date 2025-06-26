#!/usr/bin/env python3
"""
Frame-based Preprocessing: Convert raw chord data into frame sequences
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import random

from src.config.tokenization_config import (
    SILENCE_TOKEN,
    CHORD_TOKEN_START,
    CHORD_SILENCE_TOKEN,
    CHORD_ONSET_HOLD_START,
    MELODY_ONSET_HOLD_START,
    MIN_MIDI_NOTE,
    MAX_MIDI_NOTE,
    UNIQUE_MIDI_NOTES,
    MELODY_VOCAB_SIZE,
    PAD_TOKEN,
    midi_to_onset_hold_tokens,
    token_to_midi_and_type,
)
from src.data.datastructures import FrameSequence

def validate_sequence(sequence: FrameSequence) -> Dict[str, Any]:
    """Validate processed sequence for common issues"""
    validation_results = {
        "valid": True,
        "issues": [],
        "stats": {}
    }
    
    melody_tokens = sequence.melody_tokens
    chord_tokens = sequence.chord_tokens
    
    # Remove padding for analysis
    non_pad_mask = melody_tokens != PAD_TOKEN
    melody_content = melody_tokens[non_pad_mask]
    chord_content = chord_tokens[non_pad_mask]
    
    # Check for completely silent sequences
    if len(melody_content) > 0:
        melody_silence_ratio = np.sum(melody_content == SILENCE_TOKEN) / len(melody_content)
        chord_silence_ratio = np.sum(chord_content == CHORD_SILENCE_TOKEN) / len(chord_content)
        
        # Flag sequences that are completely silent
        if melody_silence_ratio >= 1.0 and chord_silence_ratio >= 1.0:
            validation_results["valid"] = False
            validation_results["issues"].append("Completely silent sequence (100% melody + chord silence)")
        
        # Flag sequences with 100% melody silence (even if chords exist)
        elif melody_silence_ratio >= 1.0:
            validation_results["valid"] = False
            validation_results["issues"].append("Melody-only silent sequence (100% melody silence)")
        
        # Flag sequences with 100% chord silence (even if melody exists)  
        elif chord_silence_ratio >= 1.0:
            validation_results["valid"] = False
            validation_results["issues"].append("Chord-only silent sequence (100% chord silence)")
        
        # Flag sequences with very limited content
        if len(melody_content) < 10:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Very short content length: {len(melody_content)}")
            
        # Check for musical content
        unique_melody = len(np.unique(melody_content[melody_content != SILENCE_TOKEN]))
        unique_chords = len(np.unique(chord_content[chord_content != CHORD_SILENCE_TOKEN]))
        
        if unique_melody == 0 and unique_chords == 0:
            validation_results["valid"] = False
            validation_results["issues"].append("No musical content (all silence tokens)")
    else:
        validation_results["valid"] = False
        validation_results["issues"].append("Empty sequence after padding removal")
    
    # Check melody token ranges (interleaved)
    invalid_melody = melody_tokens[
        (melody_tokens < 0) | 
        ((melody_tokens != SILENCE_TOKEN) & (melody_tokens != PAD_TOKEN) & 
         ((melody_tokens < MELODY_ONSET_HOLD_START) | (melody_tokens >= PAD_TOKEN)))
    ]
    if len(invalid_melody) > 0:
        validation_results["valid"] = False
        validation_results["issues"].append(f"Invalid melody tokens: {np.unique(invalid_melody)}")
    
    # Check chord token ranges  
    invalid_chords = chord_tokens[(chord_tokens < CHORD_TOKEN_START) & (chord_tokens != PAD_TOKEN)]
    if len(invalid_chords) > 0:
        validation_results["valid"] = False
        validation_results["issues"].append(f"Invalid chord tokens: {np.unique(invalid_chords)}")
    
    # Check for PAD tokens in chord sequences (should use CHORD_SILENCE_TOKEN instead)
    pad_in_chords = np.sum(chord_tokens == PAD_TOKEN)
    if pad_in_chords > 0:
        validation_results["valid"] = False
        validation_results["issues"].append(f"PAD tokens found in chord sequence: {pad_in_chords}")
    
    # Calculate statistics
    content_length = np.sum(non_pad_mask)
    
    validation_results["stats"] = {
        "content_length": int(content_length),
        "padding_length": int(256 - content_length),
        "padding_ratio": float(1 - content_length / 256),
        "melody_silence_ratio": float(np.sum(melody_content == SILENCE_TOKEN) / len(melody_content)) if len(melody_content) > 0 else 1.0,
        "chord_silence_ratio": float(np.sum(chord_content == CHORD_SILENCE_TOKEN) / len(chord_content)) if len(chord_content) > 0 else 1.0,
    }
    
    return validation_results

class ChordTokenizer:
    def __init__(self):
        self.chord_to_token = {}
        self.token_to_chord = {}
        self.next_token_id = CHORD_ONSET_HOLD_START
        self.silence_token = CHORD_SILENCE_TOKEN
        
    def _get_chord_key(self, root: int, intervals: List[int], inversion: int) -> Tuple[int, Tuple[int, ...], int]:
        if not intervals:
            raise ValueError("Chord must have at least one interval")
        if not all(isinstance(x, int) for x in intervals):
            raise ValueError("All intervals must be integers")
        if not isinstance(root, int) or not isinstance(inversion, int):
            raise ValueError("Root and inversion must be integers")
            
        return (root, tuple(sorted(intervals)), inversion)
    
    def encode_chord(self, root: int, intervals: List[int], inversion: int) -> Tuple[int, int]:
        chord_key = self._get_chord_key(root, intervals, inversion)
        
        if chord_key not in self.chord_to_token:
            onset_token = self.next_token_id
            hold_token = self.next_token_id + 1
            self.next_token_id += 2
            
            self.chord_to_token[chord_key] = (onset_token, hold_token)
            self.token_to_chord[onset_token] = (*chord_key, True)
            self.token_to_chord[hold_token] = (*chord_key, False)
            
        return self.chord_to_token[chord_key]
    
    def decode_chord(self, token: int) -> Tuple[int, List[int], int, bool]:
        if token == self.silence_token:
            return -1, [], -1, False
        if token not in self.token_to_chord:
            raise ValueError(f"Unknown token ID: {token}")
            
        root, intervals, inversion, is_onset = self.token_to_chord[token]
        return root, list(intervals), inversion, is_onset

class MIDITokenizer:
    def __init__(self):
        self.silence_token = SILENCE_TOKEN
        self.min_midi_note = MIN_MIDI_NOTE
        self.max_midi_note = MAX_MIDI_NOTE
        
        # Build token mappings using interleaved approach
        self.note_to_token = {}
        self.token_to_note = {self.silence_token: (-1, False)}
        
        for midi_num in range(self.min_midi_note, self.max_midi_note + 1):
            onset_token, hold_token = midi_to_onset_hold_tokens(midi_num)
            
            self.note_to_token[midi_num] = (onset_token, hold_token)
            self.token_to_note[onset_token] = (midi_num, True)
            self.token_to_note[hold_token] = (midi_num, False)

    def encode_midi_note(self, midi_number: int) -> Tuple[int, int]:
        """Encode MIDI note to onset/hold token pair"""
        if not (self.min_midi_note <= midi_number <= self.max_midi_note):
            return self.silence_token, self.silence_token
        
        return self.note_to_token[midi_number]

    def decode_note(self, token: int) -> Tuple[int, int, bool]:
        """Decode token to (midi_number, unused, is_onset)"""
        if token == self.silence_token:
            return -1, -1, False
        if token not in self.token_to_note:
            raise ValueError(f"Unknown token ID: {token}")
        
        midi_num, is_onset = self.token_to_note[token]
        return midi_num, -1, is_onset

    @property
    def next_token_id(self):
        return CHORD_TOKEN_START

class FramePreprocessor:
    def __init__(self, sequence_length: int = 256):
        self.sequence_length = sequence_length
        self.chord_tokenizer = ChordTokenizer()
        self.melody_tokenizer = MIDITokenizer()
        
    def convert_song_to_frames(self, song_data: Dict) -> Tuple[List[int], List[int]]:
        num_beats = int(song_data.get('annotations', {}).get('num_beats', 0))
        if num_beats == 0:
            return [], []
        total_frames = num_beats * 4
        
        melody_tokens = [self.melody_tokenizer.silence_token] * total_frames
        chord_tokens = [self.chord_tokenizer.silence_token] * total_frames
        
        for note in song_data.get('annotations', {}).get('melody', []):
            onset_frame = int(note['onset'] * 4)
            offset_frame = int(note['offset'] * 4)
            pitch_class = note['pitch_class']
            octave = note['octave']
            midi_number = (octave + 1) * 12 + pitch_class
            if not (0 <= midi_number <= self.melody_tokenizer.max_midi_note):
                continue

            onset_token, hold_token = self.melody_tokenizer.encode_midi_note(midi_number)
            if onset_frame < total_frames:
                melody_tokens[onset_frame] = onset_token
                for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                    melody_tokens[frame] = hold_token

        for chord in song_data.get('annotations', {}).get('harmony', []):
            onset_frame = int(chord['onset'] * 4)
            offset_frame = int(chord['offset'] * 4)
            try:
                root = chord.get('root_pitch_class')
                intervals = chord.get('root_position_intervals', [])
                inversion = chord.get('inversion', 0)
                if root is None or not intervals:
                    continue
                onset_token, hold_token = self.chord_tokenizer.encode_chord(root, intervals, inversion)
                if onset_frame < total_frames:
                    chord_tokens[onset_frame] = onset_token
                    for frame in range(onset_frame + 1, min(offset_frame, total_frames)):
                        chord_tokens[frame] = hold_token
            except (KeyError, ValueError, TypeError):
                continue
        return melody_tokens, chord_tokens
    
    def process_song(self, song_id: str, song_data: Dict) -> List[FrameSequence]:
        sequences = []
        
        if 'MELODY' not in song_data.get('tags', []) or 'HARMONY' not in song_data.get('tags', []):
            return sequences
        
        melody_tokens, chord_tokens = self.convert_song_to_frames(song_data)
        
        if not melody_tokens or not chord_tokens:
            return sequences
        
        for start_idx in range(0, len(melody_tokens), self.sequence_length):
            end_idx = min(start_idx + self.sequence_length, len(melody_tokens))
            
            melody_seq = melody_tokens[start_idx:end_idx]
            chord_seq = chord_tokens[start_idx:end_idx]
            
            if len(melody_seq) < self.sequence_length:
                pad_len = self.sequence_length - len(melody_seq)
                melody_seq.extend([PAD_TOKEN] * pad_len)
                chord_seq.extend([CHORD_SILENCE_TOKEN] * pad_len)
            
            sequence = FrameSequence(
                melody_tokens=np.array(melody_seq),
                chord_tokens=np.array(chord_seq),
                frame_times=np.zeros(self.sequence_length),
                frame_durations=np.zeros(self.sequence_length),
                song_id=song_id,
                start_frame=start_idx
            )
            
            # Validate sequence before adding
            validation = validate_sequence(sequence)
            if validation["valid"]:
                sequences.append(sequence)
            else:
                print(f"Filtered sequence {song_id}:{start_idx} - {', '.join(validation['issues'])}")
        
        return sequences

def save_processed_data(sequences: List[FrameSequence], chord_tokenizer: ChordTokenizer, 
                       melody_tokenizer: MIDITokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, seq in enumerate(sequences):
        sequence_filename = f"sequence_{i:06d}.pkl"
        with open(output_dir / sequence_filename, 'wb') as f:
            pickle.dump(seq, f)

    tokenizer_info = {
        "melody_vocab_size": MELODY_VOCAB_SIZE,
        "chord_vocab_size": chord_tokenizer.next_token_id - CHORD_TOKEN_START,
        "total_vocab_size": chord_tokenizer.next_token_id,
        "pad_token_id": PAD_TOKEN,
        "chord_token_start": CHORD_TOKEN_START,
        "chord_silence_token": CHORD_SILENCE_TOKEN,
        "midi_range": {"min": MIN_MIDI_NOTE, "max": MAX_MIDI_NOTE},
        "unique_midi_notes": UNIQUE_MIDI_NOTES,
        "token_to_chord": chord_tokenizer.token_to_chord,
        "token_to_note": melody_tokenizer.token_to_note,
    }

    with open(output_dir / 'tokenizer_info.json', 'w') as f:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(tokenizer_info, f, cls=NpEncoder, indent=4)
    print("Tokenizer info saved.")

def main():
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    raw_data_path = data_dir / "raw" / "Hooktheory copy.json"
    output_dir = data_dir / "interim"
    
    print("Loading raw data...")
    try:
        with open(raw_data_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the Hooktheory dataset is placed in the `data/raw` directory.")
        return
    
    preprocessor = FramePreprocessor(sequence_length=256)
    
    print("Processing songs into frame sequences...")
    song_sequences = defaultdict(list)
    
    for song_id, song_data in raw_data.items():
        if not isinstance(song_data, dict) or 'annotations' not in song_data:
            continue
            
        sequences = preprocessor.process_song(song_id, song_data)
        if sequences:
            song_sequences[song_id] = sequences
    
    song_ids = list(song_sequences.keys())
    random.seed(42)
    random.shuffle(song_ids)
    
    n = len(song_ids)
    train_idx, valid_idx = int(0.8 * n), int(0.9 * n)
    
    train_songs = song_ids[:train_idx]
    valid_songs = song_ids[train_idx:valid_idx]
    test_songs = song_ids[valid_idx:]
    
    train_sequences = [seq for song_id in train_songs for seq in song_sequences[song_id]]
    valid_sequences = [seq for song_id in valid_songs for seq in song_sequences[song_id]]
    test_sequences = [seq for song_id in test_songs for seq in song_sequences[song_id]]
    
    print("Saving processed data...")
    save_processed_data(train_sequences, preprocessor.chord_tokenizer, 
                       preprocessor.melody_tokenizer, output_dir / 'train')
    save_processed_data(valid_sequences, preprocessor.chord_tokenizer,
                       preprocessor.melody_tokenizer, output_dir / 'valid')
    save_processed_data(test_sequences, preprocessor.chord_tokenizer,
                       preprocessor.melody_tokenizer, output_dir / 'test')
    
    print("\nPreprocessing complete!")
    print(f"Total songs: {len(song_sequences)}")
    print(f"Train/Valid/Test songs: {len(train_songs)}/{len(valid_songs)}/{len(test_songs)}")
    print(f"Train/Valid/Test sequences: {len(train_sequences)}/{len(valid_sequences)}/{len(test_sequences)}")
    
    with open(output_dir / 'train' / 'tokenizer_info.json', 'r') as f:
        final_tokenizer_info = json.load(f)

    print(f"Vocabulary sizes:")
    print(f"  Melody tokens: {final_tokenizer_info['melody_vocab_size']}")
    print(f"  Chord tokens: {final_tokenizer_info['chord_vocab_size']}")
    print(f"  Total tokens: {final_tokenizer_info['total_vocab_size']}")

if __name__ == "__main__":
    main() 