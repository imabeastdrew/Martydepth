#!/usr/bin/env python3
"""
Frame-based Preprocessing: Convert raw chord data into frame sequences
"""

import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np

from src.config.tokenization_config import (
    SILENCE_TOKEN,
    CHORD_TOKEN_START,
    CHORD_SILENCE_TOKEN,
    MELODY_ONSET_HOLD_START,
    MIN_MIDI_NOTE,
    MAX_MIDI_NOTE,
    UNIQUE_MIDI_NOTES,
    MELODY_VOCAB_SIZE,
    PAD_TOKEN,
    midi_to_onset_hold_tokens,
    midi_to_token_index,
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
        # Count silence tokens and hold tokens
        melody_silence_count = np.sum(melody_content == SILENCE_TOKEN)
        melody_hold_count = np.sum(melody_content >= MELODY_ONSET_HOLD_START)
        melody_onset_count = len(melody_content) - melody_silence_count - melody_hold_count
        
        # Calculate ratios
        melody_silence_ratio = melody_silence_count / len(melody_content)
        melody_hold_ratio = melody_hold_count / len(melody_content)
        melody_onset_ratio = melody_onset_count / len(melody_content)
        
        # Flag sequences that are completely silent
        if melody_silence_ratio == 1.0:
            validation_results["valid"] = False
            validation_results["issues"].append("completely_silent_melody")
            
        # Store statistics
        validation_results["stats"].update({
            "melody_silence_ratio": melody_silence_ratio,
            "melody_hold_ratio": melody_hold_ratio,
            "melody_onset_ratio": melody_onset_ratio,
            "sequence_length": len(melody_content)
        })
    
    return validation_results

def get_chord_intervals(chord_type):
    """Get intervals for a chord type number."""
    # Chord type mapping based on common chord types
    chord_intervals = {
        0: [0, 4, 7],      # Major
        1: [0, 3, 7],      # Minor
        2: [0, 4, 7, 10],  # Dominant 7th
        3: [0, 4, 7, 11],  # Major 7th
        4: [0, 3, 7, 10],  # Minor 7th
        5: [0, 3, 6],      # Diminished
        6: [0, 4, 8],      # Augmented
        7: [0, 3, 6, 9],   # Diminished 7th
        8: [0, 3, 7, 11],  # Minor/Major 7th
        9: [0, 4, 7, 9],   # Major 6th
        10: [0, 3, 7, 9],  # Minor 6th
    }
    return chord_intervals.get(chord_type, [0, 4, 7])  # Default to major triad if unknown

class ChordTokenizer:
    """Tokenizer for chord sequences."""
    
    def __init__(self):
        # List to store unique chord patterns
        self.chord_patterns = []
        # Start token after melody tokens
        self.token_offset = CHORD_TOKEN_START
        # Track unique interval patterns for analysis
        self.interval_patterns = set()
        # Special tokens - use 0 for silence (same as melody) instead of PAD_TOKEN
        self.silence_token = 0  # Use 0 for chord silence, same as melody silence
        self.pad_token = PAD_TOKEN
        
    def _find_or_add_pattern(self, root: int, intervals: List[int], inversion: int) -> int:
        """Find existing pattern or add new one and return its token."""
        # Create pattern tuple (root, intervals as tuple for comparison, inversion)
        pattern = (root, tuple(intervals), inversion)
        
        # Try to find existing pattern
        try:
            idx = self.chord_patterns.index(pattern)
            return idx + self.token_offset
        except ValueError:
            # Add new pattern
            self.chord_patterns.append(pattern)
            # Track unique interval patterns for analysis
            self.interval_patterns.add(tuple(sorted(set(intervals))))
            return len(self.chord_patterns) - 1 + self.token_offset
    
    def encode_chord(self, chord_data: Dict) -> Tuple[int, int]:
        """Encode a chord into onset and hold tokens."""
        if not chord_data:
            return self.silence_token, self.silence_token
            
        root = chord_data['root_pitch_class']
        intervals = chord_data['root_position_intervals']
        inversion = chord_data.get('inversion', 0)
        
        onset_token = self._find_or_add_pattern(root, intervals, inversion)
        # FIXED: Use a fixed offset instead of dynamic len(chord_patterns)
        # Hold tokens start after all possible onset tokens
        hold_token = onset_token + self.get_max_patterns()
        
        return onset_token, hold_token
    
    def decode_token(self, token: int) -> Optional[Dict]:
        """Decode a token back to chord information."""
        if token == self.silence_token:
            return None
        if token == self.pad_token:
            return None
            
        # Check if it's a hold token using fixed offset
        max_patterns = self.get_max_patterns()
        is_hold = token >= self.token_offset + max_patterns
        if is_hold:
            token -= max_patterns  # Subtract fixed offset, not len(chord_patterns)
            
        # Get pattern from list
        pattern_idx = token - self.token_offset
        if pattern_idx < 0 or pattern_idx >= len(self.chord_patterns):
            return None
            
        root, intervals, inversion = self.chord_patterns[pattern_idx]
        
        return {
            'root_pitch_class': root,
            'root_position_intervals': list(intervals),
            'inversion': inversion,
            'is_hold': is_hold
        }
    
    def _convert_to_standard_intervals(self, consecutive_intervals):
        """Convert consecutive intervals to standard intervals from root.
        
        Args:
            consecutive_intervals: List of intervals between consecutive notes
            
        Returns:
            List of intervals from the root note
        """
        if not consecutive_intervals:
            return [0]  # Just the root
            
        # Convert consecutive intervals to cumulative intervals
        standard_intervals = [0]  # Start with root
        current_sum = 0
        for interval in consecutive_intervals:
            current_sum += interval
            standard_intervals.append(current_sum)
        
        return standard_intervals

    def is_melody_note_in_chord(self, melody_note: int, chord_token: int) -> bool:
        """Check if a melody note is part of a chord."""
        if chord_token == self.silence_token or chord_token == self.pad_token:
            return False
            
        # Get chord pattern
        chord_info = self.decode_token(chord_token)
        if not chord_info:
            return False
            
        # Get all possible note positions for this chord
        root = chord_info['root_pitch_class']
        consecutive_intervals = chord_info['root_position_intervals']
        
        # Convert to standard intervals from root
        standard_intervals = self._convert_to_standard_intervals(consecutive_intervals)
        
        # Check if melody note (mod 12) matches any chord note
        melody_pitch_class = melody_note % 12
        chord_notes = {(root + interval) % 12 for interval in standard_intervals}
        
        return melody_pitch_class in chord_notes
    
    def get_vocab_size(self) -> int:
        """Get total vocabulary size including onset and hold tokens."""
        max_patterns = self.get_max_patterns()
        return max_patterns * 2  # Double for hold tokens (using fixed maximum)
    
    def save(self, save_dir: Path):
        """Save tokenizer information."""
        tokenizer_info = {
            'chord_patterns': self.chord_patterns,
            'token_offset': self.token_offset,
            'interval_patterns': list(self.interval_patterns),
            'vocab_size': self.get_vocab_size()
        }
        
        with open(save_dir / 'tokenizer_info.json', 'w') as f:
            json.dump(tokenizer_info, f)
    
    @classmethod
    def load(cls, load_dir: Path) -> 'ChordTokenizer':
        """Load tokenizer from saved information."""
        with open(load_dir / 'tokenizer_info.json', 'r') as f:
            info = json.load(f)
            
        tokenizer = cls()
        tokenizer.chord_patterns = info['chord_patterns']
        tokenizer.token_offset = info['token_offset']
        tokenizer.interval_patterns = set(tuple(p) for p in info['interval_patterns'])
        
        return tokenizer

    def get_max_patterns(self) -> int:
        """Get the maximum number of patterns that will be discovered.
        This provides a fixed offset for hold token calculation."""
        # Use a large fixed number to ensure hold tokens don't overlap with onset tokens
        # This should be larger than the actual number of unique chord patterns
        return 2500  # Estimated maximum unique chord patterns in the dataset

class MIDITokenizer:
    """Tokenizer for MIDI note sequences."""
    
    def __init__(self):
        """Initialize the tokenizer with predefined token ranges."""
        # Token ranges
        self.onset_tokens = range(0, UNIQUE_MIDI_NOTES)  # 0-87 for onsets
        self.silence_token = SILENCE_TOKEN  # 88 for silence
        self.hold_tokens = range(MELODY_ONSET_HOLD_START, MELODY_ONSET_HOLD_START + UNIQUE_MIDI_NOTES)  # 89-176 for holds
        self.pad_token = PAD_TOKEN  # 177 for padding
        
        # Create mappings
        self.note_to_token = {}  # Maps MIDI note to (onset, hold) token pair
        self.token_to_note = {}  # Maps token to MIDI note
        
        # Initialize mappings
        for midi_note in range(MIN_MIDI_NOTE, MAX_MIDI_NOTE + 1):
            onset_token = midi_note - MIN_MIDI_NOTE
            hold_token = onset_token + MELODY_ONSET_HOLD_START
            
            self.note_to_token[midi_note] = (onset_token, hold_token)
            self.token_to_note[onset_token] = midi_note
            self.token_to_note[hold_token] = midi_note
    
    def encode_midi_note(self, midi_note: int) -> Tuple[int, int]:
        """Convert MIDI note to onset/hold token pair."""
        if midi_note not in self.note_to_token:
            return self.silence_token, self.silence_token
        return self.note_to_token[midi_note]
    
    def decode_token(self, token: int) -> Tuple[Optional[int], bool]:
        """Decode token to MIDI note and onset/hold flag.
        
        Args:
            token: Token to decode
            
        Returns:
            Tuple of (midi_note, is_onset) or (None, False) for special tokens
        """
        if token == self.silence_token or token == self.pad_token:
            return None, False
            
        midi_note = self.token_to_note.get(token)
        if midi_note is None:
            return None, False
            
        is_onset = token < MELODY_ONSET_HOLD_START
        return midi_note, is_onset

class FramePreprocessor:
    """Frame-based preprocessor that converts songs to sixteenth-note resolution sequences.
    
    Uses 4 frames per beat to capture sub-beat timing and properly implements
    onset/hold token logic for both melody and chords based on fractional timing data.
    This enables much higher chord length entropy and better rhythmic precision.
    """
    def __init__(self, sequence_length: int = 256):
        self.sequence_length = sequence_length
        self.chord_tokenizer = ChordTokenizer()
        self.melody_tokenizer = MIDITokenizer()
        
    def convert_song_to_frames(self, song_data: Dict) -> Tuple[List[int], List[int]]:
        """Convert song data to frame sequences using sixteenth-note resolution.
        
        Args:
            song_data (dict): Raw song data
            
        Returns:
            tuple: (melody_sequence, chord_sequence)
        """
        annotations = song_data.get('annotations', {})
        if not annotations:
            return [], []
            
        num_beats = annotations.get('num_beats', 0)
        if num_beats == 0:
            return [], []
            
        # Convert to sixteenth-note resolution (4 frames per beat)
        num_frames = int(num_beats * 4)
        
        melody_seq = []
        chord_seq = []
        
        # Get melody and harmony data directly from annotations
        melody_data = annotations.get('melody', [])
        harmony_data = annotations.get('harmony', [])
        
        if not melody_data or not harmony_data:
            return [], []
        
        # Track current active notes/chords for hold token logic
        current_melody_note = None
        current_melody_onset_frame = -1
        current_chord_data = None
        current_chord_onset_frame = -1
        has_invalid_notes = False
        
        # Create sixteenth-note aligned sequences
        for frame_idx in range(num_frames):
            frame_beat = frame_idx / 4.0  # Convert frame index back to beat notation
            
            # Find melody note active at this frame
            new_melody_note = None
            for note in melody_data:
                if note['onset'] <= frame_beat < note['offset']:
                    new_melody_note = note['pitch_class'] + (note['octave'] * 12)
                    # Check if note is in valid range
                    if not (MIN_MIDI_NOTE <= new_melody_note <= MAX_MIDI_NOTE):
                        has_invalid_notes = True
                    break
            
            # Melody token logic: onset vs hold vs silence
            if new_melody_note is not None:
                if new_melody_note != current_melody_note:
                    # New note - use onset token
                    onset_token, hold_token = self.melody_tokenizer.encode_midi_note(new_melody_note)
                    melody_seq.append(onset_token)
                    current_melody_note = new_melody_note
                    current_melody_onset_frame = frame_idx
                else:
                    # Same note continues - use hold token
                    onset_token, hold_token = self.melody_tokenizer.encode_midi_note(new_melody_note)
                    melody_seq.append(hold_token)
            else:
                # No note active - use silence
                melody_seq.append(SILENCE_TOKEN)
                current_melody_note = None
                current_melody_onset_frame = -1
            
            # Find chord active at this frame
            new_chord_data = None
            for chord in harmony_data:
                if chord['onset'] <= frame_beat < chord['offset']:
                    new_chord_data = chord
                    break
            
            # Chord token logic: onset vs hold vs silence
            if new_chord_data is not None:
                # Check if this is the same chord as previous frame
                if (current_chord_data is None or 
                    new_chord_data['root_pitch_class'] != current_chord_data['root_pitch_class'] or
                    new_chord_data['root_position_intervals'] != current_chord_data['root_position_intervals'] or
                    new_chord_data.get('inversion', 0) != current_chord_data.get('inversion', 0)):
                    # New chord - use onset token
                    onset_token, hold_token = self.chord_tokenizer.encode_chord(new_chord_data)
                    chord_seq.append(onset_token)
                    current_chord_data = new_chord_data
                    current_chord_onset_frame = frame_idx
                else:
                    # Same chord continues - use hold token
                    onset_token, hold_token = self.chord_tokenizer.encode_chord(new_chord_data)
                    chord_seq.append(hold_token)
            else:
                # No chord active - use silence
                chord_seq.append(self.chord_tokenizer.silence_token)
                current_chord_data = None
                current_chord_onset_frame = -1
        
        # If sequence has invalid notes, return empty lists to filter it out
        if has_invalid_notes:
            print(f"Filtered sequence - invalid MIDI notes")
            return [], []
        
        # Pad sequences to desired length (256 frames)
        if len(melody_seq) < self.sequence_length:
            pad_len = self.sequence_length - len(melody_seq)
            melody_seq.extend([PAD_TOKEN] * pad_len)
            chord_seq.extend([PAD_TOKEN] * pad_len)
        else:
            # Truncate if longer than sequence_length
            melody_seq = melody_seq[:self.sequence_length]
            chord_seq = chord_seq[:self.sequence_length]
        
        return melody_seq, chord_seq
    
    def process_song(self, song_id: str, song_data: Dict) -> List[FrameSequence]:
        """Process a song into frame sequences.
        
        Args:
            song_id (str): Unique identifier for the song
            song_data (dict): Raw song data
            
        Returns:
            list: List of FrameSequence objects
        """
        melody_seq, chord_seq = self.convert_song_to_frames(song_data)
        if not melody_seq or not chord_seq:
            return []
            
        # Create sequence object
        sequence = FrameSequence(
            melody_tokens=np.array(melody_seq),
            chord_tokens=np.array(chord_seq),
            frame_times=np.zeros(self.sequence_length),
            frame_durations=np.zeros(self.sequence_length),
            song_id=song_id,
            start_frame=0
        )
        
        # Validate sequence before adding
        validation = validate_sequence(sequence)
        if validation["valid"]:
            return [sequence]
        else:
            print(f"Filtered sequence {song_id} - {', '.join(validation['issues'])}")
            return []

def save_processed_data(sequences: List[FrameSequence], chord_tokenizer: ChordTokenizer, 
                       melody_tokenizer: MIDITokenizer, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, seq in enumerate(sequences):
        sequence_filename = f"sequence_{i:06d}.pkl"
        with open(output_dir / sequence_filename, 'wb') as f:
            pickle.dump(seq, f)

    # Build token_to_chord mapping
    token_to_chord = {}
    for token in range(chord_tokenizer.token_offset, chord_tokenizer.token_offset + chord_tokenizer.get_vocab_size()):
        chord_info = chord_tokenizer.decode_token(token)
        if chord_info is not None:
            token_to_chord[str(token)] = {
                'root_pitch_class': chord_info['root_pitch_class'],
                'root_position_intervals': chord_info['root_position_intervals'],
                'inversion': chord_info['inversion'],
                'is_hold': chord_info['is_hold']
            }

    # Calculate the actual maximum token ID to determine total vocab size
    max_token_id = 0
    for token_str in token_to_chord.keys():
        max_token_id = max(max_token_id, int(token_str))
    
    # Total vocab size should be max_token_id + 1 to include all tokens from 0 to max_token_id
    total_vocab_size = max_token_id + 1

    tokenizer_info = {
        "melody_vocab_size": MELODY_VOCAB_SIZE,
        "chord_vocab_size": chord_tokenizer.get_vocab_size(),
        "total_vocab_size": total_vocab_size,
        "pad_token_id": PAD_TOKEN,
        "chord_token_start": CHORD_TOKEN_START,
        "chord_silence_token": chord_tokenizer.silence_token,  # Use actual silence token (0)
        "midi_range": {"min": MIN_MIDI_NOTE, "max": MAX_MIDI_NOTE},
        "unique_midi_notes": UNIQUE_MIDI_NOTES,
        "token_to_chord": token_to_chord
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
    
    # Track statistics
    total_melody_notes = 0
    melody_in_chord = 0
    unique_chords = set()
    inversion_counts = defaultdict(int)
    
    for song_id, song_data in raw_data.items():
        if not isinstance(song_data, dict):
            continue
            
        annotations = song_data.get('annotations', {})
        if not annotations:
            continue
            
        melody_data = annotations.get('melody')
        harmony_data = annotations.get('harmony')
        
        if not melody_data or not harmony_data:
            continue
            
        # Track chord patterns in this song
        for chord in harmony_data:
            root = chord['root_pitch_class']
            intervals = chord['root_position_intervals']
            inversion = chord.get('inversion', 0)
            unique_chords.add((root, tuple(intervals), inversion))
            inversion_counts[inversion] += 1
            
        sequences = preprocessor.process_song(song_id, song_data)
        if sequences:
            song_sequences[song_id] = sequences
            
            # Analyze melody-in-chord ratio for this sequence
            for seq in sequences:
                melody_tokens = seq.melody_tokens
                chord_tokens = seq.chord_tokens
                
                for m_token, c_token in zip(melody_tokens, chord_tokens):
                    if m_token != SILENCE_TOKEN and m_token != PAD_TOKEN:
                        total_melody_notes += 1
                        # Get MIDI note number from token
                        midi_note, is_onset = preprocessor.melody_tokenizer.decode_token(m_token)
                        if midi_note is not None:  # Skip silence/pad tokens
                            if preprocessor.chord_tokenizer.is_melody_note_in_chord(
                                midi_note,  # Pass the actual MIDI note
                                c_token
                            ):
                                melody_in_chord += 1
    
    # Print chord analysis
    print("\nChord Analysis:")
    print(f"Total unique chord patterns: {len(unique_chords)}")
    print("\nInversion distribution:")
    for inv, count in sorted(inversion_counts.items()):
        print(f"  Inversion {inv}: {count} occurrences")
        
    # Print melody-in-chord analysis
    print("\nMelody-in-Chord Analysis:")
    if total_melody_notes > 0:
        in_chord_ratio = (melody_in_chord / total_melody_notes) * 100
        print(f"Total melody notes: {total_melody_notes}")
        print(f"Notes in chord: {melody_in_chord}")
        print(f"Melody-in-chord ratio: {in_chord_ratio:.2f}%")
    
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
    
    print("\nSaving processed data...")
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

    print(f"\nVocabulary sizes:")
    print(f"  Melody tokens: {final_tokenizer_info['melody_vocab_size']}")
    print(f"  Chord tokens: {final_tokenizer_info['chord_vocab_size']}")
    print(f"  Total tokens: {final_tokenizer_info['total_vocab_size']}")
    
    # Print interval pattern analysis
    print("\nUnique interval patterns:")
    for pattern in sorted(preprocessor.chord_tokenizer.interval_patterns):
        print(f"  {pattern}")

if __name__ == "__main__":
    main() 