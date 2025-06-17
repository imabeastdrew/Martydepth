#!/usr/bin/env python3
"""
Evaluation metrics for generated sequences
"""

from typing import Dict, List
import numpy as np
from collections import Counter
from scipy.stats import entropy

from src.config.tokenization_config import (
    SILENCE_TOKEN,
    MELODY_VOCAB_SIZE,
    MIDI_ONSET_START,
    MIDI_HOLD_START,
    MAX_MIDI_NOTE,
    CHORD_TOKEN_START,
)

def parse_sequences(sequences: List[np.ndarray], tokenizer_info: Dict):
    """
    Parses token sequences into structured musical events.
    For now, this is a simplified parser.
    """
    parsed_data = []
    for seq in sequences:
        notes = []
        chords = []
        active_notes = {}  # pitch -> start_time

        for time_step, token in enumerate(seq):
            # Odd timesteps are melody
            if time_step % 2 == 1:
                if MIDI_ONSET_START <= token < MIDI_HOLD_START:
                    pitch = token - MIDI_ONSET_START
                    if pitch in active_notes: # End previous note
                        notes.append({'pitch': pitch, 'start': active_notes.pop(pitch), 'end': time_step})
                    active_notes[pitch] = time_step # Start new note
                elif MIDI_HOLD_START <= token < CHORD_TOKEN_START:
                    pitch = token - MIDI_HOLD_START
                    # This is a hold token, we just continue the note
                elif token == SILENCE_TOKEN and len(active_notes) > 0:
                     # End all active notes on silence
                    for pitch, start_time in list(active_notes.items()):
                        notes.append({'pitch': pitch, 'start': start_time, 'end': time_step})
                        del active_notes[pitch]

            # Even timesteps are chords
            else:
                if token >= CHORD_TOKEN_START:
                    # Append previous chord if it exists and is different
                    if not chords or chords[-1]['token'] != token:
                        if chords:
                            chords[-1]['end'] = time_step
                        chords.append({'token': token, 'start': time_step, 'end': -1})

        # Finalize any open notes/chords
        for pitch, start_time in active_notes.items():
            notes.append({'pitch': pitch, 'start': start_time, 'end': len(seq)})
        if chords:
            chords[-1]['end'] = len(seq)
            
        parsed_data.append({'notes': notes, 'chords': chords})
    return parsed_data

def is_pitch_in_chord(pitch: int, chord_token: int, tokenizer_info: Dict) -> bool:
    """
    Checks if a MIDI pitch is part of a given chord.
    This is a placeholder and needs a proper implementation based on tokenizer_info.
    For now, it returns True half of the time for demonstration.
    """
    # This requires a map from chord token to a set of MIDI pitch classes.
    # e.g., C Major chord token -> {0, 4, 7}
    # This info should be in tokenizer_info['id_to_chord_token']
    return (pitch + chord_token) % 2 == 0


def calculate_harmony_metrics(sequences: List[np.ndarray], tokenizer_info: Dict) -> Dict:
    """
    Calculates harmony metrics.
    - Melody Note in Chord Ratio
    """
    parsed_data = parse_sequences(sequences, tokenizer_info)
    in_harmony_count = 0
    total_notes = 0

    for data in parsed_data:
        total_notes += len(data['notes'])
        for note in data['notes']:
            # Find active chord at note onset
            active_chord = None
            for chord in data['chords']:
                if chord['start'] <= note['start'] < chord['end']:
                    active_chord = chord
                    break
            
            if active_chord:
                if is_pitch_in_chord(note['pitch'], active_chord['token'], tokenizer_info):
                    in_harmony_count += 1
    
    ratio = (in_harmony_count / total_notes) if total_notes > 0 else 0
    return {"melody_note_in_chord_ratio": ratio * 100}


def calculate_synchronization_metrics(sequences: List[np.ndarray], tokenizer_info: Dict) -> Dict:
    """
    Calculates synchronization metrics.
    - ∆ Chord-Note Onset Interval
    """
    parsed_data = parse_sequences(sequences, tokenizer_info)
    onset_intervals = []

    for data in parsed_data:
        if not data['notes'] or not data['chords']:
            continue
        
        note_onsets = np.array(sorted([n['start'] for n in data['notes']]))
        chord_onsets = np.array(sorted([c['start'] for c in data['chords']]))

        for n_onset in note_onsets:
            # Find the closest chord onset
            if len(chord_onsets) > 0:
                distances = np.abs(chord_onsets - n_onset)
                onset_intervals.append(np.min(distances))

    avg_interval = np.mean(onset_intervals) if onset_intervals else 0
    # The paper multiplies by 10^-3, which seems odd for frame differences.
    # I will just return the average frame difference.
    return {"delta_chord_note_onset_interval": avg_interval}


def calculate_rhythm_diversity_metrics(sequences: List[np.ndarray], tokenizer_info: Dict) -> Dict:
    """
    Calculates rhythm diversity metrics.
    - Chord Length Entropy
    """
    parsed_data = parse_sequences(sequences, tokenizer_info)
    all_chord_lengths = []
    for data in parsed_data:
        for chord in data['chords']:
            length = chord['end'] - chord['start']
            if length > 0:
                all_chord_lengths.append(length)

    if not all_chord_lengths:
        return {"chord_length_entropy": 0.0}

    counts = Counter(all_chord_lengths)
    total_chords = len(all_chord_lengths)
    probabilities = [count / total_chords for count in counts.values()]
    
    return {"chord_length_entropy": entropy(probabilities, base=2)} 