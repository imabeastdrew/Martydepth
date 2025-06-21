#!/usr/bin/env python3
"""
Evaluation metrics for generated sequences
"""

from typing import Dict, List
import numpy as np
from collections import Counter
from scipy.stats import entropy, wasserstein_distance

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


def calculate_harmony_metrics(sequences: List[np.ndarray], tokenizer_info: Dict) -> Dict:
    """
    Calculates harmony metrics.
    - Melody Note in Chord Ratio
    """
    parsed_data = parse_sequences(sequences, tokenizer_info)
    in_harmony_count = 0
    total_notes = 0
    
    # Get silence token values from config
    melody_silence_token = SILENCE_TOKEN
    chord_silence_token = tokenizer_info.get("chord_token_start", CHORD_TOKEN_START)

    for data in parsed_data:
        for note in data['notes']:
            # Find active chord at note onset
            active_chord = None
            for chord in data['chords']:
                if chord['start'] <= note['start'] < chord['end']:
                    active_chord = chord
                    break
            
            # Per paper, exclude frames where either is silence
            if active_chord and note['pitch'] != melody_silence_token and active_chord['token'] != chord_silence_token:
                total_notes += 1
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
    
    return {"chord_length_entropy": entropy(probabilities, base=np.e)}

def calculate_emd_metrics(generated_sequences: List[np.ndarray], 
                          ground_truth_sequences: List[np.ndarray], 
                          tokenizer_info: Dict) -> Dict:
    """
    Calculates the Earth Mover's Distance (EMD) for rhythm and synchronization
    metrics, comparing generated sequences to ground truth sequences.
    """
    # Parse both generated and ground truth sequences
    gen_parsed = parse_sequences(generated_sequences, tokenizer_info)
    gt_parsed = parse_sequences(ground_truth_sequences, tokenizer_info)

    # --- Onset Interval EMD ---
    gen_onset_intervals = []
    for data in gen_parsed:
        if not data['notes'] or not data['chords']: continue
        note_onsets = np.array(sorted([n['start'] for n in data['notes']]))
        chord_onsets = np.array(sorted([c['start'] for c in data['chords']]))
        for n_onset in note_onsets:
            if len(chord_onsets) > 0:
                gen_onset_intervals.append(np.min(np.abs(chord_onsets - n_onset)))

    gt_onset_intervals = []
    for data in gt_parsed:
        if not data['notes'] or not data['chords']: continue
        note_onsets = np.array(sorted([n['start'] for n in data['notes']]))
        chord_onsets = np.array(sorted([c['start'] for c in data['chords']]))
        for n_onset in note_onsets:
            if len(chord_onsets) > 0:
                gt_onset_intervals.append(np.min(np.abs(chord_onsets - n_onset)))

    # Create histograms using specified bins
    onset_bins = np.arange(18) # 0, 1, ..., 17
    gen_onset_hist, _ = np.histogram(gen_onset_intervals, bins=onset_bins, density=True)
    gt_onset_hist, _ = np.histogram(gt_onset_intervals, bins=onset_bins, density=True)

    # Calculate EMD for onset intervals
    onset_emd = wasserstein_distance(gen_onset_hist, gt_onset_hist)

    # --- Chord Length EMD ---
    gen_chord_lengths = [c['end'] - c['start'] for d in gen_parsed for c in d['chords'] if c['end'] > c['start']]
    gt_chord_lengths = [c['end'] - c['start'] for d in gt_parsed for c in d['chords'] if c['end'] > c['start']]

    # Create histograms using specified bins
    length_bins = np.arange(34) # 0, 1, ..., 33
    gen_length_hist, _ = np.histogram(gen_chord_lengths, bins=length_bins, density=True)
    gt_length_hist, _ = np.histogram(gt_chord_lengths, bins=length_bins, density=True)

    # Calculate EMD for chord lengths
    length_emd = wasserstein_distance(gen_length_hist, gt_length_hist)
    
    # Paper multiplies onset EMD by 1000
    return {
        "onset_interval_emd": onset_emd * 1000,
        "chord_length_emd": length_emd
    } 