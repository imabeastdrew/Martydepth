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
    MELODY_ONSET_HOLD_START,
    MAX_MIDI_NOTE,
    CHORD_TOKEN_START,
)
from src.data.preprocess_frames import get_complete_chord_intervals

def parse_sequences(sequences: List[np.ndarray], tokenizer_info: Dict):
    """
    Parses token sequences into structured musical events.
    For now, this is a simplified parser.
    """
    parsed_data = []
    for seq in sequences:
        notes = []
        chords = []
        active_note = None  # For monophonic melody, only one note can be active

        for time_step, token in enumerate(seq):
            # --- Melody Parsing ---
            if token < CHORD_TOKEN_START:
                # In the new tokenization, onset tokens are even indices after MELODY_ONSET_HOLD_START
                is_onset = token >= MELODY_ONSET_HOLD_START and (token - MELODY_ONSET_HOLD_START) % 2 == 0
                is_silence = token == SILENCE_TOKEN

                if is_onset:
                    # Convert token to pitch index by removing offset and dividing by 2
                    pitch = (token - MELODY_ONSET_HOLD_START) // 2
                    # If a note was already playing, end it.
                    if active_note:
                        notes.append({'pitch': active_note['pitch'], 'start': active_note['start'], 'end': time_step})
                    # Start the new note
                    active_note = {'pitch': pitch, 'start': time_step}
                elif is_silence:
                    if active_note:
                        notes.append({'pitch': active_note['pitch'], 'start': active_note['start'], 'end': time_step})
                        active_note = None
            
            # --- Chord Parsing ---
            else:
                if token >= CHORD_TOKEN_START:
                    if not chords or chords[-1]['token'] != token:
                        if chords:
                            chords[-1]['end'] = time_step
                        chords.append({'token': token, 'start': time_step, 'end': -1})

        # Finalize any open notes/chords
        if active_note:
            notes.append({'pitch': active_note['pitch'], 'start': active_note['start'], 'end': len(seq)})
        if chords:
            chords[-1]['end'] = len(seq)
            
        parsed_data.append({'notes': notes, 'chords': chords})
    return parsed_data

def convert_to_standard_intervals(consecutive_intervals):
    """Convert consecutive intervals to standard intervals from root.
    
    Args:
        consecutive_intervals: List of intervals between consecutive notes
        
    Returns:
        List of intervals from the root note
    """
    if not consecutive_intervals:
        return [0]  # Just the root
        
    standard_intervals = [0]  # Start with root
    current_sum = 0
    for interval in consecutive_intervals:
        current_sum += interval
        standard_intervals.append(current_sum)
    
    return standard_intervals

def check_harmony(melody_note, chord_info):
    """Check if a melody note is in harmony with a chord.
    
    Args:
        melody_note (int): MIDI pitch number of the melody note
        chord_info (dict): Dictionary containing chord information with:
            - root_pitch_class: Root note of the chord (0-11)
            - root_position_intervals: List of consecutive intervals
        
    Returns:
        bool: True if the melody note is in harmony with the chord
    """
    if melody_note == -1:  # Rest
        return True
        
    # Extract chord information
    root = chord_info.get('root_pitch_class', 0)
    consecutive_intervals = chord_info.get('root_position_intervals', [])
    
    # Convert to standard intervals
    standard_intervals = convert_to_standard_intervals(consecutive_intervals)
    
    # Get melody pitch class (0-11)
    melody_pitch_class = melody_note % 12
    
    # Get all chord notes (including root)
    chord_notes = {(root + interval) % 12 for interval in standard_intervals}
    
    return melody_pitch_class in chord_notes

def calculate_harmony_metrics(sequences: List[np.ndarray], tokenizer_info: Dict) -> Dict:
    """
    Calculates harmony metrics.
    - Melody Note in Chord Ratio
    
    The metric follows the paper's methodology:
    1. Only considers frames where both melody and chord are active (not silence)
    2. Checks if the melody note is part of the chord's pitch classes
    3. Reports percentage of in-harmony notes
    """
    parsed_data = parse_sequences(sequences, tokenizer_info)
    in_harmony_count = 0
    total_notes = 0
    
    # Get silence token values from config
    melody_silence_token = SILENCE_TOKEN
    chord_silence_token = tokenizer_info.get("chord_silence_token", CHORD_TOKEN_START - 1)

    for data in parsed_data:
        for note in data['notes']:
            # Find active chord at note onset
            active_chord = None
            for chord in data['chords']:
                if chord['start'] <= note['start'] < chord['end']:
                    active_chord = chord
                    break
            
            # Skip if either is silence
            if not active_chord or note['pitch'] == melody_silence_token or active_chord['token'] == chord_silence_token:
                continue
                
            # Get chord info and check harmony
            chord_info = tokenizer_info['token_to_chord'][str(active_chord['token'])]
            total_notes += 1
            if check_harmony(note['pitch'], chord_info):
                in_harmony_count += 1
    
    ratio = (in_harmony_count / total_notes) if total_notes > 0 else 0
    return {
        "melody_note_in_chord_ratio": ratio * 100,  # Convert to percentage
        "total_notes_analyzed": total_notes,
        "in_harmony_notes": in_harmony_count
    }


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