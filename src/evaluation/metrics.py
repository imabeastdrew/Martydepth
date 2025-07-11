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
from src.data.preprocess_frames import MIDITokenizer

def validate_interleaved_sequences(sequences: List[np.ndarray], tokenizer_info: Dict) -> None:
    """
    Validates that sequences are in the correct interleaved format.
    Raises ValueError with helpful message if format is incorrect.
    """
    if not sequences:
        raise ValueError("Empty sequence list provided")
    
    chord_token_start = tokenizer_info.get('chord_token_start', CHORD_TOKEN_START)
    melody_vocab_size = tokenizer_info.get('melody_vocab_size', MELODY_VOCAB_SIZE)
    
    for i, seq in enumerate(sequences[:3]):  # Check first 3 sequences
        # Ensure sequence is a numpy array for consistent processing
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq)
            
        if len(seq) == 0:
            continue
            
        if len(seq) % 2 != 0:
            raise ValueError(f"Sequence {i} has odd length ({len(seq)}). "
                           "Interleaved sequences must have even length [chord_0, melody_0, chord_1, melody_1, ...]")
        
        # Check if all even indices (chord positions) have chord-like tokens
        chord_tokens = np.array(seq[0::2])  # Convert to numpy array for comparison
        melody_tokens = np.array(seq[1::2])  # Convert to numpy array for comparison
        
        # Check if this looks like a chord-only sequence (common mistake)
        if np.all(chord_tokens >= chord_token_start) and np.all(melody_tokens >= chord_token_start):
            raise ValueError(f"Sequence {i} appears to be chord-only (all tokens >= {chord_token_start}). "
                           "Expected interleaved format [chord_0, melody_0, chord_1, melody_1, ...]. "
                           "Use create_interleaved_sequences() to fix this.")
        
        # Check if this looks like a melody-only sequence
        if np.all(chord_tokens < chord_token_start) and np.all(melody_tokens < chord_token_start):
            raise ValueError(f"Sequence {i} appears to be melody-only (all tokens < {chord_token_start}). "
                           "Expected interleaved format [chord_0, melody_0, chord_1, melody_1, ...].")

def parse_sequences(sequences: List[np.ndarray], tokenizer_info: Dict):
    """
    Parses token sequences into structured musical events.
    Handles interleaved sequences where even indices are chord tokens and odd indices are melody tokens.
    
    IMPORTANT: Expects interleaved sequences [chord_0, melody_0, chord_1, melody_1, ...]
    Use create_interleaved_sequences() if you have separate melody and chord arrays.
    """
    # Validate input format
    validate_interleaved_sequences(sequences, tokenizer_info)
    
    parsed_data = []
    melody_tokenizer = MIDITokenizer()  # Create tokenizer instance
    
    for seq in sequences:
        # Ensure sequence is a numpy array for consistent indexing
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq)
            
        notes = []
        chords = []
        active_note = None  # For monophonic melody, only one note can be active
        active_chord = None  # Track current chord

        # Parse interleaved sequence: [chord_0, melody_0, chord_1, melody_1, ...]
        seq_length = len(seq) // 2  # Number of time steps
        
        for time_step in range(seq_length):
            chord_token = seq[time_step * 2]     # Even indices are chord tokens
            melody_token = seq[time_step * 2 + 1]  # Odd indices are melody tokens
            
            # CRITICAL FIX: Skip PAD tokens
            pad_token_id = 178  # PAD_TOKEN from config
            if melody_token == pad_token_id or chord_token == pad_token_id:
                continue  # Skip padded frames
            
            # --- Melody Parsing ---
            if melody_token < CHORD_TOKEN_START:
                # Get MIDI note and onset/hold info using MIDITokenizer
                midi_note, is_onset = melody_tokenizer.decode_token(melody_token)
                is_silence = melody_token == SILENCE_TOKEN

                if midi_note is not None and is_onset:
                    # If a note was already playing, end it.
                    if active_note:
                        notes.append({'pitch': active_note['pitch'], 'start': active_note['start'], 'end': time_step})
                    # Start the new note
                    active_note = {'pitch': midi_note, 'start': time_step}
                elif is_silence:
                    if active_note:
                        notes.append({'pitch': active_note['pitch'], 'start': active_note['start'], 'end': time_step})
                        active_note = None
            
            # --- Chord Parsing (Updated for unified silence token) ---
            silence_token = SILENCE_TOKEN  # Use unified silence token (88)
            
            if chord_token == silence_token:
                # Chord silence - end current chord if any
                if active_chord:
                    chords.append({'token': active_chord['token'], 'start': active_chord['start'], 'end': time_step})
                    active_chord = None
            elif chord_token >= CHORD_TOKEN_START:
                token_str = str(chord_token)
                if token_str in tokenizer_info['token_to_chord']:
                    chord_info = tokenizer_info['token_to_chord'][token_str]
                    
                    if not chord_info['is_hold']:  # Onset token - start new chord
                        if active_chord:
                            chords.append({'token': active_chord['token'], 'start': active_chord['start'], 'end': time_step})
                        active_chord = {'token': chord_token, 'start': time_step}
                    # Hold tokens: do nothing, continue current chord
                    # (The active_chord continues until we hit an onset or silence)

        # Finalize any open notes/chords
        if active_note:
            notes.append({'pitch': active_note['pitch'], 'start': active_note['start'], 'end': seq_length})
        if active_chord:
            chords.append({'token': active_chord['token'], 'start': active_chord['start'], 'end': seq_length})
            
        parsed_data.append({'notes': notes, 'chords': chords})
    return parsed_data

def create_interleaved_sequences(melody_tokens: np.ndarray, chord_tokens: np.ndarray) -> List[np.ndarray]:
    """
    Helper function to create interleaved sequences from separate melody and chord token arrays.
    Used for offline model evaluation where melody and chords are separate.
    
    Args:
        melody_tokens: Array of melody tokens, shape (batch_size, seq_len)
        chord_tokens: Array of chord tokens, shape (batch_size, seq_len)
        
    Returns:
        List of interleaved sequences [chord_0, melody_0, chord_1, melody_1, ...]
    """
    sequences = []
    
    if melody_tokens.ndim == 1:
        melody_tokens = melody_tokens.reshape(1, -1)
    if chord_tokens.ndim == 1:
        chord_tokens = chord_tokens.reshape(1, -1)
    
    for i in range(len(melody_tokens)):
        melody_seq = melody_tokens[i]
        chord_seq = chord_tokens[i]
        
        # CRITICAL FIX: Remove PAD tokens before processing
        # Find first PAD token position (PAD tokens should only be at the end)
        pad_token_id = 178  # PAD_TOKEN from config
        
        # Find effective sequence length (before padding)
        melody_end = len(melody_seq)
        chord_end = len(chord_seq)
        
        for j in range(len(melody_seq)):
            if melody_seq[j] == pad_token_id:
                melody_end = j
                break
                
        for j in range(len(chord_seq)):
            if chord_seq[j] == pad_token_id:
                chord_end = j
                break
        
        # Use the shorter of the two non-padded lengths
        effective_len = min(melody_end, chord_end)
        if effective_len == 0:
            continue  # Skip empty sequences
            
        melody_seq = melody_seq[:effective_len]
        chord_seq = chord_seq[:effective_len]
        
        # Create interleaved sequence: [chord_0, melody_0, chord_1, melody_1, ...]
        interleaved = np.empty(effective_len * 2, dtype=np.int64)
        interleaved[0::2] = chord_seq   # Even indices: chords
        interleaved[1::2] = melody_seq  # Odd indices: melody
        
        sequences.append(interleaved)
    
    return sequences

def fix_online_sequences(generated_sequences: List, melody_sequences: List) -> List[np.ndarray]:
    """
    Fix online model sequences by prepending the first melody note to create proper interleaved format.
    
    Online models often generate sequences missing the first melody note (used as input),
    resulting in odd-length sequences like [chord_0, melody_1, chord_1, melody_2, ...].
    This function fixes them to proper format [chord_0, melody_0, chord_1, melody_1, ...].
    
    Args:
        generated_sequences: List of sequences from online generation (possibly odd length)
        melody_sequences: List of complete melody sequences (used to get first melody note)
        
    Returns:
        List of properly formatted interleaved sequences with even length
    """
    fixed_sequences = []
    
    for i, gen_seq in enumerate(generated_sequences):
        # Convert to list if needed
        if isinstance(gen_seq, np.ndarray):
            gen_seq = gen_seq.tolist()
        
        # Check if sequence has odd length (typical online model issue)
        if len(gen_seq) % 2 == 1:
            # Get the first melody note from the input sequence
            if i < len(melody_sequences):
                first_melody_note = melody_sequences[i][0] if hasattr(melody_sequences[i], '__getitem__') else melody_sequences[i]
                
                # Create proper interleaved format: [chord_0, melody_0, rest_of_sequence...]
                fixed_seq = [gen_seq[0], first_melody_note] + gen_seq[1:]
                fixed_sequences.append(np.array(fixed_seq))
            else:
                # Fallback: truncate to even length
                fixed_sequences.append(np.array(gen_seq[:-1]))
        else:
            # Already even length, keep as is
            fixed_sequences.append(np.array(gen_seq))
    
    return fixed_sequences

# Test set baseline constants (from calculate_test_set_baselines.py output - December 2024)
# NOTE: These values need to be recalculated after fixing the EMD onset filtering and PAD token bugs!
TEST_SET_BASELINES = {
    "harmony_ratio_percent": 65.88,  # TO BE UPDATED
    "total_frames_analyzed": 511988,  # TO BE UPDATED
    "chord_length_entropy": 2.1701,  # TO BE UPDATED
    "onset_interval_emd_internal_variation": 28.8910,  # TO BE UPDATED
    "onset_interval_emd_perfect_sync": 0.0000,  # Should remain 0.0000
    "description": "Baselines calculated from test set ground truth with fixed EMD onset filtering and PAD token handling"
}

def print_baseline_comparison(harmony_metrics: Dict, emd_metrics: Dict):
    """
    Helper function to print model performance compared to test set baselines.
    
    Args:
        harmony_metrics: Output from calculate_harmony_metrics()
        emd_metrics: Output from calculate_emd_metrics()
    """
    baselines = TEST_SET_BASELINES
    
    print("=== MODEL PERFORMANCE vs TEST SET BASELINES ===")
    
    # Harmony ratio
    model_harmony = harmony_metrics.get("melody_note_in_chord_ratio", 0)
    baseline_harmony = baselines["harmony_ratio_percent"]
    harmony_ratio = model_harmony / baseline_harmony if baseline_harmony > 0 else 0
    print(f"Harmony Ratio: {model_harmony:.2f}% vs {baseline_harmony:.2f}% (baseline)")
    print(f"  → Performance: {harmony_ratio:.1%} of baseline {'✅' if harmony_ratio >= 0.8 else '❌'}")
    
    # Chord length entropy
    model_entropy = emd_metrics.get("chord_length_entropy", 0)
    baseline_entropy = baselines["chord_length_entropy"]
    entropy_ratio = model_entropy / baseline_entropy if baseline_entropy > 0 else 0
    print(f"Chord Length Entropy: {model_entropy:.3f} vs {baseline_entropy:.3f} (baseline)")
    print(f"  → Performance: {entropy_ratio:.1%} of baseline {'✅' if entropy_ratio >= 0.8 else '❌'}")
    
    # EMD
    model_emd = emd_metrics.get("onset_interval_emd", float('inf'))
    baseline_emd = baselines["onset_interval_emd_internal_variation"]
    if not np.isnan(model_emd) and baseline_emd > 0:
        emd_performance = baseline_emd / model_emd  # Higher is better (lower EMD)
        print(f"Onset Interval EMD: {model_emd:.2f} vs {baseline_emd:.2f} (baseline)")
        print(f"  → Performance: {'✅' if emd_performance >= 1.0 else '❌'} ({'better' if emd_performance > 1.0 else 'worse'} than baseline)")
    else:
        print(f"Onset Interval EMD: {model_emd} (calculation failed)")
    
    # Overall assessment
    overall_good = (harmony_ratio >= 0.8 and entropy_ratio >= 0.8 and 
                   (not np.isnan(model_emd) and model_emd <= baseline_emd))
    print(f"\nOverall Assessment: {'✅ GOOD' if overall_good else '❌ NEEDS IMPROVEMENT'}")

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

def check_harmony(pitch: int, chord_info: Dict) -> bool:
    """Check if a pitch is part of a chord."""
    # Get chord info
    root = chord_info['root_pitch_class']
    intervals = chord_info['root_position_intervals']
    
    # Convert consecutive intervals to standard intervals from root
    standard_intervals = [0]  # Start with root
    current_sum = 0
    for interval in intervals:
        current_sum += interval
        standard_intervals.append(current_sum)
    
    # Convert intervals to pitch classes
    chord_pitch_classes = {(root + interval) % 12 for interval in standard_intervals}
    
    # Convert melody pitch to pitch class
    melody_pitch_class = pitch % 12
    
    return melody_pitch_class in chord_pitch_classes

def calculate_harmony_metrics(sequences: List[np.ndarray], tokenizer_info: Dict) -> Dict:
    """
    Calculates harmony metrics using the same logic as calculate_test_set_baselines.py
    - Melody Note in Chord Ratio
    
    The metric follows the paper's methodology:
    1. Only considers frames where both melody and chord are active (not silence)
    2. Checks if the melody note is part of the chord's pitch classes
    3. Reports percentage of in-harmony frames
    
    CRITICAL: This function expects interleaved sequences where:
    - Even indices (0, 2, 4, ...) are chord tokens
    - Odd indices (1, 3, 5, ...) are melody tokens
    
    Args:
        sequences: List of interleaved sequences [chord_0, melody_0, chord_1, melody_1, ...]
                  If you have separate melody/chord arrays, use create_interleaved_sequences() first.
        tokenizer_info: Tokenizer information dictionary
        
    Raises:
        ValueError: If sequences are not in correct interleaved format
    """
    # Use the same parsing logic as calculate_test_set_baselines.py
    parsed_data = parse_sequences(sequences, tokenizer_info)
    in_harmony_frames = 0
    total_frames = 0
    
    # Get silence token values from config - now both use unified silence token
    silence_token = SILENCE_TOKEN  # Unified silence token (88) for both melody and chords
    token_to_chord = tokenizer_info.get("token_to_chord", {})

    for data in parsed_data:
        for note in data['notes']:
            # Skip silence notes
            if note['pitch'] == silence_token:
                continue
                
            # For each frame in the note's duration
            for frame in range(note['start'], note['end']):
                # Find active chord at this frame
                active_chord = None
                for chord in data['chords']:
                    if chord['start'] <= frame < chord['end']:
                        active_chord = chord
                        break
                
                # Skip if no chord or silence chord
                if not active_chord or active_chord['token'] == silence_token:
                    continue
                    
                # Get chord info and check harmony - include all chord frames (onset and hold)
                chord_token_str = str(active_chord['token'])
                if chord_token_str not in token_to_chord:
                    continue
                    
                chord_info = token_to_chord[chord_token_str]
                # Count all chord frames (removed hold token filter to match baseline logic)
                    
                total_frames += 1
                if check_harmony(note['pitch'], chord_info):
                    in_harmony_frames += 1
    
    ratio = (in_harmony_frames / total_frames) if total_frames > 0 else 0
    return {
        "melody_note_in_chord_ratio": ratio * 100,  # Convert to percentage
        "total_frames_analyzed": total_frames,
        "in_harmony_frames": in_harmony_frames
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
        
        # Only consider onset times
        note_onsets = [n['start'] for n in data['notes']]
        
        # Get chord onsets, filtering out hold tokens and silence
        chord_onsets = []
        for c in data['chords']:
            token_str = str(c['token'])
            if token_str in tokenizer_info['token_to_chord']:
                chord_info = tokenizer_info['token_to_chord'][token_str]
                if not chord_info['is_hold']:
                    chord_onsets.append(c['start'])
        
        if not note_onsets or not chord_onsets:
            continue
            
        # For each chord onset, find the nearest preceding melody note (fixed bug)
        for c_onset in chord_onsets:
            # Find nearest preceding melody note
            preceding_notes = [n for n in note_onsets if n <= c_onset]
            if preceding_notes:
                nearest_preceding = max(preceding_notes)
                onset_intervals.append(c_onset - nearest_preceding)

    avg_interval = np.mean(onset_intervals) if onset_intervals else 0
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
    Calculates the Earth Mover's Distance (EMD) for onset intervals and entropy for chord lengths
    using the same logic as calculate_test_set_baselines.py.
    
    Following paper's methodology:
    - Onset intervals: bins [0, 1, 2, ..., 16, 17, ∞], EMD × 10^3
    - Chord lengths: bins [0, 1, 2, ..., 32, 33, ∞], calculate entropy in nats
    
    CRITICAL: This function expects interleaved sequences where:
    - Even indices (0, 2, 4, ...) are chord tokens
    - Odd indices (1, 3, 5, ...) are melody tokens
    
    Args:
        generated_sequences: List of interleaved sequences from model [chord_0, melody_0, chord_1, melody_1, ...]
        ground_truth_sequences: List of interleaved sequences from ground truth [chord_0, melody_0, chord_1, melody_1, ...]
        tokenizer_info: Tokenizer information dictionary
        
    Raises:
        ValueError: If sequences are not in correct interleaved format
    """
    # Parse sequences using the same logic as calculate_test_set_baselines.py
    gen_parsed = parse_sequences(generated_sequences, tokenizer_info)
    gt_parsed = parse_sequences(ground_truth_sequences, tokenizer_info)

    # --- Onset Interval EMD ---
    def get_onset_intervals(parsed_data, token_info):
        intervals = []
        for data in parsed_data:
            if not data['notes'] or not data['chords']: 
                continue
            note_onsets = np.array([n['start'] for n in data['notes']])
            # Get chord onsets only (filter out hold tokens - FIXED!)
            chord_onsets = []
            for c in data['chords']:
                # CRITICAL FIX: Only include onset tokens, not hold tokens
                token_str = str(c['token'])
                if token_str in token_info.get('token_to_chord', {}):
                    chord_info = token_info['token_to_chord'][token_str]
                    if not chord_info.get('is_hold', False):  # Only onset tokens
                        chord_onsets.append(c['start'])
                else:
                    # Fallback: assume it's an onset if not in token_to_chord mapping
                    chord_onsets.append(c['start'])
            chord_onsets = np.array(chord_onsets)
            
            for c_onset in chord_onsets:
                # Find nearest preceding melody note
                preceding_notes = [n for n in note_onsets if n <= c_onset]
                if preceding_notes:
                    nearest_preceding = max(preceding_notes)
                    intervals.append(c_onset - nearest_preceding)
        return np.array(intervals)

    # Get intervals for both sets
    gen_intervals = get_onset_intervals(gen_parsed, tokenizer_info)
    gt_intervals = get_onset_intervals(gt_parsed, tokenizer_info)

    # Create histograms with paper's binning
    onset_bins = list(range(18)) + [np.inf]  # [0,1,...,17,inf]
    
    if len(gen_intervals) > 0 and len(gt_intervals) > 0:
        gen_hist, _ = np.histogram(gen_intervals, bins=onset_bins, density=True)
        gt_hist, _ = np.histogram(gt_intervals, bins=onset_bins, density=True)
        # Calculate EMD and multiply by 10^3 as per paper
        onset_emd = wasserstein_distance(gen_hist, gt_hist) * (10**3)
    else:
        onset_emd = float('nan')

    # --- Chord Length Entropy ---
    def get_chord_lengths(parsed_data):
        lengths = []
        for data in parsed_data:
            for chord in data['chords']:
                length = chord['end'] - chord['start']
                if length > 0:  # Skip zero-length chords
                    lengths.append(length)
        return np.array(lengths)

    # Calculate chord length entropy (in nats) for generated sequences
    gen_lengths = get_chord_lengths(gen_parsed)
    if len(gen_lengths) > 0:
        # Create histogram with paper's binning
        length_bins = list(range(34)) + [np.inf]  # [0,1,...,33,inf]
        hist, _ = np.histogram(gen_lengths, bins=length_bins, density=True)
        # Remove zero probabilities before calculating entropy
        hist = hist[hist > 0]
        chord_length_entropy = -np.sum(hist * np.log(hist))  # Natural log for nats
    else:
        chord_length_entropy = 0.0
    
    return {
        "onset_interval_emd": onset_emd,
        "chord_length_entropy": chord_length_entropy
    } 