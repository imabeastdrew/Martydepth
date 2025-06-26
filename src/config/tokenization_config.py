#!/usr/bin/env python3
"""
Configuration for tokenization constants.
"""

# --- Dataset-Specific Configuration ---
# Based on analysis: MIDI range -27 to 60 (88 unique pitches)
MIN_MIDI_NOTE = -27
MAX_MIDI_NOTE = 60
UNIQUE_MIDI_NOTES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1  # 88

# --- Token Types ---
SILENCE_TOKEN = 0

# --- Melody Token Range (Interleaved) ---
# 1 silence + 88 interleaved onset/hold pairs = 177 tokens
MELODY_VOCAB_SIZE = 1 + UNIQUE_MIDI_NOTES * 2  # 177
MELODY_TOKEN_START = 0
MELODY_ONSET_HOLD_START = 1  # Onset/hold pairs start at 1

# --- Padding Token ---
# Place PAD_TOKEN outside all active vocabularies
PAD_TOKEN = MELODY_VOCAB_SIZE  # 177

# --- Chord Token Range ---
CHORD_TOKEN_START = PAD_TOKEN + 1  # 178
CHORD_SILENCE_TOKEN = CHORD_TOKEN_START  # 178
CHORD_ONSET_HOLD_START = CHORD_TOKEN_START + 1  # 179

# --- MIDI Conversion Functions ---
def midi_to_token_index(midi_number: int) -> int:
    """Convert MIDI number to token array index (0-87)"""
    if not (MIN_MIDI_NOTE <= midi_number <= MAX_MIDI_NOTE):
        raise ValueError(f"MIDI {midi_number} outside valid range [{MIN_MIDI_NOTE}, {MAX_MIDI_NOTE}]")
    return midi_number - MIN_MIDI_NOTE

def token_index_to_midi(token_index: int) -> int:
    """Convert token array index (0-87) to MIDI number"""
    if not (0 <= token_index < UNIQUE_MIDI_NOTES):
        raise ValueError(f"Token index {token_index} outside valid range [0, {UNIQUE_MIDI_NOTES-1}]")
    return token_index + MIN_MIDI_NOTE

def midi_to_onset_hold_tokens(midi_number: int) -> tuple[int, int]:
    """Convert MIDI number to interleaved onset/hold token pair"""
    if not (MIN_MIDI_NOTE <= midi_number <= MAX_MIDI_NOTE):
        raise ValueError(f"MIDI {midi_number} outside valid range [{MIN_MIDI_NOTE}, {MAX_MIDI_NOTE}]")
    
    token_index = midi_to_token_index(midi_number)
    onset_token = MELODY_ONSET_HOLD_START + (token_index * 2)
    hold_token = onset_token + 1
    return onset_token, hold_token

def token_to_midi_and_type(token: int) -> tuple[int, bool]:
    """Convert token to (midi_number, is_onset)"""
    if token == SILENCE_TOKEN:
        return -1, False
    if token < MELODY_ONSET_HOLD_START or token >= PAD_TOKEN:
        raise ValueError(f"Token {token} not in melody range")
    
    # Calculate which MIDI note this token represents
    token_offset = token - MELODY_ONSET_HOLD_START
    token_index = token_offset // 2
    is_onset = (token_offset % 2) == 0
    
    midi_number = token_index_to_midi(token_index)
    return midi_number, is_onset 