#!/usr/bin/env python3
"""
Configuration for tokenization constants.
"""

# --- Dataset-Specific Configuration ---
MIN_MIDI_NOTE = -27  # Original range from your code
MAX_MIDI_NOTE = 60   # Original range from your code
UNIQUE_MIDI_NOTES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1  # 88 notes total

# --- Token Types ---
MELODY_TOKEN_START = 0  # Onset tokens start at 0
SILENCE_TOKEN = 88  # Universal silence token for both melody and chords

# --- Melody Token Range (Interleaved) ---
MELODY_ONSET_HOLD_START = 89  # Start of hold tokens (after silence token)
MELODY_VOCAB_SIZE = 178  # 88 onset + 88 hold + 1 silence + 1 pad = 178 tokens

# --- Padding Token ---
PAD_TOKEN = MELODY_VOCAB_SIZE  # Padding token ID (178)

# --- Chord Token Range ---
CHORD_TOKEN_START = PAD_TOKEN + 1  # Start of chord tokens (179)
CHORD_SILENCE_TOKEN = SILENCE_TOKEN  # Use same silence token (88) for both melody and chords

# --- MIDI Conversion Functions ---
def midi_to_token_index(midi_number: int) -> int:
    """Convert MIDI number to token array index"""
    if not (MIN_MIDI_NOTE <= midi_number <= MAX_MIDI_NOTE):
        raise ValueError(f"MIDI {midi_number} outside valid range [{MIN_MIDI_NOTE}, {MAX_MIDI_NOTE}]")
    return midi_number - MIN_MIDI_NOTE

def token_index_to_midi(token_index: int) -> int:
    """Convert token array index to MIDI number"""
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

def is_melody_onset_token(token: int) -> bool:
    """Check if a token is a melody onset token."""
    if token < MELODY_ONSET_HOLD_START or token >= PAD_TOKEN:
        return False
    return True

def is_melody_hold_token(token: int) -> bool:
    """Check if a token is a melody hold token."""
    if token < MELODY_ONSET_HOLD_START or token >= PAD_TOKEN:
        return False
    return True

def is_chord_token(token: int) -> bool:
    """Check if a token is a chord token."""
    return token >= CHORD_TOKEN_START

def get_hold_token(onset_token: int) -> int:
    """Get the hold token corresponding to an onset token."""
    if not is_melody_onset_token(onset_token):
        raise ValueError(f"Token {onset_token} is not a valid melody onset token")
    return onset_token + MELODY_ONSET_HOLD_START 