#!/usr/bin/env python3
"""
Configuration for tokenization constants.
"""

# --- Token Types ---
SILENCE_TOKEN = 0

# --- Melody Token Range (MIDI-based) ---
# Total melody tokens: 1 (silence) + 128 (onsets) + 128 (holds) = 257
MELODY_VOCAB_SIZE = 257
MELODY_TOKEN_START = 0
MIDI_ONSET_START = 1
MIDI_HOLD_START = 129
MAX_MIDI_NOTE = 127

# --- Padding Tokens ---
PAD_TOKEN = MELODY_TOKEN_START + MELODY_VOCAB_SIZE  # 257
PAD_MELODY = PAD_TOKEN
PAD_CHORD = PAD_TOKEN

# --- Chord Token Range ---
# The chord vocabulary is dynamically built, so we define the starting point.
CHORD_TOKEN_START = PAD_TOKEN + 1  # Starts at 258
CHORD_SILENCE_TOKEN = CHORD_TOKEN_START  # The first token in the chord range is for silence.
CHORD_ONSET_HOLD_START = CHORD_TOKEN_START + 1  # Actual chords start after silence. 