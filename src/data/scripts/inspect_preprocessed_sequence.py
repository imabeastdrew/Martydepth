#!/usr/bin/env python3
"""
Inspect the output of the frame preprocessor for a single song.
"""

import argparse
import json
from pathlib import Path
import random
from typing import Dict, Any

from src.data.preprocess_frames import FramePreprocessor
from src.config.tokenization_config import PAD_TOKEN, SILENCE_TOKEN, CHORD_SILENCE_TOKEN

def decode_token(token: int, preprocessor: FramePreprocessor) -> str:
    """Decodes a single token into a human-readable string."""
    if token == PAD_TOKEN:
        return "PAD"
    
    # Check if it's a melody token
    try:
        if token < preprocessor.melody_tokenizer.next_token_id:
            if token == SILENCE_TOKEN:
                return "MEL_SILENCE"
            
            midi_num, _, is_onset = preprocessor.melody_tokenizer.decode_note(token)
            return f"MEL_ONSET({midi_num})" if is_onset else f"MEL_HOLD({midi_num})"
    except ValueError:
        pass # Not a melody token

    # Check if it's a chord token
    try:
        if token >= CHORD_SILENCE_TOKEN:
            if token == CHORD_SILENCE_TOKEN:
                return "CHD_SILENCE"

            root, intervals, inversion, is_onset = preprocessor.chord_tokenizer.decode_chord(token)
            # A more readable chord name could be generated here if needed
            chord_name = f"R:{root},I:{intervals},Inv:{inversion}"
            return f"CHD_ONSET({chord_name})" if is_onset else f"CHD_HOLD({chord_name})"
    except ValueError:
        pass # Not a chord token

    return f"UNKNOWN({token})"

def inspect_song(song_id: str, song_data: Dict[str, Any], preprocessor: FramePreprocessor, num_frames: int = 32):
    """Processes a song and prints a comparison of raw vs. processed frames."""
    print(f"--- Inspecting Song ID: {song_id} ---")

    # Process the song
    sequences = preprocessor.process_song(song_id, song_data)
    if not sequences:
        print("Song could not be processed (it might have been filtered out).")
        return

    # Take the first sequence for inspection
    sequence = sequences[0]
    melody_tokens = sequence.melody_tokens
    chord_tokens = sequence.chord_tokens

    print(f"\nShowing the first {num_frames} frames of the first processed sequence...")
    print("-" * 60)
    print(f"{'Frame':<6} | {'Melody Token':<25} | {'Chord Token':<30}")
    print("-" * 60)

    for i in range(num_frames):
        mel_token_str = decode_token(melody_tokens[i], preprocessor)
        chd_token_str = decode_token(chord_tokens[i], preprocessor)
        print(f"{i:<6} | {mel_token_str:<25} | {chd_token_str:<30}")

    print("\n--- Original Annotations (for comparison) ---")
    
    # Display raw annotations for the first few beats
    num_beats_to_show = num_frames / 4
    
    print("\nMelody Notes:")
    melody_notes = song_data.get('annotations', {}).get('melody', [])
    for note in melody_notes:
        if note.get('onset', 0) < num_beats_to_show:
            print(f"  - Onset: {note.get('onset'):<5.2f}, Offset: {note.get('offset'):<5.2f}, Pitch: {(note.get('octave',0)+1)*12 + note.get('pitch_class',0)}")

    print("\nHarmony Chords:")
    harmony_chords = song_data.get('annotations', {}).get('harmony', [])
    for chord in harmony_chords:
        if chord.get('onset', 0) < num_beats_to_show:
            print(f"  - Onset: {chord.get('onset'):<5.2f}, Offset: {chord.get('offset'):<5.2f}, Chord: {chord.get('chord', 'N/A')}")
            
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect preprocessed song sequences.")
    parser.add_argument("--song_id", type=str, default=None, help="Specific song ID to inspect (e.g., '6927'). If not provided, a random song is chosen.")
    parser.add_argument("--num_frames", type=int, default=256, help="Number of frames to display from the start of the sequence.")
    
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent.parent
    raw_data_path = project_root / "data" / "raw" / "Hooktheory copy.json"

    print(f"Loading raw data from: {raw_data_path}")
    try:
        with open(raw_data_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please ensure the Hooktheory dataset is placed in the `data/raw` directory.")
        return

    # Initialize the preprocessor
    preprocessor = FramePreprocessor(sequence_length=256)

    if args.song_id:
        song_id = args.song_id
        if song_id not in raw_data:
            print(f"Error: Song ID '{song_id}' not found in the dataset.")
            return
    else:
        # Pick a random song that has both melody and harmony
        valid_song_ids = [
            sid for sid, sdata in raw_data.items()
            if 'MELODY' in sdata.get('tags', []) and 'HARMONY' in sdata.get('tags', [])
        ]
        if not valid_song_ids:
            print("No valid songs with both MELODY and HARMONY tags found.")
            return
        song_id = random.choice(valid_song_ids)

    song_data = raw_data[song_id]
    inspect_song(song_id, song_data, preprocessor, args.num_frames)

if __name__ == "__main__":
    main() 