#!/usr/bin/env python3
"""
Script to inspect the contrastive training data samples.
"""
import torch
import argparse
from pathlib import Path
import random

from src.data.dataset import create_dataloader

def inspect_data(data_dir: Path, num_samples: int, split: str):
    """
    Loads, decodes, and prints a few samples from the dataset in contrastive mode.
    """
    print(f"--- Loading data from: {data_dir} (split: {split}) ---")
    
    # Use the existing dataloader creator
    dataloader = create_dataloader(
        data_dir=data_dir,
        split=split,
        batch_size=1,  # We want to inspect one sample at a time
        shuffle=True,  # Shuffle to get random samples
        num_workers=0, # Important for local execution
        mode='contrastive'
    )
    
    # Get the tokenizer mappings
    tokenizer_info = dataloader.dataset.tokenizer_info
    # The keys in the JSON file are strings, but the model uses integer token IDs.
    token_to_note = {int(k): v for k, v in tokenizer_info['token_to_note'].items()}
    token_to_chord = {int(k): v for k, v in tokenizer_info['token_to_chord'].items()}
    
    print(f"\n--- Inspecting {num_samples} random samples ---")
    
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
            
        melody_tokens = batch['melody_tokens'].squeeze(0)
        good_chord_tokens = batch['chord_tokens'].squeeze(0)
        
        # --- Create a "bad" chord progression by shuffling ---
        # This mimics the logic in the training script
        bad_chord_tokens_list = good_chord_tokens.tolist()
        random.shuffle(bad_chord_tokens_list)
        bad_chord_tokens = torch.tensor(bad_chord_tokens_list, dtype=torch.long)
        
        # --- Decode tokens back to human-readable format ---
        def format_note(token_val):
            note_info = token_to_note.get(token_val)
            if note_info is None or note_info[0] == -1:
                return "SILENCE"
            return f"MIDI {note_info[0]}"

        def format_chord(token_val):
            chord_info = token_to_chord.get(token_val)
            if chord_info is None or chord_info[0] == -1:
                return "SILENCE"
            # chord_info is a list from JSON: [root, intervals, inversion, is_onset]
            root, intervals, inversion, _ = chord_info
            return f"R:{root} Inv:{inversion} {intervals}"

        decoded_melody = [format_note(token.item()) for token in melody_tokens]
        decoded_good_chord = [format_chord(token.item()) for token in good_chord_tokens]
        decoded_bad_chord = [format_chord(token.item()) for token in bad_chord_tokens]

        print(f"\n--- Sample #{i+1} ---")
        print(f"Song ID: {batch['song_id'][0]}")
        print("-" * 20)
        
        # Print side-by-side for comparison
        print(f"{'Melody':<15} | {'Good Chord':<25} | {'Bad Chord (Shuffled)':<25}")
        print(f"{'-'*15:<15} | {'-'*25:<25} | {'-'*25:<25}")

        for m, gc, bc in zip(decoded_melody, decoded_good_chord, decoded_bad_chord):
            print(f"{m:<15} | {gc:<25} | {bc:<25}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect contrastive training data.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/interim",
        help="Path to the interim data directory."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="Number of samples to inspect."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split to inspect (e.g., 'train', 'valid')."
    )
    
    args = parser.parse_args()
    
    inspect_data(
        data_dir=Path(args.data_dir), 
        num_samples=args.num_samples, 
        split=args.split
    ) 