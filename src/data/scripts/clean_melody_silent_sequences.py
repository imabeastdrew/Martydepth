#!/usr/bin/env python3
"""
Clean melody-silent sequences identified through UMAP analysis.
These are sequences with 100% melody silence but have chord content.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.tokenization_config import PAD_TOKEN, SILENCE_TOKEN, CHORD_SILENCE_TOKEN
from src.data.datastructures import FrameSequence

def is_melody_silent_sequence(sequence: FrameSequence) -> bool:
    """Check if a sequence has 100% melody silence (new outlier type)"""
    melody_tokens = sequence.melody_tokens
    chord_tokens = sequence.chord_tokens
    
    # Remove padding
    non_pad_mask = melody_tokens != PAD_TOKEN
    melody_content = melody_tokens[non_pad_mask]
    chord_content = chord_tokens[non_pad_mask]
    
    if len(melody_content) == 0:
        return True
        
    # Check melody silence ratio
    melody_silence_ratio = np.sum(melody_content == SILENCE_TOKEN) / len(melody_content)
    
    # Flag sequences with 100% melody silence (regardless of chord content)
    return melody_silence_ratio >= 1.0

def clean_split(split_dir: Path) -> dict:
    """Clean a single data split"""
    print(f"\n🧹 Cleaning {split_dir.name} split...")
    
    # Find all sequence files
    sequence_files = list(split_dir.glob("sequence_*.pkl"))
    print(f"Found {len(sequence_files)} sequences")
    
    valid_sequences = []
    melody_silent_sequences = []
    
    for seq_file in sequence_files:
        with open(seq_file, "rb") as f:
            sequence = pickle.load(f)
            
        if is_melody_silent_sequence(sequence):
            melody_silent_sequences.append({
                'file': seq_file.name,
                'song_id': sequence.song_id,
                'start_frame': sequence.start_frame
            })
            # Delete the melody-silent sequence file
            seq_file.unlink()
        else:
            valid_sequences.append(sequence)
    
    # Rename remaining files to be consecutive
    print(f"Renaming {len(valid_sequences)} valid sequences...")
    
    # First, move all valid sequences to temp files
    temp_files = []
    for i, seq in enumerate(valid_sequences):
        temp_file = split_dir / f"temp_sequence_{i:06d}.pkl"
        with open(temp_file, "wb") as f:
            pickle.dump(seq, f)
        temp_files.append(temp_file)
    
    # Delete original sequence files that weren't already deleted
    for seq_file in split_dir.glob("sequence_*.pkl"):
        seq_file.unlink()
    
    # Rename temp files to final names
    for i, temp_file in enumerate(temp_files):
        final_file = split_dir / f"sequence_{i:06d}.pkl"
        temp_file.rename(final_file)
    
    stats = {
        'original_count': len(sequence_files),
        'melody_silent_count': len(melody_silent_sequences),
        'valid_count': len(valid_sequences),
        'melody_silent_sequences': melody_silent_sequences
    }
    
    print(f"  Original: {stats['original_count']} sequences")
    print(f"  Removed: {stats['melody_silent_count']} melody-silent sequences")
    print(f"  Remaining: {stats['valid_count']} valid sequences")
    
    if melody_silent_sequences:
        print("  Melody-silent sequences removed:")
        for seq_info in melody_silent_sequences[:5]:  # Show first 5
            print(f"    {seq_info['song_id']}:{seq_info['start_frame']}")
        if len(melody_silent_sequences) > 5:
            print(f"    ... and {len(melody_silent_sequences) - 5} more")
    
    return stats

def main():
    data_dir = project_root / "data" / "interim"
    splits = ["train", "valid", "test"]
    
    print("🔍 CLEANING MELODY-SILENT SEQUENCES")
    print("=" * 50)
    print("Based on UMAP analysis, removing sequences with 100% melody silence...")
    
    total_stats = defaultdict(int)
    detailed_stats = {}
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"❌ {split} directory not found")
            continue
            
        stats = clean_split(split_dir)
        detailed_stats[split] = stats
        
        total_stats['original_count'] += stats['original_count']
        total_stats['melody_silent_count'] += stats['melody_silent_count']
        total_stats['valid_count'] += stats['valid_count']
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total original sequences: {total_stats['original_count']}")
    print(f"  Total melody-silent sequences removed: {total_stats['melody_silent_count']}")
    print(f"  Total valid sequences remaining: {total_stats['valid_count']}")
    print(f"  Data quality improvement: {(1 - total_stats['melody_silent_count'] / total_stats['original_count']):.1%} clean")
    
    # Save cleaning report
    report = {
        'total_stats': dict(total_stats),
        'split_stats': detailed_stats,
        'cleaning_criteria': 'Removed sequences with 100% melody silence (even if chords exist)'
    }
    
    with open(data_dir / "melody_silent_cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Cleaning complete! Report saved to {data_dir / 'melody_silent_cleaning_report.json'}")
    print("\n🎯 Next steps:")
    print("  1. Re-run your UMAP analysis to see the cleaned data")
    print("  2. The 14 melody-silent sequences should now be gone!")
    print("  3. Consider reprocessing data with updated validation for future")

if __name__ == "__main__":
    main() 