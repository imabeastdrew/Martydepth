#!/usr/bin/env python3
"""
Data Explorer: Show complete structure of Hooktheory dataset entries
"""

import json
from pathlib import Path
from typing import Any, Dict

def get_all_keys(obj: Any, prefix: str = "") -> Dict[str, type]:
    """Recursively get all keys and their types from a nested structure"""
    keys = {}
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_key = f"{prefix}.{key}" if prefix else key
            keys[current_key] = type(value).__name__
            nested_keys = get_all_keys(value, current_key)
            keys.update(nested_keys)
    elif isinstance(obj, list) and obj:
        nested_keys = get_all_keys(obj[0], prefix)
        keys.update(nested_keys)
    
    return keys

def print_melody_timing(entry: Dict):
    """Print onset and offset times for each melody note"""
    if 'annotations' in entry and 'melody' in entry['annotations']:
        melody = entry['annotations']['melody']
        print("\nMelody Timing:")
        print("Note | Onset | Offset | Duration")
        print("-" * 40)
        for i, note in enumerate(melody, 1):
            onset = note.get('onset', 'N/A')
            offset = note.get('offset', 'N/A')
            duration = offset - onset if isinstance(offset, (int, float)) and isinstance(onset, (int, float)) else 'N/A'
            print(f"{i:4d} | {onset:6.2f} | {offset:6.2f} | {duration:6.2f}")

def print_meter_info(entry: Dict):
    """Print meter information including beats, beat units, and time signatures"""
    if 'annotations' in entry and 'meters' in entry['annotations']:
        meters = entry['annotations']['meters']
        num_beats = entry['annotations'].get('num_beats', 'N/A')
        
        print("\nMeter Information:")
        print(f"Total number of beats: {num_beats}")
        print("\nTime Signature Changes:")
        print("Beat | Beats per Bar | Beat Unit")
        print("-" * 40)
        
        for meter in meters:
            beat = meter.get('beat', 'N/A')
            beats_per_bar = meter.get('beats_per_bar', 'N/A')
            beat_unit = meter.get('beat_unit', 'N/A')
            print(f"{beat:4d} | {beats_per_bar:13d} | {beat_unit:9d}")

def print_harmony_info(entry: Dict):
    """Print harmony information including root pitch class, intervals, and inversions"""
    if 'annotations' in entry and 'harmony' in entry['annotations']:
        harmony = entry['annotations']['harmony']
        
        print("\nHarmony Information:")
        print("Chord | Root | Intervals | Inversion")
        print("-" * 50)
        
        for i, chord in enumerate(harmony, 1):
            root = chord.get('root_pitch_class', 'N/A')
            intervals = chord.get('root_position_intervals', 'N/A')
            inversion = chord.get('inversion', 'N/A')
            
            # Format intervals as a string if it's a list
            if isinstance(intervals, list):
                intervals = ' '.join(map(str, intervals))
            
            print(f"{i:5d} | {root:4d} | {intervals:9s} | {inversion:9d}")

def main():
    # Get the project root directory and file path
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "raw" / "Hooktheory copy.json"
    
    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Get the first entry
    first_key = next(iter(data))
    first_entry = data[first_key]
    
    # Get and display all keys
    all_keys = get_all_keys(first_entry)
    for key, type_name in sorted(all_keys.items()):
        print(f"{key}: {type_name}")
    
    # Print melody timing information
    print_melody_timing(first_entry)
    
    # Print meter information
    print_meter_info(first_entry)
    
    # Print harmony information
    print_harmony_info(first_entry)

if __name__ == "__main__":
    main()