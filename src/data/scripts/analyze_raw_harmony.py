#!/usr/bin/env python3
"""
Analyze chord intervals in raw data to understand how they're represented.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_chord_intervals(raw_data):
    """Analyze chord intervals in raw data."""
    interval_patterns = defaultdict(int)
    chord_types = defaultdict(int)
    
    for song_id, song_data in raw_data.items():
        if not isinstance(song_data, dict):
            continue
            
        harmony = song_data.get('annotations', {}).get('harmony', [])
        if not harmony:
            continue
            
        for chord in harmony:
            # Get chord information
            root = chord.get('root_pitch_class', -1)
            intervals = tuple(sorted(chord.get('root_position_intervals', [])))
            chord_type = chord.get('chord_type', 'unknown')
            
            interval_patterns[intervals] += 1
            chord_types[chord_type] += 1
    
    print("\nInterval Pattern Analysis:")
    print("=" * 50)
    print("Top 20 most common interval patterns:")
    for intervals, count in sorted(interval_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{intervals}: {count} occurrences")
    
    print("\nChord Type Analysis:")
    print("=" * 50)
    print("All chord types and their counts:")
    for chord_type, count in sorted(chord_types.items(), key=lambda x: x[1], reverse=True):
        print(f"{chord_type}: {count} occurrences")

def main():
    project_root = Path(__file__).parent.parent.parent.parent
    raw_data_path = project_root / "data" / "raw" / "Hooktheory copy.json"
    
    print(f"Loading raw data from {raw_data_path}")
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
    
    analyze_chord_intervals(raw_data)

if __name__ == "__main__":
    main() 