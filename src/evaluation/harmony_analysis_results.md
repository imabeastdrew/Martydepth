# Harmony Analysis Results

## Dataset Overview
- Total songs analyzed: 23,656
- Total notes analyzed: 1,171,934
- Total notes in harmony: 376,943
- Overall harmony ratio: 32.16%

## Methodology Comparison
### Raw MIDI Analysis (Current Method)
- Uses raw MIDI data before tokenization
- Considers precise beat timing and chord durations
- Handles scale degrees, octaves, and accidentals
- Full chord information (type, inversion, alterations)
- Results in ~32% harmony ratio

### Tokenized Data Analysis (Previous Method)
- Uses processed/tokenized data
- Token-level note-chord overlap
- Simplified pitch class comparison
- Basic chord representation
- Results in ~7% harmony ratio

The significant difference in harmony ratios (32% vs 7%) is primarily due to the more detailed and musically accurate analysis possible with raw MIDI data compared to the simplified tokenized representation.

## Statistical Distribution
- Mean harmony ratio: 32.22%
- Median harmony ratio: 31.58%
- Standard deviation: 13.43%
- Range: 0.00% to 100.00%

## Chord Types Found
| Type | Intervals    | Description      |
|------|-------------|------------------|
| 5    | [0, 3, 6]   | Minor triad      |
| 7    | [0, 3, 6, 9]| Minor seventh    |
| 9    | [0, 4, 7, 9]| Major sixth/dominant ninth |
| 11   | [0, 4, 7]   | Major triad      |
| 13   | [0, 4, 7]   | Major triad      |

## Melody Range
- MIDI note range: -34 to 60
- Total range: ~7-8 octaves

## Analysis Details
The analysis shows that approximately one-third of melody notes are part of the underlying chord harmony. This suggests a balanced use of chord tones and non-chord tones (such as passing tones, neighbor tones, etc.) in the melodies.

## Technical Notes
- Analysis performed using `analyze_raw_harmony.py`
- Data sourced from raw MIDI files
- Harmony calculated based on note-chord overlap
- Rest notes excluded from analysis 