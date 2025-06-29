# Chord Interval Representation in Raw Data

## Key Finding

We discovered that the raw data represents chord intervals differently than standard music theory notation. This insight was crucial for achieving the correct melody-in-chord ratio (71.36%) that matches the paper's reported 70%.

## Representation Details

### Raw Data Format
In the raw data, chord intervals are stored as consecutive distances between adjacent notes in the chord. For example:
- A minor triad is stored as `[3, 4]` (minor third to perfect fifth)
- A minor seventh chord is stored as `[3, 3, 4]` (minor third to fifth to seventh)

### Standard Music Theory Format
In standard music theory and most chord analysis, intervals are measured from the root note:
- A minor triad is `[0, 3, 7]` (root, minor third, perfect fifth)
- A minor seventh chord is `[0, 3, 7, 10]` (root, minor third, perfect fifth, minor seventh)

## Conversion Process

To correctly analyze melody-in-chord relationships, we convert from the raw format to standard format:

```python
def convert_to_standard_intervals(consecutive_intervals):
    if not consecutive_intervals:
        return [0]  # Just the root
        
    standard_intervals = [0]  # Start with root
    current_sum = 0
    for interval in consecutive_intervals:
        current_sum += interval
        standard_intervals.append(current_sum)
    
    return standard_intervals
```

### Examples
1. Minor Triad:
   - Raw: `[3, 4]`
   - Conversion: `[0, 3, 7]`
   - Notes: Root → +3 semitones → +4 more semitones

2. Minor Seventh:
   - Raw: `[3, 3, 4]`
   - Conversion: `[0, 3, 6, 10]`
   - Notes: Root → +3 → +3 → +4

## Impact on Analysis

### Most Common Patterns
From our analysis of the raw data:
1. `(3, 4)`: 299,767 occurrences (minor triad)
2. `(3, 3, 4)`: 55,644 occurrences (minor seventh)
3. `(3, 4, 4)`: 18,775 occurrences
4. `(2, 5)`: 10,018 occurrences
5. `(3, 4, 7)`: 8,032 occurrences

### Melody-in-Chord Statistics
- Before understanding this representation: 22.67% melody-in-chord ratio
- After correct conversion: 71.36% melody-in-chord ratio
- Paper's reported ratio: 70%

## Implementation Impact

This understanding affects several components of our pipeline:
1. Preprocessing: Converting intervals for chord tokenization
2. Dataset: Ensuring correct chord representation in training data
3. Model evaluation: Properly measuring melody-chord relationships
4. Training: Accurate chord embeddings and relationships

## Recommendations

1. Always convert to standard intervals when:
   - Analyzing melody-chord relationships
   - Computing chord embeddings
   - Evaluating model performance
   
2. Maintain raw format when:
   - Reading from the dataset
   - Storing original chord data
   - Debugging raw data issues 