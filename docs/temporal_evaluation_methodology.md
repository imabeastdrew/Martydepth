# Temporal Evaluation Methodology

## Overview

This document describes how our temporal evaluation system implements the methodology from research papers on online music accompaniment, specifically matching the evaluation approach described in papers that evaluate harmonic quality over time during sequence generation.

## Research Paper Methodology

The evaluation follows three key scenarios as described in the research literature:

### 1. Primed Generation (Figure 4a)
- **Paper Description**: "We start with the setting of RLDuet, where the models are primed with several beats of ground truth chords before generating their own chords."
- **Our Implementation**: 
  - Start with 8 beats (16 tokens) of ground truth context
  - Models generate chords for remaining beats
  - Demonstrates model's ability to anticipate without having to adapt to previous mistakes
  - **Purpose**: Shows model performance under ideal conditions

### 2. Cold Start (Figure 4b)
- **Paper Description**: "We now proceed to the cold-start setting, which is more natural and more difficult. Here, models predict chords immediately and have to adapt to the resulting melody-chord combinations."
- **Our Implementation**:
  - Start with minimal context (just first chord token)
  - Models must adapt to their own predictions
  - More challenging as models deal with out-of-distribution combinations
  - **Purpose**: Tests model's ability to recover from and adapt to its own mistakes

### 3. Perturbation (Figure 4c)
- **Paper Description**: "Finally, we introduce a deliberate perturbation in the middle of the generation process... We transpose the melody up by a tritone (6 semitones) at beat 17, resulting in both an out-of-distribution melody and almost guaranteeing that the next chord is a poor fit."
- **Our Implementation**:
  - Start with ground truth up to beat 17
  - At beat 17, transpose melody by +6 semitones (tritone)
  - Continue transposing all subsequent melody notes
  - **Purpose**: Tests model's ability to recover from serious disruptions (similar to "push test" in robotics)

## Technical Implementation

### Melody Transposition
```python
def transpose_melody_token(melody_token: int, semitones: int = 6) -> int:
    """Transpose melody token by specified semitones (6 = tritone)"""
    # Decode token to MIDI note
    midi_note, is_onset = melody_tokenizer.decode_token(melody_token)
    # Transpose and re-encode
    transposed_midi = midi_note + semitones
    return melody_tokenizer.encode_token(transposed_midi, is_onset)
```

### Scenario Implementation

#### Online Model (Interleaved Format)
- **Format**: `[chord_0, melody_0, chord_1, melody_1, ...]`
- **Primed**: Use first 16 tokens (8 beats) as context
- **Cold Start**: Use only first chord token
- **Perturbed**: Apply transposition at beat 17 (token index 35 for melody)

#### Offline Model (Direct Format)
- **Format**: `melody[i]` paired with `chord[i]`
- **Primed**: Use first 8 chord tokens as context
- **Cold Start**: Start with PAD token
- **Perturbed**: Transpose melody sequence from beat 17 onwards

### Key Differences from Previous Implementation

1. **Perturbation Type**: 
   - ❌ **Old**: Random chord injection
   - ✅ **New**: Melody transposition (+6 semitones)

2. **Perturbation Timing**:
   - ❌ **Old**: Beat 16
   - ✅ **New**: Beat 17 (matches paper)

3. **Perturbation Target**:
   - ❌ **Old**: Chord tokens
   - ✅ **New**: Melody tokens

4. **Cold Start Context**:
   - ❌ **Old**: First chord + melody
   - ✅ **New**: Minimal context (first chord only)

## Evaluation Metrics

### Harmony Quality Calculation
- **Metric**: Binary harmony score (1.0 = in harmony, 0.0 = not in harmony)
- **Calculation**: Uses existing `check_harmony()` function
- **Aggregation**: Mean harmony score across test sequences at each beat
- **Visualization**: Line plots showing harmony quality over time

### Expected Behavior Patterns

Based on the research paper findings:

1. **Primed Scenario**: 
   - Should maintain high harmony quality throughout
   - Baseline for comparison with other scenarios

2. **Cold Start Scenario**:
   - Initial drop in harmony quality
   - Gradual recovery as model adapts
   - Strong models should approach primed performance

3. **Perturbed Scenario**:
   - Sharp drop in harmony at beat 17 (perturbation point)
   - Quick recovery indicates robust model
   - Failure to recover indicates poor adaptation

## Configuration

The evaluation is configured via `src/evaluation/configs/temporal_evaluation.yaml`:

```yaml
# Scenario-specific parameters
primed_context_beats: 8  # Number of beats for primed context
perturbation_beat: 17    # Beat at which to apply perturbation (matches paper)

# Evaluation parameters
max_beats: 32           # Evaluate up to 32 beats
temperature: 1.0        # Sampling temperature
top_k: 50              # Top-k filtering
```

## Usage

### Jupyter Notebook
```python
# Load configuration
config = yaml.safe_load(open('src/evaluation/configs/temporal_evaluation.yaml'))

# Run evaluation
online_results = generate_online_temporal(
    model=online_model,
    dataloader=online_dataloader,
    tokenizer_info=tokenizer_info,
    device=device,
    scenarios=['primed', 'cold_start', 'perturbed'],
    max_beats=config['max_beats'],
    perturbation_beat=config['perturbation_beat']
)
```

### Command Line
```bash
python src/evaluation/run_temporal_evaluation.py \
    --online_artifact "user/project/online_model:version" \
    --offline_artifact "user/project/offline_model:version" \
    --perturbation_beat 17 \
    --max_beats 32
```

## Visualization

The system creates comprehensive WandB visualizations:

1. **Individual Model Plots**: Line plots for each model/scenario combination
2. **Comparison Dashboard**: Multi-model comparison across scenarios
3. **Data Tables**: Detailed beat-by-beat metrics
4. **Summary Statistics**: Mean performance, recovery trends

## Validation

To validate the implementation matches the paper methodology:

1. **Perturbation Verification**: Check that melody transposition occurs at beat 17
2. **Recovery Patterns**: Look for expected drops and recoveries in harmony quality
3. **Scenario Differences**: Verify primed > cold start > perturbed performance patterns
4. **Baseline Comparison**: Compare against ground truth harmony scores

## Research Paper Compliance

✅ **Primed Generation**: Matches RLDuet setting with ground truth context  
✅ **Cold Start**: Minimal context, model adapts to own predictions  
✅ **Perturbation**: Tritone transposition at beat 17  
✅ **Evaluation Length**: 32 beats total  
✅ **Harmony Metric**: Binary harmony calculation  
✅ **Visualization**: Time-series plots showing recovery patterns  

This implementation provides a faithful reproduction of the research paper methodology for evaluating online music accompaniment models. 