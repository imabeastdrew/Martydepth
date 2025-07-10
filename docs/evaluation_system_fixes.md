# Evaluation System Fixes and Usage Guide

## Overview

This document describes the architectural improvement made to the evaluation system using the **adapter pattern**. The main issue was that offline model evaluation was conceptually wrong - offline models should only generate chords, but were being forced to return interleaved sequences.

## Clean Architecture Solution: Adapter Pattern

### The Problem
- **Offline models** conceptually should only generate chord sequences: `[chord_0, chord_1, chord_2, ...]`
- **Online models** naturally generate interleaved sequences: `[chord_0, melody_0, chord_1, melody_1, ...]`
- **Metrics functions** require interleaved sequences for melody-in-chord harmony calculations
- Previous "fix" made offline models pretend to generate melody tokens they never actually created

### The Solution: Adapter Pattern
1. **Offline models remain conceptually pure** - `generate_offline()` returns only what the model generates (chords) plus the input melody
2. **Format conversion happens at evaluation time** - use `create_interleaved_sequences()` when needed for metrics
3. **Clean separation of concerns** - model generation vs. evaluation formatting

## Updated Function Signatures

### `generate_offline()` - Clean Architecture
```python
def generate_offline(...) -> tuple[list, list, list]:
    """
    Returns: (generated_chord_sequences, ground_truth_chord_sequences, melody_sequences)
    """
```

**Key Changes:**
- Returns 3 separate arrays instead of pre-interleaved sequences
- Model interface matches what the model actually does
- No longer creates fake melody tokens in the output

### Evaluation Scripts - Adapter Pattern
```python
# Generate chord sequences (clean model interface)
generated_chords, ground_truth_chords, melody_sequences = generate_offline(...)

# Convert to interleaved format for metrics calculation (adapter pattern)
from src.evaluation.metrics import create_interleaved_sequences

generated_interleaved = create_interleaved_sequences(
    np.array(melody_sequences), np.array(generated_chords)
)
ground_truth_interleaved = create_interleaved_sequences(
    np.array(melody_sequences), np.array(ground_truth_chords)
)

# Calculate metrics using interleaved sequences
harmony_metrics = calculate_harmony_metrics(generated_interleaved, tokenizer_info)
emd_metrics = calculate_emd_metrics(generated_interleaved, ground_truth_interleaved, tokenizer_info)
```

## Updated Files

### Core Changes
- **`src/evaluation/evaluate_offline.py`**: Updated `generate_offline()` to return clean chord-only sequences
- **`src/evaluation/run_offline_evaluation.py`**: Uses adapter pattern for metrics
- **`notebooks/evaluate_models.ipynb`**: Updated to use new signature and adapter pattern

### Helper Functions
- **`create_interleaved_sequences()`**: Converts separate melody/chord arrays to interleaved format
- **`validate_interleaved_sequences()`**: Catches format errors with helpful messages

## Benefits of This Approach

### ✅ **Architectural Clarity**
- Models do what they're supposed to do (offline = chord generation)
- No conceptual confusion about what the model generates
- Clean interfaces that match the underlying algorithms

### ✅ **Maintainability**
- Easy to use offline models for other purposes (not just evaluation)
- Clear separation between model generation and metrics formatting
- Consistent with ReaLChords paper methodology

### ✅ **Flexibility**
- Can use offline models for chord-only applications
- Easy to add new metrics that might need different formats
- No breaking changes to the model's core purpose

### ✅ **Compatibility**
- Works with existing evaluation infrastructure
- Maintains backwards compatibility with validation functions
- All existing metrics work unchanged

## Usage Examples

### Offline Model Evaluation
```python
# Clean approach
generated_chords, gt_chords, melodies = generate_offline(model, dataloader, ...)

# Convert for metrics when needed
interleaved_generated = create_interleaved_sequences(melodies, generated_chords)
interleaved_gt = create_interleaved_sequences(melodies, gt_chords)

# Calculate metrics
harmony_metrics = calculate_harmony_metrics(interleaved_generated, tokenizer_info)
```

### Online Model Evaluation (unchanged)
```python
# Online models already return interleaved sequences
generated_sequences, ground_truth_sequences = generate_online(model, dataloader, ...)

# Calculate metrics directly
harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
```

## Error Handling

The system includes robust validation:
- **`validate_interleaved_sequences()`** catches common format errors
- Helpful error messages guide users to the right solution
- Clear documentation about expected formats

## Migration Guide

### For Evaluation Scripts
Replace:
```python
# Old approach (conceptually wrong)
generated_sequences, ground_truth_sequences = generate_offline(...)
metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
```

With:
```python
# New approach (clean architecture)
generated_chords, ground_truth_chords, melody_sequences = generate_offline(...)
from src.evaluation.metrics import create_interleaved_sequences

generated_interleaved = create_interleaved_sequences(
    np.array(melody_sequences), np.array(generated_chords)
)
ground_truth_interleaved = create_interleaved_sequences(
    np.array(melody_sequences), np.array(ground_truth_chords)
)

metrics = calculate_harmony_metrics(generated_interleaved, tokenizer_info)
```

### For Notebooks
Update cells that call `generate_offline()` to handle the new 3-value return signature and use `create_interleaved_sequences()` for metrics calculation.

## Validation

Run the test script to verify everything works:
```bash
python test_evaluation_fix.py
```

This should show:
- ✅ Validation functions catch format errors correctly
- ✅ Adapter pattern creates proper interleaved sequences  
- ✅ Metrics calculation works with both online and offline formats
- ✅ Baseline comparison functions work correctly

## Overall Assessment

The adapter pattern provides a **clean, maintainable, and conceptually correct** solution that:
- Keeps model interfaces honest about what they generate
- Handles format conversion at the appropriate layer (evaluation)
- Maintains compatibility with existing metrics and infrastructure
- Provides clear error messages when format issues occur

This approach follows software engineering best practices and makes the codebase more maintainable for future development. 