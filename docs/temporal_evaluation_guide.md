# Temporal Evaluation System Guide - Pure WandB Edition

This guide explains how to use the temporal evaluation system to measure harmonic quality over time during sequence generation, with comprehensive WandB visualization similar to research papers.

## Overview

The temporal evaluation system measures how well models maintain harmonic quality as they generate sequences beat by beat. It evaluates three key scenarios:

1. **Primed with ground truth** - Model starts with correct context (first 8 beats)
2. **Cold start** - Model starts from scratch with no context
3. **Perturbed midway** - Model starts primed but gets disrupted at beat 16

**✨ Pure WandB Benefits:**
- Interactive plots with zoom, pan, hover capabilities
- Real-time collaboration and sharing
- Automatic experiment versioning
- Easy comparison across multiple runs
- Export capabilities for presentations
- No local plotting dependencies required

## Quick Start

### 1. Using the Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/temporal_evaluation.ipynb
```

**Update the configuration cell with your model artifacts:**
```python
config = {
    'online_artifact': 'your-username/project/online_model:version',
    'offline_artifact': 'your-username/project/offline_model:version',
    'wandb_project': 'your-temporal-project',
    # ... other parameters
}
```

### 2. Using the Command Line Script

```bash
python -m src.evaluation.run_temporal_evaluation \
    --online_artifact "user/project/online_model:version" \
    --offline_artifact "user/project/offline_model:version" \
    --data_dir "data/interim" \
    --wandb_project "your-temporal-project" \
    --output_dir "temporal_results"
```

### 3. Using Configuration File

```bash
# Edit the configuration file
nano src/evaluation/configs/temporal_evaluation.yaml

# Run with config
python -m src.evaluation.run_temporal_evaluation --config src/evaluation/configs/temporal_evaluation.yaml
```

## Configuration

### Model Artifacts

Update these paths with your actual trained models:
- `online_artifact`: Path to your online transformer model in WandB
- `offline_artifact`: Path to your offline transformer model in WandB

### Key Parameters

- `max_beats`: Number of beats to evaluate (32 matches research papers)
- `temperature`: Sampling temperature (0.5 = conservative, 2.0 = creative)
- `top_k`: Top-k filtering (10 = focused, 100 = diverse)
- `split`: Data split to use ("test", "valid", or "train")

### WandB Configuration

```yaml
wandb_project: "martydepth-temporal"
create_comparison_dashboard: true
log_individual_runs: true

wandb_features:
  line_plots: true          # Beat-by-beat harmony plots
  comparison_plots: true    # Multi-model comparisons
  data_tables: true         # Comprehensive data tables
  summary_metrics: true     # Statistical summaries
  bar_charts: true          # Performance comparisons
```

### Scenarios

- `primed`: Start with 8 beats of ground truth, then model generates
- `cold_start`: Model generates from scratch
- `perturbed`: Start primed, inject random chord at beat 16

## WandB Output

The system creates multiple WandB runs with comprehensive visualizations:

### 1. Individual Model Runs
Each model gets its own run with:
- **Beat-by-beat metrics**: Harmony quality over time
- **Scenario line plots**: Interactive plots for each scenario
- **Summary statistics**: Mean, final, and trend metrics
- **Data tables**: Complete results for detailed analysis

### 2. Comparison Dashboard
A unified comparison run featuring:
- **Multi-model line plots**: All models on same chart
- **Performance bar charts**: Mean harmony comparisons
- **Master data table**: All results in one place
- **Statistical summaries**: Comprehensive performance overview

### 3. Interactive Features
- **Zoom and pan**: Detailed inspection of specific time periods
- **Hover tooltips**: Exact values and metadata
- **Filtering**: By model, scenario, or beat range
- **Comparison tools**: Side-by-side run comparisons
- **Export options**: Download data or plots

## Interpreting Results

### Harmony Quality Scale
- **1.0**: Perfect harmony (melody note always in chord)
- **0.5**: Moderate harmony (50% of notes in chord)
- **0.0**: No harmony (melody never matches chord)

### Scenario Insights

**Primed Scenario:**
- Shows how well models maintain quality after good start
- Decline indicates difficulty with long-term consistency

**Cold Start Scenario:**
- Shows model's ability to establish harmonic patterns
- Gradual improvement indicates learning during generation

**Perturbed Scenario:**
- Shows model's robustness to errors
- Quick recovery indicates good error correction

### Model Comparison

**Online Model:**
- Generates chord and melody tokens simultaneously
- May show more consistent timing but less optimal harmony

**Offline Model:**
- Sees full melody before generating chords
- May show better harmony but potential timing issues

## Advanced Usage

### Parameter Optimization

Use WandB's parameter tracking to optimize settings:

```python
# Test different temperatures
for temp in [0.7, 1.0, 1.3]:
    config['temperature'] = temp
    results = generate_online_temporal(..., temperature=temp)
    # Results automatically logged to WandB
```

### Custom Scenarios

Modify the `extract_scenario_sequences` function to create custom evaluation scenarios:

```python
def extract_scenario_sequences(input_tokens, target_tokens, scenario):
    if scenario == 'my_custom_scenario':
        # Implement your custom logic here
        pass
```

### WandB Integration Tips

**1. Organize Projects:**
```python
# Use descriptive project names
wandb_project = "temporal-eval-ablation-study"
run_name = "temperature-sweep-0.7"
```

**2. Add Tags:**
```python
wandb.init(
    project=wandb_project,
    name=run_name,
    tags=["temporal", "harmony", "ablation"],
    notes="Testing temperature sensitivity"
)
```

**3. Log Hyperparameters:**
```python
wandb.config.update({
    "temperature": 1.0,
    "top_k": 50,
    "max_beats": 32,
    "model_type": "online"
})
```

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```
❌ Failed to load online model: Artifact not found
```
- Check artifact path is correct
- Ensure you have access to the WandB project
- Verify the model version exists

**2. Data Loading Errors**
```
❌ Failed to create dataloader: No such file or directory
```
- Check `data_dir` path is correct
- Ensure the data split exists (test/valid/train)
- Verify data preprocessing is complete

**3. WandB Authentication**
```
❌ wandb: ERROR Failed to authenticate
```
- Run `wandb login` in terminal
- Check your WandB API key
- Verify internet connection

**4. Memory Issues**
```
CUDA out of memory
```
- Reduce `max_beats` parameter
- Use CPU instead of GPU for evaluation
- Ensure batch_size is set to 1

### Performance Tips

**Speed up evaluation:**
- Reduce `num_test_sequences` in config
- Use smaller `max_beats` for testing
- Run on GPU if available

**Improve accuracy:**
- Increase `num_test_sequences` for more stable results
- Use multiple random seeds for robustness

**WandB Best Practices:**
- Use descriptive run names
- Add tags for easy filtering
- Include notes about experimental conditions
- Share results with team members
- Use WandB's comparison tools for analysis

## Example WandB Workflow

1. **Run Evaluation**: Execute temporal evaluation script
2. **Check Individual Runs**: Review model-specific results
3. **Analyze Comparison**: Use comparison dashboard for insights
4. **Share Results**: Share WandB links with collaborators
5. **Export Data**: Download results for presentations
6. **Iterate**: Adjust parameters based on findings

The pure WandB approach provides publication-ready visualizations with powerful interactive features for deep analysis! 