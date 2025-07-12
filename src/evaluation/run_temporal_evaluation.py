#!/usr/bin/env python3
"""
Main script for running temporal evaluation on both online and offline models.

This script loads trained models, runs temporal evaluation across different scenarios,
and logs results to WandB for comprehensive visualization and comparison.

Usage:
    python -m src.evaluation.run_temporal_evaluation \
        --online_artifact "user/project/online_model:version" \
        --offline_artifact "user/project/offline_model:version" \
        --data_dir "data/interim" \
        --output_dir "temporal_results"
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from src.data.dataset import create_dataloader
from src.evaluation.evaluate import load_model_from_wandb as load_online_model
from src.evaluation.evaluate_offline import load_model_from_wandb as load_offline_model
from src.evaluation.temporal_evaluation import (
    generate_online_temporal,
    generate_offline_temporal,
    calculate_test_set_baseline_temporal,
    log_temporal_metrics,
    create_wandb_comparison_dashboard
)


def save_results_json(results: Dict, output_path: Path):
    """
    Save temporal evaluation results to JSON file.
    
    Args:
        results: Dictionary containing all temporal evaluation results
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        for scenario, scenario_data in model_results.items():
            json_results[model_name][scenario] = {}
            for key, value in scenario_data.items():
                if isinstance(value, np.ndarray):
                    json_results[model_name][scenario][key] = value.tolist()
                else:
                    json_results[model_name][scenario][key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved results to {output_path}")


def main(args):
    """Main temporal evaluation function."""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenarios to evaluate
    scenarios = ['primed', 'cold_start', 'perturbed']
    
    print("=== Loading Models ===")
    
    # Load online model
    print(f"Loading online model from: {args.online_artifact}")
    online_model, online_tokenizer_info, online_config = load_online_model(args.online_artifact, device)
    online_model.eval()
    print("Online model loaded successfully")
    
    # Load offline model
    print(f"Loading offline model from: {args.offline_artifact}")
    offline_model, offline_config, offline_tokenizer_info = load_offline_model(args.offline_artifact, device)
    offline_model.eval()
    print("Offline model loaded successfully")
    
    # Use online tokenizer info (should be consistent)
    tokenizer_info = online_tokenizer_info
    
    # Verify tokenizer consistency
    if online_tokenizer_info != offline_tokenizer_info:
        print("WARNING: Online and offline models have different tokenizer info!")
        print("This may cause evaluation issues.")
    
    print("=== Creating Data Loaders ===")
    
    # Create dataloaders
    online_max_seq_length = online_config.get('max_seq_length') or online_config.get('max_sequence_length') or 512
    offline_max_seq_length = offline_config.get('max_seq_length') or offline_config.get('max_sequence_length') or 256
    
    # Online dataloader
    online_dataloader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split=args.split,
        batch_size=1,  # Use batch size 1 for temporal evaluation
        num_workers=0,
        sequence_length=online_max_seq_length,
        mode='online',
        shuffle=False
    )
    
    # Offline dataloader
    offline_dataloader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split=args.split,
        batch_size=1,  # Use batch size 1 for temporal evaluation
        num_workers=0,
        sequence_length=offline_max_seq_length,
        mode='offline',
        shuffle=False
    )
    
    print(f"Created dataloaders for split: '{args.split}'")
    
    print("=== Calculating Test Set Baseline ===")
    
    # Calculate test set baseline
    baseline_results = calculate_test_set_baseline_temporal(
        online_dataloader, tokenizer_info, max_beats=args.max_beats
    )
    
    print("=== Running Temporal Evaluation ===")
    
    # Run temporal evaluation for online model
    print("Evaluating online model...")
    online_results = generate_online_temporal(
        model=online_model,
        dataloader=online_dataloader,
        tokenizer_info=tokenizer_info,
        device=device,
        scenarios=scenarios,
        max_beats=args.max_beats,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Run temporal evaluation for offline model
    print("Evaluating offline model...")
    offline_results = generate_offline_temporal(
        model=offline_model,
        dataloader=offline_dataloader,
        tokenizer_info=tokenizer_info,
        device=device,
        scenarios=scenarios,
        max_beats=args.max_beats,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print("=== Logging Results to WandB ===")
    
    # Log results to WandB
    run_name = f"temporal_eval_{args.run_suffix}" if args.run_suffix else "temporal_eval"
    
    # Log individual model results
    log_temporal_metrics(online_results, "online", run_name, args.wandb_project)
    log_temporal_metrics(offline_results, "offline", run_name, args.wandb_project)
    log_temporal_metrics({"baseline": baseline_results}, "baseline", run_name, args.wandb_project)
    
    # Create comprehensive comparison dashboard
    print("=== Creating WandB Comparison Dashboard ===")
    create_wandb_comparison_dashboard(
        online_results=online_results,
        offline_results=offline_results,
        baseline_results=baseline_results,
        project_name=args.wandb_project,
        run_name=f"{run_name}_comparison"
    )
    
    print("=== Saving Results ===")
    
    # Compile all results
    all_results = {
        'online': online_results,
        'offline': offline_results,
        'baseline': baseline_results,
        'metadata': {
            'online_artifact': args.online_artifact,
            'offline_artifact': args.offline_artifact,
            'scenarios': scenarios,
            'max_beats': args.max_beats,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'split': args.split
        }
    }
    
    # Save results to JSON
    save_results_json(all_results, output_dir / 'temporal_evaluation_results.json')
    
    print("=== Summary ===")
    print(f"Temporal evaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"WandB project: {args.wandb_project}")
    print(f"Evaluated scenarios: {scenarios}")
    print(f"Max beats: {args.max_beats}")
    print(f"Models evaluated: Online, Offline, Baseline")
    
    # Print quick summary stats
    for model_name, model_results in [('Online', online_results), ('Offline', offline_results)]:
        print(f"\n{model_name} Model Summary:")
        for scenario in scenarios:
            if scenario in model_results and 'mean_harmony' in model_results[scenario]:
                mean_scores = model_results[scenario]['mean_harmony']
                if len(mean_scores) > 0:
                    avg_harmony = np.mean(mean_scores)
                    print(f"  {scenario}: {avg_harmony:.3f} average harmony")
    
    # Print baseline summary
    if 'mean_harmony' in baseline_results and len(baseline_results['mean_harmony']) > 0:
        avg_baseline = np.mean(baseline_results['mean_harmony'])
        print(f"\nBaseline Summary:")
        print(f"  ground_truth: {avg_baseline:.3f} average harmony")
    
    print(f"\n✅ Check your WandB project '{args.wandb_project}' for interactive visualizations!")
    print(f"🎯 Look for runs: '{run_name}_online', '{run_name}_offline', '{run_name}_baseline', '{run_name}_comparison'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run temporal evaluation on online and offline models with pure WandB logging.")
    
    # Model artifacts
    parser.add_argument("--online_artifact", type=str, required=True,
                        help="W&B artifact path for online model (e.g., 'user/project/model:version')")
    parser.add_argument("--offline_artifact", type=str, required=True,
                        help="W&B artifact path for offline model (e.g., 'user/project/model:version')")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="data/interim",
                        help="Directory containing the processed data")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to evaluate on (e.g., 'test', 'valid')")
    
    # Evaluation parameters
    parser.add_argument("--max_beats", type=int, default=32,
                        help="Maximum number of beats to evaluate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for generation")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering for generation")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="temporal_results",
                        help="Directory to save results JSON file")
    parser.add_argument("--wandb_project", type=str, default="martydepth-temporal",
                        help="WandB project name for logging")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="Suffix to add to WandB run name")
    
    args = parser.parse_args()
    main(args) 