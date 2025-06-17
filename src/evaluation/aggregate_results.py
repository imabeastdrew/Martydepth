#!/usr/bin/env python3
"""
Aggregate results from a parallel evaluation run on W&B.
"""

import argparse
import wandb
import pandas as pd

def aggregate_results(project: str, group_name: str):
    """
    Pulls all runs from a W&B group, aggregates their metrics,
    and prints a summary.
    
    Args:
        project: The W&B project name (e.g., "martydepth").
        group_name: The group name for the evaluation run 
                    (e.g., "eval_offline_j3a5xlg7").
    """
    api = wandb.Api()
    
    print(f"Fetching runs from project '{project}' with group '{group_name}'...")
    
    try:
        runs = api.runs(path=project, filters={"group": group_name})
    except Exception as e:
        print(f"Could not fetch runs. Error: {e}")
        return
        
    if not runs:
        print("No runs found for the specified group. Make sure the eval_id is correct.")
        return
        
    metrics_list = []
    for run in runs:
        # We store the main metrics in the summary
        metrics_list.append(run.summary._json_dict)
        
    # Create a DataFrame for easy analysis
    df = pd.DataFrame(metrics_list)
    
    print(f"\nFound {len(df)} shards.")
    
    # Identify metric columns
    metric_cols = [col for col in df.columns if isinstance(df[col].iloc[0], (int, float))]
    # Exclude some default wandb metrics if they exist
    metric_cols = [m for m in metric_cols if not m.startswith('_')]
    
    if not metric_cols:
        print("No metric columns found in the run summaries.")
        return
        
    print("\n--- Aggregated Results ---")
    
    # Calculate mean and standard deviation for each metric
    summary = df[metric_cols].agg(['mean', 'std'])
    
    print(summary)
    print("--------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate W&B evaluation results.")
    parser.add_argument("eval_group", type=str, help="The evaluation group name (e.g., 'eval_offline_j3a5xlg7')")
    parser.add_argument("--project", type=str, default="martydepth", help="W&B project name")
    
    args = parser.parse_args()
    
    aggregate_results(args.project, args.eval_group) 