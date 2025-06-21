#!/usr/bin/env python3
"""
Unified script for evaluating a trained OfflineTeacher model.
"""
import argparse
import torch
from pathlib import Path
import json

from src.data.dataset import create_dataloader
from src.evaluation.evaluate_offline import load_model_from_wandb, generate_offline
from src.evaluation.metrics import (
    calculate_harmony_metrics,
    calculate_emd_metrics,
)

def main(args):
    """Main evaluation function."""
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model, tokenizer, and config from W&B
    print(f"Loading model from artifact: {args.artifact_path}")
    model, config, tokenizer_info = load_model_from_wandb(args.artifact_path, device)
    model.eval()
    print("Model, tokenizer, and config loaded successfully.")

    # Dataloader
    max_seq_length = config.get('max_seq_length') or config.get('max_sequence_length') or 512
    dataloader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=max_seq_length,
        mode='offline',
        shuffle=False
    )
    print(f"Test dataloader created for split: '{args.split}'")

    # Generate sequences
    print("Generating accompaniments...")
    generated_sequences, ground_truth_sequences = generate_offline(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    print(f"Generated {len(generated_sequences)} sequences.")

    # Calculate metrics
    print("\n--- Calculating Metrics ---")
    harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
    emd_metrics = calculate_emd_metrics(generated_sequences, ground_truth_sequences, tokenizer_info)

    all_metrics = {
        **harmony_metrics,
        **emd_metrics,
    }

    print("\n--- Evaluation Results ---")
    print(json.dumps(all_metrics, indent=4))
    print("--------------------------")

    # Optionally, save results to a file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an OfflineTeacher model.")
    parser.add_argument("--artifact_path", type=str, required=True,
                        help="W&B artifact path for the model (e.g., 'user/project/model:version').")
    parser.add_argument("--data_dir", type=str, default="data/interim",
                        help="Directory containing the processed data.")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to evaluate on (e.g., 'test', 'valid').")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for generation.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for the dataloader.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save the results JSON file.")
    
    args = parser.parse_args()
    main(args) 