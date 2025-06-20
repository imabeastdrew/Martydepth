#!/usr/bin/env python3
"""
A command-line script to run model visualizations.

Example usage:
python src/evaluation/scripts/run_visualizations.py \
    --model_type reward \
    --checkpoint_path checkpoints/reward_model-name.pth \
    --config_path checkpoints/reward_model_config.yaml \
    --data_dir data/interim
"""
import argparse
from pathlib import Path
import torch
import yaml
import json

from src.data.dataset import create_dataloader
from src.models.contrastive_reward_model import ContrastiveRewardModel
from src.models.online_transformer import OnlineTransformer
from src.evaluation.visualize_model import (
    plot_attention_heatmap,
    explain_reward_model_with_shap,
    plot_model_architecture
)

def main():
    """Main function to run visualizations."""
    parser = argparse.ArgumentParser(description="Run model visualizations.")
    parser.add_argument("--model_type", type=str, required=True, choices=['reward', 'transformer'],
                        help="Type of model to visualize ('reward' or 'transformer').")
    parser.add_argument("--checkpoint_path", type=Path, required=True,
                        help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--config_path", type=Path, required=True,
                        help="Path to the training YAML configuration file.")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to the interim data directory.")
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations"),
                        help="Directory to save visualization outputs.")
    
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    print(f"Saving visualizations to: {args.output_dir}")

    # --- Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Assumes tokenizer_info is in the same directory as the data splits
    with open(args.data_dir / "train" / "tokenizer_info.json", 'r') as f:
        tokenizer_info = json.load(f)

    # Dataloader (using validation set for visualization)
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        split="valid",
        batch_size=config.get('batch_size', 16), # Default batch size if not in config
        num_workers=0, # Simpler for offline script
        sequence_length=config['max_seq_length'],
        mode='contrastive' if args.model_type == 'reward' else 'autoregressive',
        shuffle=False
    )

    # --- Model Loading ---
    if args.model_type == 'reward':
        model = ContrastiveRewardModel(
            melody_vocab_size=tokenizer_info['melody_vocab_size'],
            chord_vocab_size=tokenizer_info['chord_vocab_size'],
            **config['model_kwargs'] # Assumes model params are nested
        ).to(device)
    elif args.model_type == 'transformer':
        # NOTE: You will need to implement a `get_attention` method on OnlineTransformer
        # for attention visualization to work correctly.
        model = OnlineTransformer(
            melody_vocab_size=tokenizer_info['melody_vocab_size'],
            chord_vocab_size=tokenizer_info['chord_vocab_size'],
            **config['model_kwargs']
        ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    print(f"Model loaded from {args.checkpoint_path}")

    # --- Run Visualizations ---
    if args.model_type == 'reward':
        print("Running SHAP explanation for the reward model...")
        explain_reward_model_with_shap(
            model, dataloader, device, tokenizer_info,
            save_path=args.output_dir / "reward_model_shap.png"
        )
        print("Plotting reward model architecture...")
        plot_model_architecture(
            model, dataloader, device,
            save_path=args.output_dir / "reward_model_architecture.png"
        )
    elif args.model_type == 'transformer':
        print("Running attention visualization for the transformer...")
        plot_attention_heatmap(
            model, dataloader, device, tokenizer_info,
            save_path=args.output_dir / "transformer_attention.png"
        )
        print("Plotting transformer model architecture...")
        plot_model_architecture(
            model, dataloader, device,
            save_path=args.output_dir / "transformer_architecture.png"
        )

    print("All visualizations completed.")

if __name__ == "__main__":
    main() 