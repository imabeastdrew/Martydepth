#!/usr/bin/env python3
"""
Evaluation script for trained models.
"""

import argparse
from pathlib import Path
import torch
import wandb
import tempfile
import json
import numpy as np
from typing import Dict

from src.models.online_transformer import OnlineTransformer
from src.evaluation.metrics import (
    calculate_harmony_metrics,
    calculate_synchronization_metrics,
    calculate_rhythm_diversity_metrics,
)

def load_model_from_wandb(artifact_path: str, device: torch.device):
    """
    Loads a model and its configuration from a W&B artifact.
    
    Args:
        artifact_path: Path to the W&B artifact (e.g., "entity/project/artifact_name:version").
        device: The device to load the model onto.
    
    Returns:
        A tuple of (model, config, tokenizer_info).
    """
    print(f"Loading model from W&B artifact: {artifact_path}")
    api = wandb.Api()
    
    try:
        model_artifact = api.artifact(artifact_path, type='model')
    except wandb.errors.CommError as e:
        print(f"Error fetching artifact: {e}")
        raise
            
    if not model_artifact:
        raise ValueError(f"Artifact {artifact_path} not found or is not of type 'model'")
        
    print(f"Found model artifact: {model_artifact.name}")

    # Get the run that created this artifact to access the config
    run = model_artifact.logged_by()
    
    # Get config
    config = argparse.Namespace(**run.config)
    
    # Download artifact files
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = model_artifact.download(root=tmpdir)
        model_path = Path(artifact_dir) / "model.pth"
        tokenizer_path = Path(artifact_dir) / "tokenizer_info.json"
        
        # Load tokenizer info
        with open(tokenizer_path, 'r') as f:
            tokenizer_info = json.load(f)
            
        # Update config with tokenizer info
        config.vocab_size = tokenizer_info['total_vocab_size']
        config.melody_vocab_size = tokenizer_info['melody_vocab_size']
        config.chord_vocab_size = tokenizer_info['chord_vocab_size']

        # Instantiate model
        model = OnlineTransformer(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_length=config.max_sequence_length
        ).to(device)
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
    print("Model loaded successfully.")
    return model, config, tokenizer_info

def generate(model: OnlineTransformer,
             tokenizer_info: Dict,
             device: torch.device,
             num_sequences: int = 10,
             max_length: int = 512,
             prompt: list = None,
             temperature: float = 1.0,
             top_k: int = 50) -> list:
    """
    Generate sequences from the model using top-k sampling.
    """
    model.eval()
    generated_sequences = []
    
    # Use a default prompt if none is provided
    # A single chord token to start.
    # The first token should be a chord token, let's pick one from the vocab.
    # A common way is to start with a common chord, e.g. C major.
    # For now, let's just pick a valid token index.
    # Chord tokens start after melody tokens. Melody vocab size is the offset.
    chord_start_token_idx = tokenizer_info['melody_vocab_size']
    if prompt is None:
        prompt = [chord_start_token_idx] 

    with torch.no_grad():
        for _ in range(num_sequences):
            input_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
            
            for _ in range(max_length - len(prompt)):
                # Get logits
                logits = model(input_ids)[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

            generated_sequences.append(input_ids.squeeze(0).cpu().numpy())
            
    print(f"Generated {len(generated_sequences)} sequences.")
    return generated_sequences

def main(args):
    """Main evaluation function."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, config, tokenizer_info = load_model_from_wandb(args.artifact_path, device)
    
    # Generate sequences
    generated_sequences = generate(
        model=model,
        tokenizer_info=tokenizer_info,
        device=device,
        num_sequences=args.num_sequences,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Calculate metrics
    harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
    sync_metrics = calculate_synchronization_metrics(generated_sequences, tokenizer_info)
    rhythm_metrics = calculate_rhythm_diversity_metrics(generated_sequences, tokenizer_info)
    
    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Run: {args.artifact_path}")
    print(f"Harmony: {harmony_metrics}")
    print(f"Synchronization: {sync_metrics}")
    print(f"Rhythm Diversity: {rhythm_metrics}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained OnlineTransformer model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path (entity/project/artifact_name:version)")
    parser.add_argument("--num_sequences", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--max_length", type=int, default=512, help="Max length of generated sequences")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    
    args = parser.parse_args()
    main(args) 