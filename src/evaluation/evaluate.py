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
from tqdm import tqdm

from src.models.online_transformer import OnlineTransformer
from src.data.dataset import create_dataloader
from src.evaluation.metrics import (
    calculate_harmony_metrics,
    calculate_synchronization_metrics,
    calculate_rhythm_diversity_metrics,
    calculate_emd_metrics,
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
    config = dict(run.config)
    
    # Download artifact files
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = model_artifact.download(root=tmpdir)
        model_path = Path(artifact_dir) / "model.pth"
        tokenizer_path = Path(artifact_dir) / "tokenizer_info.json"
        
        # Load tokenizer info
        with open(tokenizer_path, 'r') as f:
            tokenizer_info = json.load(f)
            
        # The online transformer is for a combined vocabulary
        vocab_size = tokenizer_info['total_vocab_size']

        # Handle different possible key names for max_seq_length
        max_seq_length = config.get('max_sequence_length') or 512

        # Instantiate model
        model = OnlineTransformer(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            max_seq_length=max_seq_length
        ).to(device)
        
        # Load state dict
        # Note: Using weights_only=True is a security best practice
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
    print("Model loaded successfully.")
    return model, tokenizer_info, config

def generate_online(model: OnlineTransformer,
                    dataloader: torch.utils.data.DataLoader,
                    tokenizer_info: Dict,
                    device: torch.device,
                    temperature: float = 1.0,
                    top_k: int = 50,
                    wait_beats: int = 2) -> list:
    """
    Generate sequences by providing the melody and predicting the chords.
    This function is optimized to process entire batches at once.
    Implements wait-and-see behavior where the model can observe melody for a few beats
    before starting to generate chords.
    
    Args:
        model: The online transformer model
        dataloader: Dataloader providing melody sequences
        tokenizer_info: Dictionary containing tokenization information
        device: Device to run generation on
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top logits to sample from
        wait_beats: Number of beats to wait before starting generation (1 beat = 4 frames)
        
    Returns:
        Tuple of generated sequences and ground truth sequences
    """
    model.eval()
    generated_sequences = []
    ground_truth_sequences = []

    # Get token indices from tokenizer info
    melody_vocab_size = tokenizer_info['melody_vocab_size']
    chord_token_start = melody_vocab_size + 1  # After PAD token
    chord_silence_token = chord_token_start  # First chord token is silence
    
    frames_per_beat = 4  # Standard in our dataset
    wait_frames = wait_beats * frames_per_beat

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating online sequences"):
            input_tokens = batch['input_tokens'].to(device)
            melody_tokens = input_tokens[:, 1::2]  # Extract melody tokens
            ground_truth_sequences.extend(batch['target_tokens'].cpu().numpy())

            batch_size = melody_tokens.shape[0]
            seq_length = melody_tokens.shape[1]

            # Start with silence tokens for the waiting period
            generated_so_far = torch.full((batch_size, 1), chord_silence_token, device=device)

            for t in range(seq_length):
                # 1. Append the next ground truth melody token
                current_melody_tokens = melody_tokens[:, t].unsqueeze(1)
                generated_so_far = torch.cat([generated_so_far, current_melody_tokens], dim=1)

                # 2. Predict the next token (which should be a chord)
                logits = model(generated_so_far)[:, -1, :] / temperature
                
                # If we're still in the waiting period, force silence tokens
                if t < wait_frames:
                    logits = torch.full_like(logits, float('-inf'))
                    logits[:, chord_silence_token] = 0.0
                else:
                    # Top-k filtering for normal generation
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        mask = torch.full_like(logits, float('-inf'))
                        mask.scatter_(1, top_k_indices, top_k_logits)
                        logits = mask

                # Sample the next chord token
                probs = torch.softmax(logits, dim=-1)
                next_chord_tokens = torch.multinomial(probs, num_samples=1)

                # 3. Append the generated chord token
                generated_so_far = torch.cat([generated_so_far, next_chord_tokens], dim=1)

            # Collect results for the batch
            # Skip the first token as it's just the initial silence
            for i in range(batch_size):
                full_sequence = generated_so_far[i, 1:].cpu().numpy()
                generated_sequences.append(full_sequence)

    print(f"\nGenerated {len(generated_sequences)} sequences in online mode.")
    return generated_sequences, ground_truth_sequences

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
    model, tokenizer_info, config = load_model_from_wandb(args.artifact_path, device)

    # Create test dataloader (offline mode to get melody and chords separately)
    test_loader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=0, # Easier for local debugging
        sequence_length=config['max_seq_length'],
        mode='online' # Needs to be online to get target tokens
    )
    
    # Generate sequences
    generated_sequences, ground_truth_sequences = generate_online(
        model=model,
        dataloader=test_loader,
        tokenizer_info=tokenizer_info,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Calculate metrics
    harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
    emd_metrics = calculate_emd_metrics(generated_sequences, ground_truth_sequences, tokenizer_info)
    
    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Run: {args.artifact_path}")
    print(f"Harmony: {harmony_metrics}")
    print(f"EMD Metrics: {emd_metrics}")
    print("--------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained OnlineTransformer model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path (entity/project/artifact_name:version)")
    parser.add_argument("data_dir", type=str, help="Path to the data directory with test split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (1 is recommended).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    
    args = parser.parse_args()
    main(args) 