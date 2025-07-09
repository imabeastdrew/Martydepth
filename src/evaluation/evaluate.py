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
import torch.nn.functional as F

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

        # Set default values for missing config parameters
        dropout = config.get('dropout', 0.1)  # Default to 0.1 if not specified
        pad_token_id = tokenizer_info.get('pad_token_id', 0)  # Default to 0 if not specified

        print("\nModel Configuration:")
        print(f"Vocab size: {vocab_size}")
        print(f"Embed dim: {config['embed_dim']}")
        print(f"Num heads: {config['num_heads']}")
        print(f"Num layers: {config['num_layers']}")
        print(f"Max seq length: {max_seq_length}")
        print(f"Dropout: {dropout}")
        print(f"Pad token ID: {pad_token_id}")

        # Instantiate model
        model = OnlineTransformer(
            vocab_size=vocab_size,
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=dropout,
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id
        ).to(device)
        
        # Load state dict
        # Note: Using weights_only=True is a security best practice
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Check model weights for NaN values
        print("\nChecking model weights for NaN values:")
        nan_found = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"Found NaN in {name}")
                nan_found = True
            if torch.isinf(param).any():
                print(f"Found Inf in {name}")
                nan_found = True
        if not nan_found:
            print("No NaN or Inf values found in model weights.")
            
    print("Model loaded successfully.")
    return model, tokenizer_info, config

def generate_online(model, dataloader, tokenizer_info, device, max_length=255, temperature=1.0, top_k=50, wait_beats=None):
    """
    Generate chord sequences using the online model.
    
    Args:
        model: The trained online transformer model
        dataloader: DataLoader containing melody sequences
        tokenizer_info: Dictionary containing tokenizer information
        device: Device to run inference on
        max_length: Maximum sequence length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering for sampling
        wait_beats: Optional parameter for beat timing (currently not implemented)
        
    Returns:
        Tuple of (generated_sequences, ground_truth_sequences)
    """
    model.eval()
    
    # Calculate token configuration from available tokenizer_info keys
    chord_token_start = tokenizer_info['chord_token_start']
    chord_vocab_size = tokenizer_info['chord_vocab_size']
    
    # Calculate the missing keys based on tokenizer structure
    # The chord tokenizer creates onset tokens followed by hold tokens
    # Hold tokens start after all onset tokens
    chord_onset_start = chord_token_start
    chord_onset_end = chord_token_start + (chord_vocab_size // 2) - 1  # Half the vocab is onset tokens
    hold_token_offset = chord_vocab_size // 2  # Offset to convert onset to hold token
    
    print(f"Token ranges: chord_onset=[{chord_onset_start}, {chord_onset_end}], hold_offset={hold_token_offset}")
    print(f"Sampling: temperature={temperature}, top_k={top_k}")
    
    generated_sequences = []
    ground_truth_sequences = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating online sequences")):
            # Extract melody tokens and ground truth - same as debug version
            input_tokens = batch['input_tokens'].to(device)
            melody_sequences = input_tokens[:, 1::2]  # Extract melody tokens from interleaved input
            ground_truth_sequences.extend(batch['target_tokens'].cpu().numpy())

            batch_size, seq_len = melody_sequences.shape

            # Log first batch info like debug version
            if batch_idx == 0:
                print(f"First batch: batch_size={batch_size}, seq_len={seq_len}")
                print(f"Melody range: [{melody_sequences.min()}, {melody_sequences.max()}]")
            
            # Initialize chord sequences - exactly like debug version
            chord_sequences = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            chord_sequences[:, 0] = 0  # START token
            
            # Track chord durations and current chords - exactly like debug version
            chord_durations = torch.ones(batch_size, dtype=torch.long, device=device)
            current_chords = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Autoregressive generation - exactly like debug version
            for t in range(1, seq_len):
                # Prepare input: all previous [chord, melody] pairs - exactly like debug version
                input_length = 2 * t
                input_tokens_batch = torch.zeros(batch_size, input_length, dtype=torch.long, device=device)
                
                # Fill interleaved input: [chord_0, melody_0, chord_1, melody_1, ...] - exactly like debug version
                for i in range(t):
                    input_tokens_batch[:, 2*i] = chord_sequences[:, i]
                    input_tokens_batch[:, 2*i + 1] = melody_sequences[:, i]
                
                # Generate chord predictions - exactly like debug version
                logits = model(input_tokens_batch)
                chord_logits = logits[:, -1, chord_onset_start:chord_onset_end+1]

                # Apply temperature scaling
                chord_logits = chord_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(chord_logits, min(top_k, chord_logits.size(-1)), dim=-1)
                    chord_logits = torch.full_like(chord_logits, float('-inf'))
                    chord_logits.scatter_(-1, top_k_indices, top_k_vals)
                
                # Sample new chord tokens - exactly like debug version
                chord_probs = F.softmax(chord_logits, dim=-1)
                sampled_indices = torch.multinomial(chord_probs, 1).squeeze(-1)
                sampled_chord_tokens = sampled_indices + chord_onset_start
                
                # Chord continuation logic - exactly like debug version
                for seq_idx in range(batch_size):
                    if t == 1:
                        # First timestep: always use new chord - exactly like debug version
                        chord_sequences[seq_idx, t] = sampled_chord_tokens[seq_idx]
                        current_chords[seq_idx] = sampled_chord_tokens[seq_idx]
                        chord_durations[seq_idx] = 1
                    else:
                        # Subsequent timesteps: decide whether to continue or start new chord - exactly like debug version
                        # Simple heuristic: 70% chance to continue if duration < 16 (4 beats), otherwise new chord
                        if chord_durations[seq_idx] < 16 and torch.rand(1).item() < 0.7:
                            # Continue current chord (use hold token) - exactly like debug version
                            chord_sequences[seq_idx, t] = current_chords[seq_idx] + hold_token_offset
                            chord_durations[seq_idx] += 1
                        else:
                            # Start new chord - exactly like debug version
                            chord_sequences[seq_idx, t] = sampled_chord_tokens[seq_idx]
                            current_chords[seq_idx] = sampled_chord_tokens[seq_idx]
                            chord_durations[seq_idx] = 1
            
            # Create interleaved sequences for this batch - exactly like debug version
            for seq_idx in range(batch_size):
                interleaved = []
                for t in range(seq_len):
                    interleaved.append(chord_sequences[seq_idx, t].item())
                    interleaved.append(melody_sequences[seq_idx, t].item())
                generated_sequences.append(interleaved)
            
            # Log progress for first batch like debug version
            if batch_idx == 0:
                print(f"Generated {len(generated_sequences)} sequences from first batch")
                print(f"First sequence length: {len(generated_sequences[0])}")
                print(f"First sequence preview: {generated_sequences[0][:10]}")

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