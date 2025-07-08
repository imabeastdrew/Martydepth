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

def generate_online(model: OnlineTransformer,
                    dataloader: torch.utils.data.DataLoader,
                    tokenizer_info: Dict,
                    device: torch.device,
                    temperature: float = 1.0,
                    top_k: int = 50,
                    wait_beats: int = 2,
                    min_chord_frames: int = 2,
                    max_chord_frames: int = 32,
                    change_prob: float = 0.3) -> list:
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
        top_k: Number of top logits to sample from (0 to disable)
        wait_beats: Number of beats to wait before starting generation (1 beat = 4 frames)
        min_chord_frames: Minimum number of frames for a chord
        max_chord_frames: Maximum number of frames for a chord
        change_prob: Probability to change chord after min duration is met (0.0 to 1.0)
        
    Returns:
        Tuple of generated sequences and ground truth sequences
    """
    """Generate sequences using the online transformer model."""
    model.eval()
    generated_sequences = []
    ground_truth_sequences = []
    
    # Get vocabulary info from tokenizer_info
    melody_vocab_size = tokenizer_info['melody_vocab_size']
    chord_token_start = melody_vocab_size + 1  # After PAD token
    chord_vocab_size = tokenizer_info['chord_vocab_size']
    frames_per_beat = 4
    wait_frames = wait_beats * frames_per_beat
    
    print("\nDebug - Vocabulary Info:")
    print(f"Melody vocab size: {melody_vocab_size}")
    print(f"Chord token start: {chord_token_start}")
    print(f"Chord vocab size: {chord_vocab_size}")
    print(f"Total vocab size: {tokenizer_info['total_vocab_size']}")
    print(f"Wait frames: {wait_frames}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating online sequences")):
            # Extract melody tokens and create initial sequence
            input_tokens = batch['input_tokens'].to(device)
            melody_tokens = input_tokens[:, 1::2]  # Extract melody tokens
            ground_truth_sequences.extend(batch['target_tokens'].cpu().numpy())

            batch_size = melody_tokens.shape[0]
            seq_length = melody_tokens.shape[1]

            if batch_idx == 0:
                print("\nDebug - First Batch Info:")
                print(f"Batch size: {batch_size}")
                print(f"Sequence length: {seq_length}")
                print(f"Melody tokens shape: {melody_tokens.shape}")
                print(f"Melody tokens range: [{melody_tokens.min().item()}, {melody_tokens.max().item()}]")
                print("\nFirst sequence melody tokens:")
                print(melody_tokens[0][:20])  # Print first 20 tokens

            # Start with empty sequence - we'll predict the first chord based on first melody token
            generated_so_far = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
            
            # Track current chord and its duration for each sequence in batch
            current_chords = torch.zeros(batch_size, dtype=torch.long, device=device)
            chord_durations = torch.zeros(batch_size, device=device)
            is_hold_token = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Generate one token at a time
            for t in range(seq_length):
                # 1. Create input sequence for the model
                # Interleave generated chords with melody tokens
                melody_prefix = melody_tokens[:, :t+1]
                input_seq = torch.full((batch_size, 2*(t+1)), chord_token_start, dtype=torch.long, device=device)
                if t > 0:
                    input_seq[:, 0:-2:2] = generated_so_far  # Even indices for previous chords
                input_seq[:, 1::2] = melody_prefix     # Odd indices for melody

                if t == 0 and batch_idx == 0:
                    print("\nDebug - Input Sequence:")
                    print(f"Input shape: {input_seq.shape}")
                    print(f"Input range: [{input_seq.min().item()}, {input_seq.max().item()}]")
                    print(f"First sequence: {input_seq[0]}")
                    print(f"Vocab size: {tokenizer_info['total_vocab_size']}")
                    print(f"Input >= vocab_size: {(input_seq >= tokenizer_info['total_vocab_size']).any().item()}")

                # Create attention mask (1 for tokens, 0 for padding)
                attention_mask = torch.zeros_like(input_seq, dtype=torch.bool, device=device)  # False = keep, True = mask
                
                # Validate input tokens are within vocabulary range
                if (input_seq >= tokenizer_info['total_vocab_size']).any():
                    print("WARNING: Input tokens found outside vocabulary range!")
                    input_seq = torch.clamp(input_seq, 0, tokenizer_info['total_vocab_size'] - 1)

                # Get model predictions with attention mask
                raw_logits = model(input_seq, padding_mask=attention_mask)
                logits = raw_logits[:, -1, :]  # Get logits for next token

                # Mask out non-chord tokens
                mask = torch.full_like(logits, float('-inf'))
                mask[:, chord_token_start:chord_token_start + chord_vocab_size] = 0
                logits = logits + mask

                # Apply temperature scaling
                logits = logits / temperature

                # Sample new chord tokens
                probs = torch.softmax(logits, dim=-1)
                new_chord_tokens = torch.multinomial(probs, num_samples=1)

                # Update tracking variables
                current_chords = new_chord_tokens.squeeze(-1)
                chord_durations = torch.zeros_like(chord_durations)
                is_hold_token = torch.zeros_like(is_hold_token)

                # Append the generated chord token
                generated_so_far = torch.cat([generated_so_far, new_chord_tokens], dim=1)

            # Collect results for the batch
            # Skip the first token as it's just the initial silence
            for i in range(batch_size):
                full_sequence = generated_so_far[i, 1:].cpu().numpy()
                generated_sequences.append(full_sequence)

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