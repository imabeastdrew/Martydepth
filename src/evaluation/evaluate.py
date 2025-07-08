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
    model.eval()
    generated_sequences = []
    ground_truth_sequences = []

    # Get token indices from tokenizer info
    melody_vocab_size = tokenizer_info['melody_vocab_size']
    chord_token_start = melody_vocab_size + 1  # After PAD token
    chord_silence_token = chord_token_start  # First chord token is silence
    chord_vocab_size = tokenizer_info['chord_vocab_size']
    
    frames_per_beat = 4  # Standard in our dataset
    wait_frames = wait_beats * frames_per_beat

    print("\nDebug - Vocabulary Info:")
    print(f"Melody vocab size: {melody_vocab_size}")
    print(f"Chord token start: {chord_token_start}")
    print(f"Chord vocab size: {chord_vocab_size}")
    print(f"Total vocab size: {tokenizer_info['total_vocab_size']}")
    print(f"Wait frames: {wait_frames}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating online sequences")):
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

            # Start with silence tokens for the waiting period
            generated_so_far = torch.full((batch_size, 1), chord_silence_token, device=device)
            
            # Track current chord and its duration for each sequence in batch
            current_chords = torch.full((batch_size,), chord_silence_token, device=device)
            chord_durations = torch.zeros(batch_size, device=device)
            is_hold_token = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # Generate one token at a time
            for t in range(seq_length):
                # 1. Create input sequence for the model
                # Interleave generated chords with melody tokens
                melody_prefix = melody_tokens[:, :t+1]
                input_seq = torch.zeros((batch_size, generated_so_far.size(1) + melody_prefix.size(1)), 
                                     dtype=torch.long, device=device)
                input_seq[:, 0::2] = generated_so_far  # Even indices for chords
                input_seq[:, 1::2] = melody_prefix     # Odd indices for melody

                if batch_idx == 0 and t == 0:
                    print("\nDebug - First Step Input:")
                    print(f"Input sequence shape: {input_seq.shape}")
                    print(f"Input sequence range: [{input_seq.min().item()}, {input_seq.max().item()}]")
                    print("First sequence input:")
                    print(input_seq[0])

                # 2. Get model predictions
                logits = model(input_seq)[:, -1, :]  # Get logits for next token

                if batch_idx == 0 and t == 0:
                    print("\nDebug - First Step Logits:")
                    print(f"Logits shape: {logits.shape}")
                    print(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                    print(f"Number of inf/-inf: {torch.isinf(logits).sum().item()}")
                    print(f"Number of NaN: {torch.isnan(logits).sum().item()}")

                # 3. Determine which sequences need new chords
                # Need new chord if:
                # a) Current chord duration exceeds max_chord_frames
                # b) Current chord duration >= min_chord_frames AND random chance < change_prob
                # c) Current chord is silence and we're past the waiting period
                rand_change = (torch.rand(batch_size, device=device) < change_prob) & (chord_durations >= min_chord_frames)
                need_new_chord = ((chord_durations >= max_chord_frames) | 
                                rand_change |
                                ((current_chords == chord_silence_token) & (t >= wait_frames)))

                if batch_idx == 0 and t == 0:
                    print("\nDebug - First Step Chord Changes:")
                    print(f"Number needing new chord: {need_new_chord.sum().item()}")
                    print(f"Current chord durations: {chord_durations[:5]}")  # First 5 sequences
                    print(f"Is hold token: {is_hold_token[:5]}")  # First 5 sequences

                # 4. Generate next tokens
                if t < wait_frames:
                    # During waiting period, only output silence
                    next_chord_tokens = torch.full((batch_size, 1), chord_silence_token, device=device)
                    chord_durations += 1
                else:
                    # For sequences that need new chords, sample from the distribution
                    if need_new_chord.any():
                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(logits[need_new_chord], 
                                                                   min(top_k, logits.size(-1)))
                            filtered_logits = torch.full_like(logits[need_new_chord], float('-inf'))
                            filtered_logits.scatter_(1, top_k_indices, top_k_logits)
                            logits[need_new_chord] = filtered_logits

                            if batch_idx == 0 and t == wait_frames:
                                print("\nDebug - First Generation Step:")
                                print(f"Top-k logits range: [{top_k_logits.min().item():.2f}, {top_k_logits.max().item():.2f}]")
                                print(f"Top-k indices range: [{top_k_indices.min().item()}, {top_k_indices.max().item()}]")
                                print(f"Filtered logits range: [{filtered_logits.min().item():.2f}, {filtered_logits.max().item():.2f}]")
                        
                        # Apply temperature scaling after filtering
                        logits[need_new_chord] = logits[need_new_chord] / temperature
                        
                        # Sample new chord tokens
                        probs = torch.softmax(logits[need_new_chord], dim=-1)

                        if batch_idx == 0 and t == wait_frames:
                            print("\nDebug - Sampling:")
                            print(f"Temperature scaled logits range: [{logits[need_new_chord].min().item():.2f}, {logits[need_new_chord].max().item():.2f}]")
                            print(f"Probabilities range: [{probs.min().item():.2e}, {probs.max().item():.2e}]")
                            print(f"Probabilities sum: {probs.sum(dim=-1)}")
                            print(f"Number of zero probs: {(probs == 0).sum().item()}")
                            print(f"Number of NaN probs: {torch.isnan(probs).sum().item()}")
                            print(f"Top 5 probabilities for first sequence:")
                            top5_probs, top5_indices = torch.topk(probs[0], 5)
                            for prob, idx in zip(top5_probs, top5_indices):
                                print(f"Token {idx.item()}: {prob.item():.4f}")

                        new_chord_tokens = torch.multinomial(probs, num_samples=1)
                        
                        if batch_idx == 0 and t == wait_frames:
                            print("\nDebug - Generated Tokens:")
                            print(f"New chord tokens range: [{new_chord_tokens.min().item()}, {new_chord_tokens.max().item()}]")
                            print(f"First few tokens: {new_chord_tokens[:5].squeeze(-1)}")
                        
                        # Update current chords and reset durations for changed sequences
                        current_chords[need_new_chord] = new_chord_tokens.squeeze(-1)
                        chord_durations[need_new_chord] = 0
                        is_hold_token[need_new_chord] = False
                    
                    # For continuing chords, use hold tokens
                    hold_mask = ~need_new_chord & ~is_hold_token
                    if hold_mask.any():
                        # Convert onset tokens to hold tokens by adding chord_vocab_size/2
                        hold_offset = chord_vocab_size // 2
                        current_chords[hold_mask] = current_chords[hold_mask] + hold_offset
                        is_hold_token[hold_mask] = True
                    
                    # Create next token tensor with current chords
                    next_chord_tokens = current_chords.unsqueeze(1)
                    
                    # Increment durations
                    chord_durations += 1

                # 5. Append the generated chord token
                generated_so_far = torch.cat([generated_so_far, next_chord_tokens], dim=1)

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