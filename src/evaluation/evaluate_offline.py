#!/usr/bin/env python3
"""
Evaluation script for the trained OFFLINE teacher model.
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

from src.models.offline_teacher import OfflineTeacherModel
from src.data.dataset import create_dataloader
from src.evaluation.metrics import (
    calculate_harmony_metrics,
    calculate_emd_metrics,
)
from src.config.tokenization_config import CHORD_SILENCE_TOKEN, CHORD_TOKEN_START, PAD_TOKEN

def log_results_to_wandb(args, metrics, sequences, tokenizer_info):
    """Logs evaluation results and generated samples to W&B."""
    run_name = f"eval_offline_{args.eval_id}_{args.shard_id}"
    run = wandb.init(
        project="martydepth",
        job_type="evaluation",
        name=run_name,
        config=vars(args),
        group=f"eval_offline_{args.eval_id}" # Group runs from the same evaluation
    )
    
    model_artifact = run.use_artifact(args.artifact_path, type='model')
    run.summary.update(metrics)
    
    table = wandb.Table(columns=["id", "generated_sequence"])
    for i, seq in enumerate(sequences[:10]):
        table.add_data(i, str(seq))
    
    run.log({"generated_samples": table})
    run.finish()
    print(f"Logged evaluation results to W&B run: {run.url}")

def load_model_from_wandb(artifact_path: str, device: torch.device):
    """
    Loads an offline model and its configuration from a W&B artifact.
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

    run = model_artifact.logged_by()
    config = dict(run.config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = model_artifact.download(root=tmpdir)
        model_path = Path(artifact_dir) / "model.pth"
        tokenizer_path = Path(artifact_dir) / "tokenizer_info.json"
        
        with open(tokenizer_path, 'r') as f:
            tokenizer_info = json.load(f)
            
        config['melody_vocab_size'] = tokenizer_info['melody_vocab_size']
        config['chord_vocab_size'] = tokenizer_info['chord_vocab_size']

        # Handle different possible key names for max_seq_length
        max_seq_length = config.get('max_seq_length') or config.get('max_sequence_length') or 256
        print(f"Using max_seq_length: {max_seq_length}")

        model = OfflineTeacherModel(
            melody_vocab_size=config['melody_vocab_size'],
            chord_vocab_size=config['chord_vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_length=max_seq_length,
            pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Offline Teacher Model Initialized:\n  Architecture: {config['num_layers']}E + {config['num_layers']}D\n  Embed dimension: {config['embed_dim']}\n  Attention heads: {config['num_heads']}\n  Total parameters: {total_params:,}")
    print("Offline model loaded successfully.")
    return model, config, tokenizer_info

def generate_offline(model: OfflineTeacherModel,
                     dataloader: torch.utils.data.DataLoader,
                     tokenizer_info: Dict,
                     device: torch.device,
                     temperature: float = 1.0,
                     top_k: int = 50,
                     min_chord_frames: int = 2,
                     max_chord_frames: int = 32,
                     change_prob: float = 0.3) -> tuple[list, list]:
    """
    Generate chord sequences for given melodies using the offline model.
    The offline model can see the entire melody sequence before generating each chord.
    
    Args:
        model: The offline teacher model
        dataloader: Dataloader providing melody sequences
        tokenizer_info: Dictionary containing tokenization information
        device: Device to run generation on
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top logits to sample from (0 to disable)
        min_chord_frames: Minimum number of frames for a chord
        max_chord_frames: Maximum number of frames for a chord
        change_prob: Probability to change chord after min duration is met (0.0 to 1.0)
        
    Returns:
        Tuple of (generated sequences, ground truth sequences)
    """
    model.eval()
    generated_sequences = []
    ground_truth_sequences = []
    
    # Get token indices from tokenizer info
    melody_vocab_size = tokenizer_info['melody_vocab_size']
    chord_token_start = melody_vocab_size + 1  # After PAD token
    chord_silence_token = chord_token_start  # First chord token is silence

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating offline sequences"):
            melody_tokens = batch['melody_tokens'].to(device)
            ground_truth_sequences.extend(batch['target_tokens'].cpu().numpy())
            
            batch_size = melody_tokens.shape[0]
            seq_length = melody_tokens.shape[1]
            
            # Start with silence tokens
            generated_so_far = torch.full((batch_size, 1), chord_silence_token, device=device)
            
            # Track current chord and its duration for each sequence in batch
            current_chords = torch.full((batch_size,), chord_silence_token, device=device)
            chord_durations = torch.zeros(batch_size, device=device)

            for t in range(seq_length):
                # Get model predictions
                logits = model(melody_tokens, generated_so_far)[:, -1, :]
                
                # Determine which sequences need new chords
                need_new_chord = (chord_durations >= max_chord_frames) | (
                    (chord_durations >= min_chord_frames) & 
                    (torch.rand(batch_size, device=device) < change_prob)  # Configurable change probability
                )
                
                # For sequences that need new chords, sample from the distribution
                if need_new_chord.any():
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits[need_new_chord], 
                                                               min(top_k, logits.size(-1)))
                        filtered_logits = torch.full_like(logits[need_new_chord], float('-inf'))
                        filtered_logits.scatter_(1, top_k_indices, top_k_logits)
                        logits[need_new_chord] = filtered_logits
                    
                    # Apply temperature scaling after filtering
                    logits[need_new_chord] = logits[need_new_chord] / temperature
                    
                    # Sample new chord tokens
                    probs = torch.softmax(logits[need_new_chord], dim=-1)
                    new_chord_tokens = torch.multinomial(probs, num_samples=1)
                    
                    # Update current chords and reset durations for changed sequences
                    current_chords[need_new_chord] = new_chord_tokens.squeeze(-1)
                    chord_durations[need_new_chord] = 0
                
                # Create next token tensor with current chords
                next_chord_tokens = current_chords.unsqueeze(1)
                
                # Increment durations
                chord_durations += 1
                
                # Append the generated chord token
                generated_so_far = torch.cat([generated_so_far, next_chord_tokens], dim=1)

            # Collect results for the batch
            # Skip the first token as it's just the initial silence
            for i in range(batch_size):
                full_sequence = generated_so_far[i, 1:].cpu().numpy()
                generated_sequences.append(full_sequence)

    print(f"\nGenerated {len(generated_sequences)} sequences in offline mode.")
    return generated_sequences, ground_truth_sequences

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, config, tokenizer_info = load_model_from_wandb(args.artifact_path, device)

    test_loader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=0,
        sequence_length=config.get('max_sequence_length') or config.get('max_seq_length'),
        mode='offline'
    )
    
    generated_sequences, ground_truth_sequences = generate_offline(
        model=model,
        dataloader=test_loader,
        tokenizer_info=tokenizer_info,
        device=device
    )
    
    harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
    emd_metrics = calculate_emd_metrics(generated_sequences, ground_truth_sequences, tokenizer_info)
    
    all_metrics = {**harmony_metrics, **emd_metrics}
    
    print("\n--- Offline Evaluation Results ---")
    print(f"Artifact: {args.artifact_path}")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("---------------------------------")
    
    log_results_to_wandb(args, all_metrics, generated_sequences, tokenizer_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an offline teacher model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path for the model (e.g., entity/project/model:version).")
    parser.add_argument("data_dir", type=str, help="Directory containing the processed test data.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    main(args) 