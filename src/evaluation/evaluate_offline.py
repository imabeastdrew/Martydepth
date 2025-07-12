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

from src.models.offline_teacher_t5 import T5OfflineTeacherModel
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
    
    # Extract run ID from the model artifact to find the corresponding tokenizer artifact
    run_id = run.id
    
    # Parse the artifact path to get entity and project
    parts = artifact_path.split('/')
    if len(parts) >= 3:
        entity = parts[0]
        project = parts[1]
        tokenizer_artifact_path = f"{entity}/{project}/tokenizer_info_{run_id}:latest"
    else:
        raise ValueError(f"Invalid artifact path format: {artifact_path}")
    
    print(f"Looking for tokenizer artifact: {tokenizer_artifact_path}")
    
    # Load tokenizer info from separate artifact
    try:
        tokenizer_artifact = api.artifact(tokenizer_artifact_path, type='tokenizer')
        print(f"Found tokenizer artifact: {tokenizer_artifact.name}")
    except wandb.errors.CommError as e:
        print(f"Error fetching tokenizer artifact: {e}")
        print("Falling back to loading tokenizer info from local data directory...")
        
        # Fallback: Load from local data directory
        
        tokenizer_path = Path("data/interim/test/tokenizer_info.json")
        if tokenizer_path.exists():
            with open(tokenizer_path, 'r') as f:
                tokenizer_info = json.load(f)
            print(f"Loaded tokenizer info from local file: {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Could not find tokenizer artifact and local tokenizer file not found at {tokenizer_path}")
    else:
        # Download tokenizer artifact
        with tempfile.TemporaryDirectory() as tokenizer_tmpdir:
            tokenizer_artifact_dir = tokenizer_artifact.download(root=tokenizer_tmpdir)
            tokenizer_path = Path(tokenizer_artifact_dir) / "tokenizer_info.json"
            
            with open(tokenizer_path, 'r') as f:
                tokenizer_info = json.load(f)
            print("Loaded tokenizer info from WandB artifact")
    
    # Download model artifact files
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = model_artifact.download(root=tmpdir)
        artifact_path = Path(artifact_dir)
        
        # Look for offline teacher model files
        model_files = list(artifact_path.glob("offline_teacher_epoch_*.pth"))
        if not model_files:
            # Fallback to any .pth file
            model_files = list(artifact_path.glob("*.pth"))
        
        if not model_files:
            all_files = list(artifact_path.glob("*"))
            raise FileNotFoundError(f"No model file found in artifact. Available files: {[f.name for f in all_files]}")
        
        # Use the first (or only) model file found
        model_path = model_files[0]
        print(f"Loading offline model from: {model_path.name}")
        
        config['melody_vocab_size'] = tokenizer_info['melody_vocab_size']
        config['chord_vocab_size'] = tokenizer_info['chord_vocab_size']

        # Handle different possible key names for max_seq_length
        max_seq_length = config.get('max_seq_length') or config.get('max_sequence_length') or 256
        print(f"Using max_seq_length: {max_seq_length}")

        model = T5OfflineTeacherModel(
            melody_vocab_size=config['melody_vocab_size'],
            chord_vocab_size=config['chord_vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            max_seq_length=max_seq_length,
            pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN),
            total_vocab_size=tokenizer_info.get('total_vocab_size', 4779)  # Use unified vocabulary
        ).to(device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Full checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from full checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Direct state dict format
            model.load_state_dict(checkpoint)
            print("Loaded model from direct state dict")
        model.eval()
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Offline Teacher Model Initialized:\n  Architecture: {config['num_layers']}E + {config['num_layers']}D\n  Embed dimension: {config['embed_dim']}\n  Attention heads: {config['num_heads']}\n  Total parameters: {total_params:,}")
    print("Offline model loaded successfully.")
    return model, config, tokenizer_info

def generate_offline(model: T5OfflineTeacherModel,
                     dataloader: torch.utils.data.DataLoader,
                     tokenizer_info: Dict,
                     device: torch.device,
                     temperature: float = 1.0,
                     top_k: int = 50) -> tuple[list, list, list]:
    """
    Generate chord sequences for given melodies using the offline model.
    The offline model can see the entire melody sequence before generating each chord.
    
    FIXED: Remove hardcoded timing heuristics and let model decide timing naturally.
    
    Args:
        model: The offline teacher model
        dataloader: Dataloader providing melody sequences
        tokenizer_info: Dictionary containing tokenization information
        device: Device to run generation on
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top logits to sample from (0 to disable)
        
    Returns:
        Tuple of (generated_chord_sequences, ground_truth_chord_sequences, melody_sequences)
    """
    model.eval()
    generated_chord_sequences = []
    ground_truth_chord_sequences = []
    melody_sequences = []
    
    # Get token indices from tokenizer info
    chord_token_start = tokenizer_info['chord_token_start']
    total_vocab_size = tokenizer_info['total_vocab_size']
    pad_token_id = tokenizer_info.get('pad_token_id', PAD_TOKEN)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating offline chord sequences"):
            melody_tokens = batch['melody_tokens'].to(device)
            ground_truth_chord_tokens = batch['chord_target'].cpu().numpy()
            melody_tokens_np = melody_tokens.cpu().numpy()
            
            batch_size = melody_tokens.shape[0]
            seq_length = melody_tokens.shape[1]
            
            # Start with PAD token (T5 standard, matching training)
            generated_so_far = torch.full((batch_size, 1), pad_token_id, device=device)

            for t in range(seq_length):
                # Get model predictions using T5 model interface
                logits = model(
                    melody_tokens=melody_tokens,
                    chord_tokens=generated_so_far
                )[:, -1, :]  # [batch_size, total_vocab_size]
                
                # Filter to chord tokens only
                chord_logits = logits[:, chord_token_start:total_vocab_size]
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(chord_logits, 
                                                           min(top_k, chord_logits.size(-1)))
                    filtered_logits = torch.full_like(chord_logits, float('-inf'))
                    filtered_logits.scatter_(1, top_k_indices, top_k_logits)
                    chord_logits = filtered_logits
                
                # Apply temperature scaling after filtering
                chord_logits = chord_logits / temperature
                
                # Sample chord tokens (model chooses onset vs hold naturally)
                probs = torch.softmax(chord_logits, dim=-1)
                new_chord_tokens = torch.multinomial(probs, num_samples=1)
                
                # Convert back to full vocabulary space
                chord_tokens = new_chord_tokens.squeeze(-1) + chord_token_start
                
                # Append the generated chord token
                generated_so_far = torch.cat([generated_so_far, chord_tokens.unsqueeze(1)], dim=1)

            # Store chord-only sequences (clean separation of concerns)
            for i in range(batch_size):
                generated_chord_seq = generated_so_far[i, 1:].cpu().numpy()  # Skip PAD token
                ground_truth_chord_seq = ground_truth_chord_tokens[i]
                melody_seq = melody_tokens_np[i]
                
                # Ensure all sequences have the same length
                min_len = min(len(generated_chord_seq), len(ground_truth_chord_seq), len(melody_seq))
                generated_chord_seq = generated_chord_seq[:min_len]
                ground_truth_chord_seq = ground_truth_chord_seq[:min_len]
                melody_seq = melody_seq[:min_len]
                
                # Store the separate sequences (not interleaved)
                generated_chord_sequences.append(generated_chord_seq)
                ground_truth_chord_sequences.append(ground_truth_chord_seq)
                melody_sequences.append(melody_seq)

    print(f"\nGenerated {len(generated_chord_sequences)} chord sequences in offline mode.")
    return generated_chord_sequences, ground_truth_chord_sequences, melody_sequences

def evaluate_offline_model_clean(model, dataloader, tokenizer_info, device, **generation_kwargs):
    """
    DEMONSTRATION: How evaluation could work with cleaner separation.
    This approach keeps the model interface clean while supporting evaluation needs.
    """
    # Generate chord-only sequences (clean model interface)
    generated_chords, ground_truth_chords, melodies = generate_offline(
        model, dataloader, tokenizer_info, device, **generation_kwargs
    )
    
    # Convert to interleaved format for metrics (evaluation-specific logic)
    from src.evaluation.metrics import create_interleaved_sequences
    generated_interleaved = create_interleaved_sequences(
        np.array(melodies), np.array(generated_chords)
    )
    ground_truth_interleaved = create_interleaved_sequences(
        np.array(melodies), np.array(ground_truth_chords)
    )
    
    # Calculate metrics
    harmony_metrics = calculate_harmony_metrics(generated_interleaved, tokenizer_info)
    emd_metrics = calculate_emd_metrics(generated_interleaved, ground_truth_interleaved, tokenizer_info)
    
    return {
        'generated_sequences': generated_interleaved,
        'ground_truth_sequences': ground_truth_interleaved,
        'harmony_metrics': harmony_metrics,
        'emd_metrics': emd_metrics,
        'raw_chords': generated_chords,  # Available if needed
        'raw_melodies': melodies
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, config, tokenizer_info = load_model_from_wandb(args.artifact_path, device)

    test_loader, _ = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=0,
        sequence_length=256,
        mode='offline',
        shuffle=False
    )
    
    # Generate chord sequences using the clean adapter pattern
    generated_chords, ground_truth_chords, melody_sequences = generate_offline(
        model=model,
        dataloader=test_loader,
        tokenizer_info=tokenizer_info,
        device=device,
    )
    print(f"Generated {len(generated_chords)} chord sequences.")

    # Convert to interleaved format for metrics calculation (adapter pattern)
    from src.evaluation.metrics import create_interleaved_sequences
    print("Converting to interleaved format for metrics calculation...")
    
    generated_interleaved = create_interleaved_sequences(
        np.array(melody_sequences), np.array(generated_chords)
    )
    ground_truth_interleaved = create_interleaved_sequences(
        np.array(melody_sequences), np.array(ground_truth_chords)
    )
    
    print(f"Created {len(generated_interleaved)} interleaved sequences for evaluation.")

    # Calculate metrics using interleaved sequences
    print("\n--- Calculating Metrics ---")
    harmony_metrics = calculate_harmony_metrics(generated_interleaved, tokenizer_info)
    emd_metrics = calculate_emd_metrics(generated_interleaved, ground_truth_interleaved, tokenizer_info)
    
    all_metrics = {**harmony_metrics, **emd_metrics}
    
    print("\n--- Offline Evaluation Results ---")
    print(f"Artifact: {args.artifact_path}")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("---------------------------------")
    
    log_results_to_wandb(args, all_metrics, generated_interleaved, tokenizer_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an offline teacher model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path for the model (e.g., entity/project/model:version).")
    parser.add_argument("data_dir", type=str, help="Directory containing the processed test data.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    main(args) 