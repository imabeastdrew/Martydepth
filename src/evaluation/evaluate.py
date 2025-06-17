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
)

def log_results_to_wandb(args, metrics, sequences, tokenizer_info):
    """Logs evaluation results and generated samples to W&B."""
    run_name = f"eval_online_{args.eval_id}_{args.shard_id}"
    run = wandb.init(
        project="martydepth",
        job_type="evaluation",
        name=run_name,
        config=vars(args),
        group=f"eval_online_{args.eval_id}" # Group runs from the same evaluation
    )
    
    # Log the input artifact
    model_artifact = run.use_artifact(args.artifact_path, type='model')
    
    # Log the metrics
    run.summary.update(metrics)
    
    # Create a table of generated examples
    table = wandb.Table(columns=["id", "generated_sequence"])
    for i, seq in enumerate(sequences[:10]): # Log first 10 examples
        table.add_data(i, str(seq))
    
    run.log({"generated_samples": table})
    
    run.finish()
    print(f"Logged evaluation results to W&B run: {run.url}")

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
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
    print("Model loaded successfully.")
    return model, config, tokenizer_info

def generate_online(model: OnlineTransformer,
                    dataloader: torch.utils.data.DataLoader,
                    tokenizer_info: Dict,
                    device: torch.device,
                    temperature: float = 1.0,
                    top_k: int = 50) -> list:
    """
    Generate sequences by providing the melody and predicting the chords.
    """
    model.eval()
    generated_sequences = []

    # Get chord start token index for initial prompt
    chord_start_token_idx = tokenizer_info['melody_vocab_size']

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating online sequences"):
            melody_tokens = batch['melody_tokens'].to(device)
            
            for i in range(melody_tokens.shape[0]): # Process each sequence in the batch
                single_melody = melody_tokens[i]
                
                # Start with a single chord token prompt
                generated_so_far = [chord_start_token_idx]
                
                for melody_token in single_melody:
                    input_ids = torch.tensor(generated_so_far, dtype=torch.long, device=device).unsqueeze(0)
                    
                    # Predict next token (which should be a chord)
                    logits = model(input_ids)[:, -1, :] / temperature
                    
                    # Top-k
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = -float('inf')
                    
                    # Sample chord
                    probs = torch.softmax(logits, dim=-1)
                    next_chord_token = torch.multinomial(probs, num_samples=1).item()
                    
                    # Append the generated chord and the given melody token
                    generated_so_far.append(next_chord_token)
                    generated_so_far.append(melody_token.item())
                    
                generated_sequences.append(np.array(generated_so_far))

    print(f"\nGenerated {len(generated_sequences)} sequences in online mode.")
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

    # Create test dataloader (offline mode to get melody and chords separately)
    test_loader = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=0, # Easier for local debugging
        sequence_length=config.max_sequence_length,
        mode='offline', # Get separate melody and chord tracks
        num_shards=args.num_shards,
        shard_id=args.shard_id
    )
    
    # Generate sequences
    generated_sequences = generate_online(
        model=model,
        dataloader=test_loader,
        tokenizer_info=tokenizer_info,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    # Calculate metrics
    harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
    sync_metrics = calculate_synchronization_metrics(generated_sequences, tokenizer_info)
    rhythm_metrics = calculate_rhythm_diversity_metrics(generated_sequences, tokenizer_info)
    
    all_metrics = {**harmony_metrics, **sync_metrics, **rhythm_metrics}
    
    # Print results
    print("\n--- Online Evaluation Results ---")
    print(f"Artifact: {args.artifact_path}")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("--------------------------")
    
    # Log to W&B
    log_results_to_wandb(args, all_metrics, generated_sequences, tokenizer_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained OnlineTransformer model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path (entity/project/artifact_name:version)")
    parser.add_argument("data_dir", type=str, help="Path to the data directory with test split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (1 is recommended).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards to split the test set into.")
    parser.add_argument("--shard_id", type=int, default=0, help="The ID of the shard to process (0-indexed).")
    parser.add_argument("--eval_id", type=str, default=lambda: wandb.util.generate_id(), help="Unique ID for the evaluation run.")
    
    args = parser.parse_args()
    main(args) 