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
from src.config.tokenization_config import CHORD_SILENCE_TOKEN

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
        max_seq_length = config.get('max_seq_length') or config.get('max_sequence_length') or 512

        model = OfflineTeacherModel(
            melody_vocab_size=config['melody_vocab_size'],
            chord_vocab_size=config['chord_vocab_size'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            max_seq_length=max_seq_length
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Offline Teacher Model Initialized:\n  Architecture: {config['num_layers']}E + {config['num_layers']}D\n  Embed dimension: {config['embed_dim']}\n  Attention heads: {config['num_heads']}\n  Total parameters: {total_params:,}")
    print("Offline model loaded successfully.")
    return model, config, tokenizer_info

def generate_offline(model: OfflineTeacherModel,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device) -> tuple[list, list]:
    """
    Generate chord sequences for given melodies using the offline model.
    Returns both generated and ground_truth interleaved sequences.
    """
    model.eval()
    generated_sequences = []
    ground_truth_sequences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating offline sequences"):
            melody_tokens = batch['melody_tokens'].to(device)
            # The ground truth chords are the targets from the dataloader
            gt_chords = batch['chord_target'].to(device)

            batch_size = melody_tokens.shape[0]
            seq_length = melody_tokens.shape[1]

            # 1. Encode the entire batch of melodies at once
            melody_embed = model.embeddings.encode_melody(melody_tokens)
            memory = model.transformer.encoder(melody_embed)

            # 2. Initialize the decoder input for the whole batch
            decoder_input = torch.full((batch_size, 1), CHORD_SILENCE_TOKEN, dtype=torch.long, device=device)

            # 3. Autoregressively generate chords for the whole batch
            for _ in range(seq_length):
                chord_embed = model.embeddings.encode_chords(decoder_input)
                causal_mask = model.create_causal_mask(decoder_input.size(1), device)
                
                decoder_output = model.transformer.decoder(chord_embed, memory, tgt_mask=causal_mask)
                logits = model.output_head(decoder_output[:, -1, :])
                
                # Greedy decoding for the entire batch
                next_tokens = logits.argmax(dim=-1).unsqueeze(1)
                decoder_input = torch.cat([decoder_input, next_tokens], dim=1)
            
            # 4. Process and collect results for the batch
            # Exclude the starting silence token from the generated chords
            final_chords_batch = decoder_input[:, 1:].cpu().numpy()
            final_melody_batch = melody_tokens.cpu().numpy()
            
            # The ground truth chords also need to be interleaved with the melody
            gt_chords_batch = gt_chords.cpu().numpy()

            for i in range(batch_size):
                final_chords = final_chords_batch[i]
                final_melody = final_melody_batch[i]
                gt_chords_single = gt_chords_batch[i]
                
                num_tokens = min(len(final_chords), len(final_melody))
                
                # Create generated interleaved sequence
                gen_interleaved = np.empty(num_tokens * 2, dtype=np.int64)
                gen_interleaved[0::2] = final_chords[:num_tokens]
                gen_interleaved[1::2] = final_melody[:num_tokens]
                generated_sequences.append(gen_interleaved)

                # Create ground truth interleaved sequence
                gt_interleaved = np.empty(num_tokens * 2, dtype=np.int64)
                gt_interleaved[0::2] = gt_chords_single[:num_tokens]
                gt_interleaved[1::2] = final_melody[:num_tokens]
                ground_truth_sequences.append(gt_interleaved)
                
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