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
    calculate_synchronization_metrics,
    calculate_rhythm_diversity_metrics,
)
from src.config.tokenization_config import CHORD_SILENCE_TOKEN

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
    config = argparse.Namespace(**run.config)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = model_artifact.download(root=tmpdir)
        model_path = Path(artifact_dir) / "model.pth"
        tokenizer_path = Path(artifact_dir) / "tokenizer_info.json"
        
        with open(tokenizer_path, 'r') as f:
            tokenizer_info = json.load(f)
            
        config.melody_vocab_size = tokenizer_info['melody_vocab_size']
        config.chord_vocab_size = tokenizer_info['chord_vocab_size']

        model = OfflineTeacherModel(
            melody_vocab_size=config.melody_vocab_size,
            chord_vocab_size=config.chord_vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_length=config.max_sequence_length
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
    print("Offline model loaded successfully.")
    return model, config, tokenizer_info

def generate_offline(model: OfflineTeacherModel,
                     dataloader: torch.utils.data.DataLoader,
                     device: torch.device) -> list:
    """
    Generate chord sequences for given melodies using the offline model.
    The generation loop is handled manually here.
    """
    model.eval()
    generated_sequences = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating offline sequences"):
            melody_tokens = batch['melody_tokens'].to(device)
            
            for i in range(melody_tokens.shape[0]):
                single_melody = melody_tokens[i].unsqueeze(0)
                
                # Encode the melody once to get the memory
                melody_embed = model.embeddings.encode_melody(single_melody)
                memory = model.transformer.encoder(melody_embed)
                
                # Start with the CHORD_SILENCE_TOKEN
                decoder_input = torch.full((1, 1), CHORD_SILENCE_TOKEN, dtype=torch.long, device=device)
                
                # Generate one chord for each melody token
                for _ in range(single_melody.size(1)):
                    chord_embed = model.embeddings.encode_chords(decoder_input)
                    causal_mask = model.create_causal_mask(decoder_input.size(1), device)
                    
                    decoder_output = model.transformer.decoder(chord_embed, memory, tgt_mask=causal_mask)
                    logits = model.output_head(decoder_output[:, -1, :])
                    
                    # Greedy decoding
                    next_token = logits.argmax(dim=-1).unsqueeze(1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
                
                # Interleave results for metrics
                # Exclude the starting silence token from the generated chords
                final_chords = decoder_input.squeeze(0)[1:].cpu().numpy()
                final_melody = single_melody.squeeze(0).cpu().numpy()
                
                num_tokens = min(len(final_chords), len(final_melody))
                
                interleaved = np.empty(num_tokens * 2, dtype=np.int64)
                interleaved[0::2] = final_chords[:num_tokens]
                interleaved[1::2] = final_melody[:num_tokens]
                generated_sequences.append(interleaved)
                
    print(f"\nGenerated {len(generated_sequences)} sequences in offline mode.")
    return generated_sequences

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, config, tokenizer_info = load_model_from_wandb(args.artifact_path, device)

    test_loader = create_dataloader(
        data_dir=Path(args.data_dir),
        split="test",
        batch_size=args.batch_size,
        num_workers=0,
        sequence_length=config.max_sequence_length,
        mode='offline'
    )
    
    generated_sequences = generate_offline(
        model=model,
        dataloader=test_loader,
        device=device
    )
    
    harmony_metrics = calculate_harmony_metrics(generated_sequences, tokenizer_info)
    sync_metrics = calculate_synchronization_metrics(generated_sequences, tokenizer_info)
    rhythm_metrics = calculate_rhythm_diversity_metrics(generated_sequences, tokenizer_info)
    
    print("\n--- Offline Evaluation Results ---")
    print(f"Artifact: {args.artifact_path}")
    print(f"Harmony: {harmony_metrics}")
    print(f"Synchronization: {sync_metrics}")
    print(f"Rhythm Diversity: {rhythm_metrics}")
    print("---------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained OfflineTeacher model.")
    parser.add_argument("artifact_path", type=str, help="W&B artifact path (entity/project/artifact_name:version)")
    parser.add_argument("data_dir", type=str, help="Path to the data directory with test split.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (1 is recommended).")
    
    args = parser.parse_args()
    main(args) 