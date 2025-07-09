#!/usr/bin/env python3
"""
UMAP Visualization for Model Embeddings
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pickle

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

from src.data.dataset import create_dataloader
from src.models.contrastive_reward_model import ContrastiveRewardModel
from src.models.discriminative_reward_model import DiscriminativeRewardModel
from src.models.offline_teacher_t5 import T5OfflineTeacherModel
from src.config.tokenization_config import (
    CHORD_TOKEN_START, CHORD_SILENCE_TOKEN, SILENCE_TOKEN, PAD_TOKEN
)

def extract_contrastive_embeddings(model: ContrastiveRewardModel, 
                                 dataloader: torch.utils.data.DataLoader,
                                 device: torch.device,
                                 max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract melody and chord embeddings from contrastive model"""
    model.eval()
    melody_embeddings = []
    chord_embeddings = []
    song_ids = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting contrastive embeddings")):
            if i * dataloader.batch_size >= max_samples:
                break
                
            melody_tokens = batch['melody_tokens'].to(device)
            chord_tokens = batch['chord_tokens'].to(device)
            
            # Create padding masks
            melody_mask = (melody_tokens == PAD_TOKEN)
            chord_mask = (chord_tokens == PAD_TOKEN)
            
            mel_embed, chord_embed = model(
                melody_tokens, chord_tokens,
                melody_padding_mask=melody_mask,
                chord_padding_mask=chord_mask
            )
            
            melody_embeddings.append(mel_embed.cpu().numpy())
            chord_embeddings.append(chord_embed.cpu().numpy())
            song_ids.extend(batch['song_id'])
    
    melody_embeddings = np.vstack(melody_embeddings)
    chord_embeddings = np.vstack(chord_embeddings)
    
    return melody_embeddings, chord_embeddings, song_ids

def extract_chord_token_embeddings(model: ContrastiveRewardModel,
                                 tokenizer_info: Dict,
                                 device: torch.device) -> Tuple[np.ndarray, List[str], List[int]]:
    """Extract embeddings for individual chord tokens"""
    model.eval()
    
    # Get chord token mappings
    token_to_chord = tokenizer_info.get('token_to_chord', {})
    
    chord_tokens = []
    chord_labels = []
    chord_token_ids = []
    
    # Extract unique chord tokens (skip onset/hold pairs for now)
    for token_id, chord_info in token_to_chord.items():
        token_id = int(token_id)
        if token_id == CHORD_SILENCE_TOKEN:
            chord_labels.append("SILENCE")
        else:
            root, intervals, inversion, is_onset = chord_info
            if is_onset:  # Only process onset tokens to avoid duplicates
                chord_label = f"R{root}_{intervals}_Inv{inversion}"
                chord_labels.append(chord_label)
        
        chord_tokens.append(token_id)
        chord_token_ids.append(token_id)
    
    # Create dummy sequences with just the chord token repeated
    seq_length = 16  # Short sequence for individual token embedding
    embeddings = []
    
    with torch.no_grad():
        for token_id in tqdm(chord_tokens, desc="Extracting chord token embeddings"):
            # Create a sequence with just this chord token
            dummy_sequence = torch.full((1, seq_length), token_id, 
                                      dtype=torch.long, device=device)
            
            # Get embedding (using chord encoder)
            embedding = model.chord_encoder(dummy_sequence)
            embeddings.append(embedding.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    return embeddings, chord_labels, chord_token_ids

def create_umap_visualization(embeddings: np.ndarray, 
                            labels: List[str],
                            title: str,
                            save_path: Optional[str] = None,
                            n_neighbors: int = 15,
                            min_dist: float = 0.1,
                            n_components: int = 2) -> np.ndarray:
    """Create UMAP visualization"""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42
    )
    
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # If we have many points, use smaller markers
    alpha = 0.7 if len(embeddings) > 100 else 0.8
    s = 20 if len(embeddings) > 100 else 50
    
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
                         alpha=alpha, s=s, c=range(len(labels)), cmap='tab20')
    
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    
    # Add some sample labels if not too many points
    if len(embeddings) <= 50:
        for i, label in enumerate(labels[:50]):  # Limit labels to avoid clutter
            plt.annotate(label, (umap_embeddings[i, 0], umap_embeddings[i, 1]), 
                        fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    return umap_embeddings

def create_melody_chord_joint_plot(melody_embeddings: np.ndarray,
                                 chord_embeddings: np.ndarray,
                                 song_ids: List[str],
                                 title: str = "Melody-Chord Joint Embedding Space",
                                 save_path: Optional[str] = None):
    """Create joint visualization of melody and chord embeddings"""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    # Combine embeddings
    combined_embeddings = np.vstack([melody_embeddings, chord_embeddings])
    
    # Create labels
    labels = ['Melody'] * len(melody_embeddings) + ['Chord'] * len(chord_embeddings)
    
    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embeddings = reducer.fit_transform(combined_embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot melody embeddings
    melody_points = umap_embeddings[:len(melody_embeddings)]
    chord_points = umap_embeddings[len(melody_embeddings):]
    
    plt.scatter(melody_points[:, 0], melody_points[:, 1], 
               alpha=0.6, s=30, c='blue', label='Melody', marker='o')
    plt.scatter(chord_points[:, 0], chord_points[:, 1], 
               alpha=0.6, s=30, c='red', label='Chord', marker='^')
    
    plt.title(title, fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved joint visualization to {save_path}")
    
    plt.show()
    
    return umap_embeddings

def analyze_chord_clusters(umap_embeddings: np.ndarray, 
                         chord_labels: List[str],
                         save_path: Optional[str] = None):
    """Analyze and visualize chord clusters"""
    
    # Group chords by root note
    root_groups = {}
    for i, label in enumerate(chord_labels):
        if label == "SILENCE":
            root = "SILENCE"
        else:
            # Extract root from label like "R0_[3, 4]_Inv0"
            try:
                root = label.split('_')[0][1:]  # Remove 'R' prefix
                root_groups.setdefault(root, []).append(i)
            except:
                root_groups.setdefault("OTHER", []).append(i)
    
    # Create visualization colored by root
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(root_groups)))
    
    for color, (root, indices) in zip(colors, root_groups.items()):
        points = umap_embeddings[indices]
        plt.scatter(points[:, 0], points[:, 1], 
                   c=[color], label=f'Root {root}', alpha=0.7, s=50)
    
    plt.title('Chord Embeddings Colored by Root Note', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved chord cluster analysis to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize model embeddings with UMAP")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["contrastive", "discriminative", "offline"],
                       help="Type of model to visualize")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="Maximum number of samples to visualize")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for data loading")
    
    args = parser.parse_args()
    
    if not UMAP_AVAILABLE:
        print("Error: UMAP not available. Install with: pip install umap-learn")
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    dataloader, tokenizer_info = create_dataloader(
        data_dir=Path(args.data_dir),
        split="valid",  # Use validation set for visualization
        batch_size=args.batch_size,
        num_workers=4,
        sequence_length=256,
        mode='contrastive' if args.model_type == 'contrastive' else 'online',
        shuffle=False
    )
    
    # Load model
    print(f"Loading {args.model_type} model...")
    if args.model_type == "contrastive":
        model = ContrastiveRewardModel(
            melody_vocab_size=tokenizer_info['melody_vocab_size'],
            chord_vocab_size=tokenizer_info['chord_vocab_size'],
            pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Extracting sequence-level embeddings...")
        melody_embeddings, chord_embeddings, song_ids = extract_contrastive_embeddings(
            model, dataloader, device, args.max_samples
        )
        
        # Visualize sequence-level embeddings
        print("Creating sequence-level visualizations...")
        create_umap_visualization(
            melody_embeddings, 
            [f"M_{sid[:8]}" for sid in song_ids], 
            "Melody Sequence Embeddings",
            save_path=output_dir / "melody_sequences_umap.png"
        )
        
        create_umap_visualization(
            chord_embeddings, 
            [f"C_{sid[:8]}" for sid in song_ids], 
            "Chord Sequence Embeddings",
            save_path=output_dir / "chord_sequences_umap.png"
        )
        
        create_melody_chord_joint_plot(
            melody_embeddings, chord_embeddings, song_ids,
            save_path=output_dir / "melody_chord_joint_umap.png"
        )
        
        # Visualize individual chord token embeddings
        print("Extracting chord token embeddings...")
        chord_token_embeddings, chord_labels, chord_token_ids = extract_chord_token_embeddings(
            model, tokenizer_info, device
        )
        
        print("Creating chord token visualizations...")
        chord_umap = create_umap_visualization(
            chord_token_embeddings,
            chord_labels,
            "Individual Chord Token Embeddings",
            save_path=output_dir / "chord_tokens_umap.png"
        )
        
        analyze_chord_clusters(
            chord_umap, chord_labels,
            save_path=output_dir / "chord_clusters_by_root.png"
        )
        
        # Save embeddings for further analysis
        print("Saving embeddings...")
        np.savez(
            output_dir / "embeddings.npz",
            melody_embeddings=melody_embeddings,
            chord_embeddings=chord_embeddings,
            chord_token_embeddings=chord_token_embeddings,
            song_ids=song_ids,
            chord_labels=chord_labels,
            chord_token_ids=chord_token_ids
        )
        
    else:
        print(f"Visualization for {args.model_type} model not implemented yet")
        return
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main() 