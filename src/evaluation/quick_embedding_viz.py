#!/usr/bin/env python3
"""
Quick UMAP visualization of chord and melody token embeddings from random initialization.
Use this to get a feel for UMAP before training models.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

from src.models.contrastive_reward_model import ContrastiveRewardModel
from src.config.tokenization_config import CHORD_TOKEN_START, PAD_TOKEN

def quick_chord_embedding_viz():
    """Quick visualization of randomly initialized chord embeddings"""
    if not UMAP_AVAILABLE:
        print("Install UMAP first: pip install umap-learn")
        return
    
    # Load tokenizer info
    project_root = Path(__file__).parent.parent.parent
    tokenizer_path = project_root / "data" / "interim" / "train" / "tokenizer_info.json"
    
    if not tokenizer_path.exists():
        print(f"Tokenizer info not found at {tokenizer_path}")
        print("Run preprocessing first!")
        return
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_info = json.load(f)
    
    # Create a small contrastive model
    model = ContrastiveRewardModel(
        melody_vocab_size=tokenizer_info['melody_vocab_size'],
        chord_vocab_size=tokenizer_info['chord_vocab_size'],
        embed_dim=128,  # Smaller for quick viz
        num_heads=4,
        num_layers=2,
        pad_token_id=tokenizer_info.get('pad_token_id', PAD_TOKEN)
    )
    
    print("Extracting chord token embeddings from random initialization...")
    
    # Get chord token mappings
    token_to_chord = tokenizer_info.get('token_to_chord', {})
    
    chord_embeddings = []
    chord_labels = []
    chord_roots = []
    
    # Extract embeddings for first 100 chord tokens to keep it manageable
    count = 0
    for token_id, chord_info in token_to_chord.items():
        if count >= 100:  # Limit for quick viz
            break
            
        token_id = int(token_id)
        
        # Get embedding from the chord encoder's embedding layer
        with torch.no_grad():
            embedding = model.chord_encoder.token_embedding(torch.tensor([token_id])).numpy()
            chord_embeddings.append(embedding)
        
        # Create labels
        if len(chord_info) == 4:  # (root, intervals, inversion, is_onset)
            root, intervals, inversion, is_onset = chord_info
            if is_onset:
                chord_labels.append(f"R{root}_I{inversion}")
                chord_roots.append(root)
            else:
                chord_labels.append(f"R{root}_I{inversion}_hold")
                chord_roots.append(root)
        else:
            chord_labels.append("SILENCE")
            chord_roots.append(-1)
        
        count += 1
    
    chord_embeddings = np.vstack(chord_embeddings)
    
    print(f"Visualizing {len(chord_embeddings)} chord token embeddings...")
    
    # Fit UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embeddings = reducer.fit_transform(chord_embeddings)
    
    # Create visualization colored by root note
    plt.figure(figsize=(14, 10))
    
    # Group by root note for coloring
    unique_roots = list(set(chord_roots))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_roots)))
    
    for root, color in zip(unique_roots, colors):
        mask = np.array(chord_roots) == root
        if np.sum(mask) > 0:
            plt.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                       c=[color], label=f'Root {root}' if root != -1 else 'Silence', 
                       alpha=0.7, s=50)
    
    plt.title('Chord Token Embeddings (Random Initialization)\nColored by Root Note', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save visualization
    viz_dir = Path("./visualizations")
    viz_dir.mkdir(exist_ok=True)
    save_path = viz_dir / "quick_chord_embeddings_umap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    print("\nNote: This shows embeddings from random initialization.")
    print("After training, you should see much more meaningful clusters!")
    print("Train a contrastive model and use visualize_embeddings.py for trained embeddings.")

if __name__ == "__main__":
    quick_chord_embedding_viz() 