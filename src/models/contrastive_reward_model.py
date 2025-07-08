#!/usr/bin/env python3
"""
Contrastive Reward Model for melody and chord progression similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class Encoder(nn.Module):
    """A transformer encoder module with embedding."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_length: int,
                 pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length + 1, embed_dim)
        
        # Initialize embeddings with proper scaling
        std = 1.0 / math.sqrt(embed_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0, std=std)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=std)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # Use Post-LN for initial stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize transformer layer norms
        for layer in self.transformer.layers:
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0.0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0.0)
            # Initialize feedforward layers
            nn.init.normal_(layer.linear1.weight, mean=0, std=1.0/math.sqrt(embed_dim))
            nn.init.zeros_(layer.linear1.bias)
            nn.init.normal_(layer.linear2.weight, mean=0, std=1.0/math.sqrt(4 * embed_dim))
            nn.init.zeros_(layer.linear2.bias)
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length

    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = tokens.shape
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max_seq_length {self.max_seq_length}")

        positions = torch.arange(seq_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Replace pad tokens with a valid index (0) before embedding
        safe_tokens = tokens.clone()
        safe_tokens[tokens == self.pad_token_id] = 0
        token_embeds = self.token_embedding(safe_tokens)
        # Zero out embeddings for pad tokens after lookup
        token_embeds[tokens == self.pad_token_id] = 0.0
        
        pos_embeds = self.position_embedding(positions)
        
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        if padding_mask is not None:
            masked_x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            sum_x = torch.sum(masked_x, dim=1)
            num_non_padded = (~padding_mask).sum(dim=1).unsqueeze(-1)
            num_non_padded = num_non_padded.clamp(min=1)
            embedding = sum_x / num_non_padded
        else:
            embedding = torch.mean(x, dim=1)
            
        return embedding

class ContrastiveRewardModel(nn.Module):
    """
    Multi-scale contrastive reward model that computes similarity between melody and chord sequences.
    
    The model consists of two identical transformer encoders, one for melody
    and one for chords. The encoders map their respective inputs to
    fixed-size embeddings. The reward (similarity) is the cosine
    similarity between these two embeddings.
    
    The model supports multi-scale evaluation by processing fragments of different lengths
    using sliding windows with 50% overlap.
    """
    def __init__(self,
                 melody_vocab_size: int,
                 chord_vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_length: int,
                 pad_token_id: int,
                 scale_factor: float = 1.0):  # Scale factor for fragment length
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.scale_factor = scale_factor
        self.fragment_length = int(max_seq_length * scale_factor)
        
        self.melody_encoder = Encoder(
            vocab_size=melody_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id
        )
        
        self.chord_encoder = Encoder(
            vocab_size=chord_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id
        )

    def get_sliding_windows(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Creates overlapping windows of tokens with 50% overlap.
        
        Args:
            tokens: Input tensor of shape [batch_size, seq_length]
            
        Returns:
            Tensor of shape [batch_size * num_windows, fragment_length]
        """
        batch_size, seq_length = tokens.shape
        stride = self.fragment_length // 2  # 50% overlap
        
        # Calculate number of complete windows
        num_windows = max(1, (seq_length - self.fragment_length) // stride + 1)
        
        windows = []
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + self.fragment_length
            if end_idx > seq_length:
                # Pad the last window if needed
                window = tokens[:, start_idx:seq_length]
                padding_size = self.fragment_length - window.size(1)
                padding = torch.full((batch_size, padding_size), self.pad_token_id, 
                                  device=tokens.device)
                window = torch.cat([window, padding], dim=1)
            else:
                window = tokens[:, start_idx:end_idx]
            windows.append(window)
            
        return torch.cat(windows, dim=0)

    def forward(self, 
                melody_tokens: torch.Tensor, 
                chord_tokens: torch.Tensor,
                melody_padding_mask: Optional[torch.Tensor] = None,
                chord_padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass using sliding windows for multi-scale evaluation.
        
        Args:
            melody_tokens: Tensor of melody tokens [batch_size, seq_length]
            chord_tokens: Tensor of chord tokens [batch_size, seq_length]
            melody_padding_mask: Optional padding mask for melody tokens
            chord_padding_mask: Optional padding mask for chord tokens

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Melody and chord embeddings for each window
        """
        # Create sliding windows
        melody_windows = self.get_sliding_windows(melody_tokens)
        chord_windows = self.get_sliding_windows(chord_tokens)
        
        if melody_padding_mask is not None:
            melody_padding_windows = self.get_sliding_windows(melody_padding_mask)
        else:
            melody_padding_windows = None
            
        if chord_padding_mask is not None:
            chord_padding_windows = self.get_sliding_windows(chord_padding_mask)
        else:
            chord_padding_windows = None
        
        # Get embeddings for each window
        melody_embedding = self.melody_encoder(melody_windows, melody_padding_windows)
        chord_embedding = self.chord_encoder(chord_windows, chord_padding_windows)
        
        return melody_embedding, chord_embedding

    def get_reward(self, melody_tokens: torch.Tensor, chord_tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculates rewards for each window and aggregates them.
        
        Args:
            melody_tokens: Tensor of melody tokens [batch_size, seq_length]
            chord_tokens: Tensor of chord tokens [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Aggregated rewards for each sequence in the batch [batch_size]
        """
        batch_size = melody_tokens.shape[0]
        melody_embedding, chord_embedding = self.forward(melody_tokens, chord_tokens)
        
        # Calculate cosine similarity for each window
        similarities = F.cosine_similarity(melody_embedding, chord_embedding, dim=-1)
        
        # Reshape to [batch_size, num_windows]
        num_windows = similarities.shape[0] // batch_size
        similarities = similarities.view(batch_size, num_windows)
        
        # Average across windows to get final reward
        return torch.mean(similarities, dim=1)

if __name__ == '__main__':
    # Test the model
    batch_size = 16
    seq_length = 256
    melody_vocab = 1000
    chord_vocab = 500

    model = ContrastiveRewardModel(
        melody_vocab_size=melody_vocab,
        chord_vocab_size=chord_vocab,
        max_seq_length=seq_length
    )

    melody = torch.randint(0, melody_vocab, (batch_size, seq_length))
    chords = torch.randint(0, chord_vocab, (batch_size, seq_length))
    
    # Test with no padding
    mel_emb, chord_emb = model(melody, chords)
    print("--- Without padding ---")
    print("Melody embedding shape:", mel_emb.shape)
    print("Chord embedding shape:", chord_emb.shape)
    
    reward = model.get_reward(melody, chords)
    print("Reward shape:", reward.shape)
    print("Example rewards:", reward[:4])

    # Test with padding
    melody_padding = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    melody_padding[:, seq_length//2:] = True # Pad half the sequence
    
    chord_padding = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    chord_padding[:, seq_length-10:] = True # Pad last 10 tokens

    mel_emb_padded, chord_emb_padded = model(
        melody, chords, 
        melody_padding_mask=melody_padding, 
        chord_padding_mask=chord_padding
    )
    print("\n--- With padding ---")
    print("Padded melody embedding shape:", mel_emb_padded.shape)
    print("Padded chord embedding shape:", chord_emb_padded.shape)

    # Verify that padding gives different results
    print("Embeddings are different with padding:", not torch.allclose(mel_emb, mel_emb_padded))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}") 