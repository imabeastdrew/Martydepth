#!/usr/bin/env python3
"""
Discriminative Reward Model for classifying melody-chord pairs.
"""

import torch
import torch.nn as nn
from typing import Optional

class DiscriminativeRewardModel(nn.Module):
    """
    A multi-scale Transformer-based model to discriminate between real and fake melody-chord pairs.
    
    It processes an interleaved sequence of melody and chord tokens and outputs a
    probability that the pair is "real". The model supports multi-scale evaluation
    by processing fragments of different lengths using sliding windows with 50% overlap.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_length: int,
                 pad_token_id: int,
                 scale_factor: float = 1.0):  # Scale factor for fragment length
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length + 1, embed_dim)
        self.pad_token_id = pad_token_id
        self.scale_factor = scale_factor
        self.fragment_length = int(max_seq_length * scale_factor)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classification_head = nn.Linear(embed_dim, 1)
        self.max_seq_length = max_seq_length

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

    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using sliding windows for multi-scale evaluation.
        
        Args:
            tokens: Interleaved melody-chord tokens [batch_size, seq_length]
            padding_mask: Optional padding mask

        Returns:
            torch.Tensor: Logits for each window [batch_size * num_windows, 1]
        """
        # Create sliding windows
        token_windows = self.get_sliding_windows(tokens)
        if padding_mask is not None:
            padding_windows = self.get_sliding_windows(padding_mask)
        else:
            padding_windows = None
            
        batch_size, window_length = token_windows.shape
        positions = torch.arange(window_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(token_windows)
        pos_embeds = self.position_embedding(positions)
        
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_windows)
        
        # Use mean pooling over the sequence dimension
        if padding_windows is not None:
            masked_x = x.masked_fill(padding_windows.unsqueeze(-1), 0.0)
            sum_x = torch.sum(masked_x, dim=1)
            num_non_padded = (~padding_windows).sum(dim=1).unsqueeze(-1)
            num_non_padded = num_non_padded.clamp(min=1)
            pooled_output = sum_x / num_non_padded
        else:
            pooled_output = torch.mean(x, dim=1)
            
        # Get logits for each window
        logits = self.classification_head(pooled_output)
        
        return logits

    def get_reward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Calculates rewards for each window and aggregates them.
        
        Args:
            tokens: Interleaved melody-chord tokens [batch_size, seq_length]
            
        Returns:
            torch.Tensor: Aggregated rewards for each sequence in the batch [batch_size]
        """
        batch_size = tokens.shape[0]
        logits = self.forward(tokens)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Reshape to [batch_size, num_windows]
        num_windows = probs.shape[0] // batch_size
        probs = probs.view(batch_size, num_windows)
        
        # Average across windows to get final reward
        return torch.mean(probs, dim=1)

if __name__ == '__main__':
    # Test the model
    batch_size = 16
    seq_length = 256
    vocab_size = 4000

    model = DiscriminativeRewardModel(
        vocab_size=vocab_size,
        max_seq_length=seq_length
    )

    # Dummy interleaved sequence
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Test with no padding
    logits = model(tokens)
    print("--- Without padding ---")
    print("Logits shape:", logits.shape)
    
    # Test with padding
    padding = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    padding[:, seq_length//2:] = True # Pad half the sequence
    
    logits_padded = model(tokens, padding_mask=padding)
    print("\n--- With padding ---")
    print("Padded logits shape:", logits_padded.shape)

    # Verify that padding gives different results
    print("Logits are different with padding:", not torch.allclose(logits, logits_padded))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}") 