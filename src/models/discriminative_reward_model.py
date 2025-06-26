#!/usr/bin/env python3
"""
Discriminative Reward Model for classifying melody-chord pairs.
"""

import torch
import torch.nn as nn
from typing import Optional

class DiscriminativeRewardModel(nn.Module):
    """
    A Transformer-based model to discriminate between real and fake melody-chord pairs.
    
    It processes an interleaved sequence of melody and chord tokens and outputs a
    single logit indicating the probability that the pair is "real".
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_length: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length + 1, embed_dim)
        
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

    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the discriminative reward model.
        
        Args:
            tokens (torch.Tensor): Interleaved melody-chord tokens [batch_size, seq_length]
            padding_mask (Optional[torch.Tensor]): Padding mask.

        Returns:
            torch.Tensor: A single logit for each sequence in the batch [batch_size, 1].
        """
        batch_size, seq_length = tokens.shape
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max_seq_length {self.max_seq_length}")
        
        positions = torch.arange(seq_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        token_embeds = self.token_embedding(tokens)
        pos_embeds = self.position_embedding(positions)
        
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Use mean pooling over the sequence dimension to get a fixed-size representation
        if padding_mask is not None:
            # Adjust for padding to avoid bias
            masked_x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            sum_x = torch.sum(masked_x, dim=1)
            num_non_padded = (~padding_mask).sum(dim=1).unsqueeze(-1)
            num_non_padded = num_non_padded.clamp(min=1)
            pooled_output = sum_x / num_non_padded
        else:
            pooled_output = torch.mean(x, dim=1)
            
        # Get the final logit
        logit = self.classification_head(pooled_output)
        
        return logit

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