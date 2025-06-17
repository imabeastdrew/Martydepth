import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class OnlineTransformer(nn.Module):
    """
    Transformer model for online chord prediction from interleaved melody-chord sequences.
    
    Input format (as per paper):
    - Single interleaved sequence [chord_1, melody_1, chord_2, melody_2, ...]
    - At time t, model sees all tokens up to t-1
    - Must predict next token given history
    
    Architecture:
    - Embedding dimension: 512
    - Number of heads: 8
    - Head dimension: 64 (512/8)
    - Number of layers: 8
    - Feedforward dimension: 2048 (4 * embed_dim)
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.1,
                 max_seq_length: int = 512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Single embedding table for all tokens
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,  # 2048 for embed_dim=512
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Single output head for next token prediction
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens"""
        # Create upper triangular mask (1s above diagonal)
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        # Invert mask (1s become 0s and vice versa) and convert to float
        return mask.masked_fill(mask == 1, float('-inf'))
        
    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass on interleaved sequence
        
        Args:
            tokens: Interleaved chord/melody sequence [batch_size, seq_length]
            padding_mask: Boolean mask for padding tokens [batch_size, seq_length]
        
        Returns:
            logits: Prediction logits for next token [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = tokens.shape
        
        # Create position indices and causal mask
        positions = torch.arange(seq_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        causal_mask = self.create_causal_mask(seq_length).to(tokens.device)
        
        # Get embeddings
        token_embeds = self.token_embedding(tokens)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Apply transformer with causal and padding masks
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Get predictions
        logits = self.output_head(x)
        
        return logits

if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_length = 256
    vocab_size = 4834  # Combined vocabulary size
    
    # Create dummy interleaved sequence
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Initialize model
    model = OnlineTransformer(vocab_size=vocab_size)
    
    # Forward pass
    logits = model(tokens)
    
    # Print shapes
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 