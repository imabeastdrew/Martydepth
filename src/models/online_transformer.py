import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.training.config import TrainingConfig
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        output = self.out_proj(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output
        
        return x

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
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        # Invert mask (1s become 0s and vice versa) and convert to float
        return ~mask
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on interleaved sequence
        
        Args:
            tokens: Interleaved chord/melody sequence [batch_size, seq_length]
                   Format: [chord_1, melody_1, chord_2, melody_2, ...]
        
        Returns:
            logits: Prediction logits for next token [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = tokens.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(tokens)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Apply transformer with causal masking
        x = self.transformer(x, is_causal=True)  # Use built-in causal masking
        
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