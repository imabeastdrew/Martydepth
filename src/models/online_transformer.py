import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class OnlineTransformer(nn.Module):
    """
    Transformer model for online chord prediction from interleaved melody-chord sequences.
    
    Input format:
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
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_length: int,
                 pad_token_id: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        
        # Single embedding table for all tokens
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length + 1, embed_dim)
        
        # Initialize embeddings with normal distribution scaled by 1/sqrt(embed_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))
        
        # Layer normalization layers
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Initialize layer norms
        nn.init.constant_(self.pre_transformer_norm.weight, 1.0)
        nn.init.constant_(self.pre_transformer_norm.bias, 0.0)
        nn.init.constant_(self.final_norm.weight, 1.0)
        nn.init.constant_(self.final_norm.bias, 0.0)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,  # 2048 for embed_dim=512
            dropout=dropout,
            batch_first=True,
            norm_first=False  # Use Post-LN initially for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Single output head for next token prediction
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize output head
        nn.init.normal_(self.output_head.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))
        nn.init.zeros_(self.output_head.bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens.
        
        Returns a boolean tensor where True indicates positions to be masked.
        """
        # Create a mask where True values indicate positions to be masked (future tokens)
        return torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1)
        
    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass on interleaved sequence
        
        Args:
            tokens: Interleaved chord/melody sequence [batch_size, seq_length]
            padding_mask: Boolean mask for padding tokens [batch_size, seq_length]
                        True indicates positions to be masked
        
        Returns:
            logits: Prediction logits for next token [batch_size, seq_length, vocab_size]
        """
        batch_size, seq_length = tokens.shape
        
        if seq_length > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_length} exceeds max_seq_length {self.max_seq_length}")
        
        # Create position indices and causal mask
        positions = torch.arange(seq_length, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        causal_mask = self.create_causal_mask(seq_length).to(tokens.device)
        
        # Get embeddings safely
        safe_tokens = tokens.clone()
        safe_tokens[safe_tokens >= self.vocab_size] = 0  # Clamp to valid range
        token_embeds = self.token_embedding(safe_tokens)
        pos_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Apply pre-transformer normalization and dropout
        x = self.pre_transformer_norm(x)
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Final layer norm and output projection
        x = self.final_norm(x)
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
    model = OnlineTransformer(vocab_size=vocab_size, embed_dim=512, num_heads=8, num_layers=8, dropout=0.1, max_seq_length=512, pad_token_id=177)
    
    # Forward pass
    logits = model(tokens)
    
    # Print shapes
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 