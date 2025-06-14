import torch
import torch.nn as nn
import math

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
    
    Input format:
    - Each frame contains [melody_t, chord_t] tokens
    - At time t, model sees [melody_0, chord_0, ..., melody_{t-1}, chord_{t-1}]
    - Must predict chord_t without seeing melody_t (online constraint)
    
    Sequence structure:
    - Even indices (0,2,4,...): melody tokens
    - Odd indices (1,3,5,...): chord tokens
    
    Vocabulary structure:
    - Token IDs 0 to 256: melody tokens (MIDI)
    - Token IDs 257 to 4832: chord tokens
    """
    
    def __init__(self,
                 vocab_size: int = 4833,  # Total vocabulary size (257 + 4576)
                 melody_vocab_size: int = 257,  # MIDI-based melody vocabulary
                 embed_dim: int = 480,  # Match ReaLChords paper
                 num_layers: int = 8,  # Match ReaLChords paper
                 num_heads: int = 6,  # Match ReaLChords paper
                 feedforward_dim: int = 1920,  # 4 * embed_dim
                 max_sequence_length: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # Store vocabulary sizes
        self.vocab_size = vocab_size
        self.melody_vocab_size = melody_vocab_size
        self.chord_vocab_size = vocab_size - melody_vocab_size
        
        # Token embeddings for the combined vocabulary
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings for the full sequence length
        self.position_embeddings = nn.Embedding(max_sequence_length, embed_dim)
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head only for chord predictions (more efficient)
        self.output_head = nn.Linear(embed_dim, self.chord_vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_length: int) -> torch.Tensor:
        """Create a causal mask for the sequence"""
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return ~mask  # Invert mask so True = can attend, False = cannot attend
    
    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_tokens: Combined melody and chord tokens [batch_size, seq_length]
                         Even positions (0,2,4...) are melody tokens
                         Odd positions (1,3,5...) are chord tokens
        
        Returns:
            chord_logits: Logits for chord predictions at odd positions
                         Shape: [batch_size, seq_length//2, chord_vocab_size]
        """
        batch_size, seq_length = input_tokens.shape
        
        # Create position masks (not token value masks)
        position_mask = torch.arange(seq_length, device=input_tokens.device) % 2
        melody_positions = (position_mask == 0)  # Even positions
        chord_positions = (position_mask == 1)   # Odd positions
        
        # Check melody tokens are in valid range at melody positions
        if torch.any((input_tokens[:, melody_positions] >= self.melody_vocab_size)):
            raise ValueError("Melody tokens must be in range [0, 257)")
        
        # Check chord tokens are in valid range at chord positions
        if torch.any((input_tokens[:, chord_positions] < self.melody_vocab_size) | 
                    (input_tokens[:, chord_positions] >= self.vocab_size)):
            raise ValueError("Chord tokens must be in range [257, 4833)")
        
        # Standard sequence position embeddings
        position_indices = torch.arange(seq_length, device=input_tokens.device)
        position_indices = position_indices.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_tokens)
        position_embeds = self.position_embeddings(position_indices)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Standard causal mask
        mask = self.create_causal_mask(seq_length).to(x.device)
        
        # Pass through transformer stack
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, mask=mask)
        
        # Extract chord positions and predict
        chord_embeds = x[:, 1::2, :]  # Odd positions = chord positions
        chord_logits = self.output_head(chord_embeds)  # Only chord vocabulary size
        
        return chord_logits

if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_length = 256
    
    # Create dummy input with proper token ranges (MIDI-based)
    melody_tokens = torch.randint(0, 257, (batch_size, seq_length//2))  # Melody tokens: 0-256
    chord_tokens = torch.randint(257, 4833, (batch_size, seq_length//2))  # Chord tokens: 257-4832
    
    # Interleave melody and chord tokens
    input_tokens = torch.stack([
        torch.stack([m, c], dim=1).flatten() 
        for m, c in zip(melody_tokens, chord_tokens)
    ], dim=0)
    
    # Initialize model
    model = OnlineTransformer()
    
    # Forward pass
    chord_logits = model(input_tokens)
    
    # Print shapes
    print(f"Input shape: {input_tokens.shape}")
    print(f"Output shape: {chord_logits.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}") 