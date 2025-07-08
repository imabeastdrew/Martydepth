import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class OfflineTeacherEmbeddings(nn.Module):
    """Separate embeddings for encoder (melody) and decoder (chords)"""
    
    def __init__(self, melody_vocab_size, chord_vocab_size, embed_dim, max_seq_length, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # Token embeddings (512-dim)
        self.melody_embedding = nn.Embedding(melody_vocab_size, embed_dim)
        self.chord_embedding = nn.Embedding(chord_vocab_size, embed_dim)
        
        # Separate positional encodings for encoder/decoder
        self.encoder_position = nn.Embedding(max_seq_length, embed_dim)
        self.decoder_position = nn.Embedding(max_seq_length, embed_dim)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with proper scaling"""
        # Initialize embeddings with 1/sqrt(d) scaling
        std = 1.0 / math.sqrt(self.embed_dim)
        nn.init.normal_(self.melody_embedding.weight, mean=0, std=std)
        nn.init.normal_(self.chord_embedding.weight, mean=0, std=std)
        nn.init.normal_(self.encoder_position.weight, mean=0, std=std)
        nn.init.normal_(self.decoder_position.weight, mean=0, std=std)
    
    def encode_melody(self, melody_tokens):
        """Embed melody tokens for encoder"""
        batch_size, seq_length = melody_tokens.shape
        
        # Replace pad tokens with a valid index (0) and then zero out their embeddings
        padding_mask = (melody_tokens == self.pad_token_id)
        safe_tokens = melody_tokens.clone()
        safe_tokens[padding_mask] = 0
        
        # Token embeddings
        token_embeds = self.melody_embedding(safe_tokens)
        token_embeds[padding_mask] = 0.0
        
        # Positional embeddings
        positions = torch.arange(seq_length, device=melody_tokens.device)
        position_embeds = self.encoder_position(positions).unsqueeze(0)
        
        return token_embeds + position_embeds
    
    def encode_chords(self, chord_tokens):
        """Embed chord tokens for decoder"""
        batch_size, seq_length = chord_tokens.shape
        
        # Replace pad tokens with a valid index (0) and then zero out their embeddings
        padding_mask = (chord_tokens == self.pad_token_id)
        safe_tokens = chord_tokens.clone()
        safe_tokens[padding_mask] = 0

        # Token embeddings
        token_embeds = self.chord_embedding(safe_tokens)
        token_embeds[padding_mask] = 0.0
        
        # Positional embeddings
        positions = torch.arange(seq_length, device=chord_tokens.device)
        position_embeds = self.decoder_position(positions).unsqueeze(0)
        
        return token_embeds + position_embeds


class CrossAttention(nn.Module):
    """Cross-attention mechanism for decoder to attend to encoder outputs"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from decoder, Key/Value from encoder
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_hidden, encoder_outputs, encoder_mask=None):
        """
        decoder_hidden: [batch, decoder_seq, embed_dim] - queries
        encoder_outputs: [batch, encoder_seq, embed_dim] - keys/values
        encoder_mask: [batch, encoder_seq] - optional padding mask
        """
        batch_size, decoder_seq, _ = decoder_hidden.shape
        encoder_seq = encoder_outputs.shape[1]
        
        # Project to Q, K, V
        Q = self.query_proj(decoder_hidden)   # [batch, decoder_seq, embed_dim]
        K = self.key_proj(encoder_outputs)    # [batch, encoder_seq, embed_dim]
        V = self.value_proj(encoder_outputs)  # [batch, encoder_seq, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, decoder_seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, encoder_seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, encoder_seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: [batch, num_heads, decoder_seq, encoder_seq]
        
        # Apply encoder mask if provided
        if encoder_mask is not None:
            mask = encoder_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, encoder_seq]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and apply to values
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        # Shape: [batch, num_heads, decoder_seq, head_dim]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, decoder_seq, self.embed_dim
        )
        
        # Final projection
        output = self.output_proj(context)
        return output


class DecoderBlock(nn.Module):
    """Decoder block with self-attention, cross-attention, and feedforward"""
    
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        # Cross-attention to encoder
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        
        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Initialize layer norms
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm1.bias, 0.0)
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm2.bias, 0.0)
        nn.init.constant_(self.norm3.weight, 1.0)
        nn.init.constant_(self.norm3.bias, 0.0)
        
        # Initialize feedforward
        nn.init.normal_(self.feedforward[0].weight, mean=0, std=1.0/math.sqrt(embed_dim))
        nn.init.zeros_(self.feedforward[0].bias)
        nn.init.normal_(self.feedforward[2].weight, mean=0, std=1.0/math.sqrt(feedforward_dim))
        nn.init.zeros_(self.feedforward[2].bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, decoder_input, encoder_outputs, decoder_mask=None, encoder_mask=None):
        """
        decoder_input: [batch, decoder_seq, embed_dim]
        encoder_outputs: [batch, encoder_seq, embed_dim]
        decoder_mask: causal mask for self-attention
        encoder_mask: padding mask for cross-attention
        """
        # 1. Self-attention (causal - can't see future chords)
        self_attn_out, _ = self.self_attention(
            decoder_input.transpose(0, 1),
            decoder_input.transpose(0, 1),
            decoder_input.transpose(0, 1),
            attn_mask=decoder_mask
        )
        self_attn_out = self_attn_out.transpose(0, 1)
        decoder_input = self.norm1(decoder_input + self.dropout(self_attn_out))
        
        # 2. Cross-attention (attend to full melody)
        cross_attn_out = self.cross_attention(decoder_input, encoder_outputs, encoder_mask)
        decoder_input = self.norm2(decoder_input + self.dropout(cross_attn_out))
        
        # 3. Feedforward
        ff_out = self.feedforward(decoder_input)
        output = self.norm3(decoder_input + self.dropout(ff_out))
        
        return output


class OfflineTeacherModel(nn.Module):
    """
    T5-style encoder-decoder for offline melody->chord generation
    Symmetric 4+4 architecture matching ReaLChords paper
    """
    
    def __init__(self,
                 melody_vocab_size: int,
                 chord_vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 max_seq_length: int,
                 pad_token_id: int):
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.embeddings = OfflineTeacherEmbeddings(
            melody_vocab_size=melody_vocab_size,
            chord_vocab_size=chord_vocab_size,
            embed_dim=embed_dim,
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id
        )
        
        # Full Transformer using PyTorch's implementation
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Use Pre-LN for stability
        )
        
        # Output projection to chord vocabulary
        self.output_head = nn.Linear(embed_dim, chord_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"🎵 Offline Teacher Model Initialized:")
        print(f"  Architecture: {num_layers}E + {num_layers}D")
        print(f"  Embed dimension: {embed_dim}")
        print(f"  Attention heads: {num_heads}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """Count total model parameters"""
        return sum(p.numel() for p in self.parameters())
    
    @staticmethod
    def create_causal_mask(seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for decoder self-attention.
        PyTorch expects additive attention mask where -inf means no attention.
        """
        # Create causal mask [seq_length, seq_length]
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        mask = mask.to(device)
        
        # Convert to additive attention mask
        causal_mask = torch.zeros_like(mask, dtype=torch.float)
        causal_mask.masked_fill_(mask, float('-inf'))
        return causal_mask
    
    def forward(
        self,
        melody_tokens: torch.Tensor,
        chord_tokens: torch.Tensor,
        melody_mask: Optional[torch.Tensor] = None,
        chord_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            melody_tokens: [batch, seq] - melody token sequence
            chord_tokens: [batch, seq] - chord token sequence
            melody_mask: [batch, seq] - True for padding tokens in melody
            chord_mask: [batch, seq] - True for padding tokens in chords
        """
        # Get embeddings
        melody_embeds = self.embeddings.encode_melody(melody_tokens)  # [batch, seq, embed_dim]
        chord_embeds = self.embeddings.encode_chords(chord_tokens)    # [batch, seq, embed_dim]
        
        # Create causal mask for decoder
        seq_length = chord_tokens.size(1)
        causal_mask = self.create_causal_mask(seq_length, chord_tokens.device)
        
        # Convert padding masks to PyTorch format (True blocks attention)
        src_key_padding_mask = melody_mask if melody_mask is not None else None
        tgt_key_padding_mask = chord_mask if chord_mask is not None else None
        
        # Forward through transformer
        # Note: PyTorch expects tgt_mask to be additive (not multiplicative)
        # and src/tgt_key_padding_mask where True means to block attention
        output = self.transformer(
            src=melody_embeds,                      # Encoder input (melody)
            tgt=chord_embeds,                       # Decoder input (chords)
            src_mask=None,                          # No causal mask for encoder
            tgt_mask=causal_mask,                   # Causal mask for decoder
            memory_mask=None,                       # No mask for cross-attention
            src_key_padding_mask=src_key_padding_mask,  # Padding mask for encoder
            tgt_key_padding_mask=tgt_key_padding_mask,  # Padding mask for decoder
            memory_key_padding_mask=src_key_padding_mask  # Use encoder padding mask for cross-attention
        )
        
        # Project to vocabulary
        logits = self.output_head(output)  # [batch, seq, chord_vocab_size]
        return logits


def _test_model():
    """Test the OfflineTeacherModel with dummy data"""
    # Configuration
    batch_size = 4
    seq_length = 256
    melody_vocab = 257
    chord_vocab = 4577
    
    # Dummy data
    melody = torch.randint(0, melody_vocab, (batch_size, seq_length))
    chords = torch.randint(0, chord_vocab, (batch_size, seq_length))
    
    # Create a dummy padding mask (e.g., last 10 tokens are padding)
    melody_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    melody_mask[:, -10:] = 0
    
    # Initialize model
    model = OfflineTeacherModel(
        melody_vocab_size=melody_vocab,
        chord_vocab_size=chord_vocab,
        embed_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_seq_length=256,
        pad_token_id=177
    )
    
    # Forward pass
    logits = model(melody, chords, melody_mask=~melody_mask) # PyTorch expects False for padding
    
    # Check shapes
    print("\n--- Model Test ---")
    print(f"Input melody shape: {melody.shape}")
    print(f"Input chords shape: {chords.shape}")
    print(f"Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_length, chord_vocab)
    print("✅ Test passed!")

if __name__ == '__main__':
    _test_model() 