import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class OfflineTeacherEmbeddings(nn.Module):
    """Separate embeddings for encoder (melody) and decoder (chords)"""
    
    def __init__(self, melody_vocab_size, chord_vocab_size, embed_dim, max_seq_length):
        super().__init__()
        
        # Token embeddings (512-dim)
        self.melody_embedding = nn.Embedding(melody_vocab_size, embed_dim)
        self.chord_embedding = nn.Embedding(chord_vocab_size, embed_dim)
        
        # Separate positional encodings for encoder/decoder
        self.encoder_position = nn.Embedding(max_seq_length, embed_dim)
        self.decoder_position = nn.Embedding(max_seq_length, embed_dim)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with small random values"""
        nn.init.normal_(self.melody_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.chord_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.encoder_position.weight, mean=0, std=0.02)
        nn.init.normal_(self.decoder_position.weight, mean=0, std=0.02)
    
    def encode_melody(self, melody_tokens):
        """Embed melody tokens for encoder"""
        batch_size, seq_length = melody_tokens.shape
        
        # Token embeddings
        token_embeds = self.melody_embedding(melody_tokens)
        
        # Positional embeddings
        positions = torch.arange(seq_length, device=melody_tokens.device)
        position_embeds = self.encoder_position(positions).unsqueeze(0)
        
        return token_embeds + position_embeds
    
    def encode_chords(self, chord_tokens):
        """Embed chord tokens for decoder"""
        batch_size, seq_length = chord_tokens.shape
        
        # Token embeddings
        token_embeds = self.chord_embedding(chord_tokens)
        
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
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_seq_length: int = 256):
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_sequence_length = max_seq_length
        
        # Embeddings
        self.embeddings = OfflineTeacherEmbeddings(
            melody_vocab_size, chord_vocab_size, embed_dim, max_seq_length
        )
        
        # Full Transformer using PyTorch's implementation
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
                batch_first=True
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
        """Create a causal mask for the decoder."""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(self, melody_tokens: torch.Tensor, chord_tokens: torch.Tensor, melody_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the teacher model.
        
        Args:
            melody_tokens: [batch, melody_seq]
            chord_tokens: [batch, chord_seq]
            melody_mask: [batch, melody_seq] - Padding mask for melody
        
        Returns:
            logits: [batch, chord_seq, chord_vocab_size]
        """
        # --- Detailed Input Validation ---
        melody_vocab_size = self.embeddings.melody_embedding.num_embeddings
        if torch.any(melody_tokens >= melody_vocab_size):
            max_token = torch.max(melody_tokens)
            raise ValueError(
                f"Melody token index out of range. "
                f"Max token ID: {max_token}, Vocab size: {melody_vocab_size}"
            )
        if torch.any(melody_tokens < 0):
            raise ValueError("Melody token contains negative indices.")

        chord_vocab_size = self.embeddings.chord_embedding.num_embeddings
        if torch.any(chord_tokens >= chord_vocab_size):
            max_token = torch.max(chord_tokens)
            raise ValueError(
                f"Chord token index out of range. "
                f"Max token ID: {max_token}, Vocab size: {chord_vocab_size}"
            )
        if torch.any(chord_tokens < 0):
            raise ValueError("Chord token contains negative indices.")
        # --------------------------------

        # 1. Get embeddings
        melody_embed = self.embeddings.encode_melody(melody_tokens)
        chord_embed = self.embeddings.encode_chords(chord_tokens)
        
        # 2. Create causal mask for the decoder
        chord_seq_length = chord_tokens.size(1)
        decoder_causal_mask = self.create_causal_mask(chord_seq_length, chord_tokens.device)
        
        # 3. Pass through the transformer
        # Note: src_key_padding_mask is for encoder, memory_key_padding_mask is for decoder cross-attention
        transformer_output = self.transformer(
            src=melody_embed,
            tgt=chord_embed,
            tgt_mask=decoder_causal_mask,
            src_key_padding_mask=melody_mask,
            memory_key_padding_mask=melody_mask
        )
        
        # 4. Project to vocabulary
        logits = self.output_head(transformer_output)
        
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
        chord_vocab_size=chord_vocab
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