import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        if config.vocab_size is None:
            raise ValueError("vocab_size must be set in TrainingConfig before model initialization.")
        if config.chord_vocab_size is None:
            raise ValueError("chord_vocab_size must be set in TrainingConfig before model initialization.")
        
        # Store configuration
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.max_sequence_length = config.sequence_length
        
        # Embeddings
        self.embeddings = OfflineTeacherEmbeddings(
            config.melody_vocab_size, config.chord_vocab_size, config.embed_dim, config.sequence_length
        )
        
        # Encoder: 4 layers using standard transformer blocks
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # Decoder: 4 layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(config.embed_dim, config.num_heads, config.feedforward_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Output projection to chord vocabulary
        self.output_head = nn.Linear(config.embed_dim, config.chord_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        print(f"🎵 Offline Teacher Model Initialized:")
        print(f"  Architecture: {config.num_layers}E + {config.num_layers}D")
        print(f"  Embed dimension: {config.embed_dim}")
        print(f"  Attention heads: {config.num_heads}")
        print(f"  Total parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """Count total model parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def create_causal_mask(self, seq_length, device):
        """Create causal mask for decoder self-attention"""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    
    def encode(self, melody_tokens, melody_mask=None):
        """
        Encode melody sequence with bidirectional attention
        
        Args:
            melody_tokens: [batch, melody_seq] - melody token IDs
            melody_mask: [batch, melody_seq] - optional padding mask
        
        Returns:
            encoder_outputs: [batch, melody_seq, embed_dim]
        """
        # Embed melody tokens
        x = self.embeddings.encode_melody(melody_tokens)
        x = self.dropout(x)
        
        # Pass through encoder layers (bidirectional attention)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_key_padding_mask=melody_mask)
        
        return x
    
    def decode(self, chord_tokens, encoder_outputs, encoder_mask=None):
        """
        Decode chord sequence with cross-attention to melody
        
        Args:
            chord_tokens: [batch, chord_seq] - chord input tokens
            encoder_outputs: [batch, melody_seq, embed_dim] - from encode()
            encoder_mask: [batch, melody_seq] - optional padding mask
        
        Returns:
            decoder_outputs: [batch, chord_seq, embed_dim]
        """
        batch_size, chord_seq = chord_tokens.shape
        
        # Embed chord tokens
        x = self.embeddings.encode_chords(chord_tokens)
        x = self.dropout(x)
        
        # Create causal mask for decoder self-attention
        causal_mask = self.create_causal_mask(chord_seq, chord_tokens.device)
        
        # Pass through decoder layers
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_outputs, causal_mask, encoder_mask)
        
        return x
    
    def forward(self, melody_tokens, chord_tokens, melody_mask=None):
        """
        Full forward pass: encode melody, decode chords
        
        Args:
            melody_tokens: [batch, melody_seq] - complete melody sequence
            chord_tokens: [batch, chord_seq] - chord input sequence (shifted)
        
        Returns:
            chord_logits: [batch, chord_seq, chord_vocab_size] - chord predictions
        """
        # Encode complete melody (bidirectional)
        encoder_outputs = self.encode(melody_tokens, melody_mask)
        
        # Decode chords with cross-attention to melody
        decoder_outputs = self.decode(chord_tokens, encoder_outputs, melody_mask)
        
        # Project to chord vocabulary
        chord_logits = self.output_head(decoder_outputs)
        
        return chord_logits


# Test the offline teacher model
if __name__ == "__main__":
    print("🚀 Testing Offline Teacher Model...")
    
    # Initialize model
    model = OfflineTeacherModel(
        melody_vocab_size=177,
        chord_vocab_size=74,
        embed_dim=512,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=6,
        feedforward_dim=2048,
        max_sequence_length=256
    )
    
    # Test data
    batch_size = 2
    melody_seq = 10
    chord_seq = 10
    
    melody_tokens = torch.randint(0, 177, (batch_size, melody_seq))
    chord_tokens = torch.randint(0, 74, (batch_size, chord_seq))
    
    # Forward pass
    chord_logits = model(melody_tokens, chord_tokens)
    
    print(f"\n Test Results:")
    print(f"Melody input shape: {melody_tokens.shape}")
    print(f"Chord input shape: {chord_tokens.shape}")
    print(f"Chord logits shape: {chord_logits.shape}")
    print(f"Expected shape: [batch={batch_size}, seq={chord_seq}, vocab=74]")
    
    # Verify shapes
    assert chord_logits.shape == (batch_size, chord_seq, 74)
    print(" All shapes correct!") 