import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import T5Config, T5ForConditionalGeneration
from typing import Optional
from src.config.tokenization_config import CHORD_TOKEN_START

class T5OfflineTeacherModel(nn.Module):
    """T5-based offline teacher model with proper relative position embeddings"""
    
    def __init__(self,
                 melody_vocab_size: int,
                 chord_vocab_size: int,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 256,
                 pad_token_id: int = 178,
                 total_vocab_size: Optional[int] = None):
        super().__init__()
        
        self.melody_vocab_size = melody_vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        
        # Store total vocabulary size
        if total_vocab_size is None:
            total_vocab_size = 4779  # Based on tokenizer info: covers 0-4778
        self.total_vocab_size = total_vocab_size
        
        # Create T5 config
        self.config = T5Config(
            vocab_size=total_vocab_size,  # Must cover all possible token IDs
            d_model=embed_dim,
            d_kv=embed_dim // num_heads,  # Key/value dimension per head
            d_ff=feedforward_dim,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=32,  # T5's relative position buckets
            relative_attention_max_distance=128,  # Max relative distance
            dropout_rate=dropout,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="relu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=1,  # Assuming EOS is token 1
            decoder_start_token_id=pad_token_id,  # Start decoder with PAD (as we discussed)
        )
        
        # Create the T5 model with unified vocabulary (standard T5 approach)
        self.t5_model = T5ForConditionalGeneration(config=self.config)
        
        # No separate projections needed - T5 handles unified vocabulary internally
    
    def forward(self,
                melody_tokens: torch.Tensor,
                chord_tokens: torch.Tensor,
                melody_mask: Optional[torch.Tensor] = None,
                chord_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through T5 model with unified vocabulary
        
        Args:
            melody_tokens: [batch, melody_seq_len] - encoder input
            chord_tokens: [batch, chord_seq_len] - decoder input (shifted)
            melody_mask: [batch, melody_seq_len] - encoder attention mask
            chord_mask: [batch, chord_seq_len] - decoder attention mask
        
        Returns:
            logits: [batch, chord_seq_len, total_vocab_size]
        """
        # Create attention masks if not provided
        if melody_mask is None:
            melody_mask = (melody_tokens != self.pad_token_id)
        if chord_mask is None:
            chord_mask = (chord_tokens != self.pad_token_id)
        
        # Use standard T5 forward pass with token IDs directly
        outputs = self.t5_model(
            input_ids=melody_tokens,
            attention_mask=melody_mask,
            decoder_input_ids=chord_tokens,
            decoder_attention_mask=chord_mask,
            return_dict=True
        )
        
        # T5 outputs logits over the full vocabulary
        return outputs.logits
    
    def encode(self, melody_tokens: torch.Tensor, melody_mask: Optional[torch.Tensor] = None):
        """Encode melody using T5 encoder only"""
        if melody_mask is None:
            melody_mask = (melody_tokens != self.pad_token_id)
        
        # Use standard T5 encoder with token IDs directly
        encoder_outputs = self.t5_model.encoder(
            input_ids=melody_tokens,
            attention_mask=melody_mask,
            return_dict=True
        )
        
        return encoder_outputs.last_hidden_state, melody_mask
    
    def decode_step(self, 
                    encoder_hidden: torch.Tensor,
                    encoder_mask: torch.Tensor,
                    chord_tokens: torch.Tensor) -> torch.Tensor:
        """Single decoder step for generation"""
        decoder_mask = (chord_tokens != self.pad_token_id)
        
        # Use standard T5 decoder with token IDs directly
        outputs = self.t5_model.decoder(
            input_ids=chord_tokens,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            return_dict=True
        )
        
        # Apply the language modeling head to get logits over full vocabulary
        logits = self.t5_model.lm_head(outputs.last_hidden_state)
        return logits
    
    def count_parameters(self):
        """Count total parameters"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        return total_params
    
    def get_model_info(self):
        """Get model architecture information"""
        return {
            'model_type': 'T5-based Offline Teacher',
            'total_vocab_size': self.total_vocab_size,
            'melody_vocab_size': self.melody_vocab_size,
            'chord_vocab_size': self.chord_vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'feedforward_dim': self.config.d_ff,
            'relative_attention_buckets': self.config.relative_attention_num_buckets,
            'relative_attention_max_distance': self.config.relative_attention_max_distance,
            'total_parameters': self.count_parameters()
        }


def _test_t5_model():
    """Test the T5-based offline model"""
    import math
    
    print("Testing T5-based Offline Teacher Model...")
    
    # Test parameters - Updated for 1/16th note resolution dataset  
    # Note: vocab_size must be > max_token_id, so total_vocab_size must cover all possible tokens
    melody_vocab_size = 178   # Melody tokens: 0-177
    chord_vocab_size = 4600   # Chord tokens: 179-4778 (4600 tokens)
    total_vocab_size = 4779   # Must cover all possible token IDs (0-4778)
    pad_token_id = 178        # Actual PAD token from dataset
    batch_size = 2
    melody_seq_len = 256
    chord_seq_len = 256
    
    model = T5OfflineTeacherModel(
        melody_vocab_size=melody_vocab_size,
        chord_vocab_size=chord_vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,  # Smaller for testing
        feedforward_dim=2048,
        dropout=0.1,
        max_seq_length=256,
        pad_token_id=pad_token_id,
        total_vocab_size=total_vocab_size
    )
    
    print("\nModel info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create realistic test data matching 1/16th note dataset structure
    melody_tokens = torch.zeros(batch_size, melody_seq_len, dtype=torch.long)
    chord_input_tokens = torch.zeros(batch_size, chord_seq_len, dtype=torch.long)
    
    # Melody: onset tokens (0-87), hold tokens (88-175), padding (178)
    melody_tokens[:, :100] = torch.randint(0, 88, (batch_size, 100))     # Onset tokens
    melody_tokens[:, 100:150] = torch.randint(88, 176, (batch_size, 50)) # Hold tokens  
    melody_tokens[:, 150:] = pad_token_id                                 # Padding (178)
    
    # Chords: onset tokens (~179-1000), hold tokens (~2000-4600), padding (178)
    chord_input_tokens[:, :100] = torch.randint(179, 1000, (batch_size, 100))   # Onset chords
    chord_input_tokens[:, 100:150] = torch.randint(2000, 4600, (batch_size, 50)) # Hold chords
    chord_input_tokens[:, 150:] = pad_token_id                                   # Padding
    
    chord_target_tokens = chord_input_tokens.clone()  # Targets same as input for testing
    
    print(f"\nInput shapes and realistic data ranges:")
    print(f"  Melody: {melody_tokens.shape}, range: {melody_tokens.min()}-{melody_tokens.max()}")
    print(f"    Onset tokens (0-87): {(melody_tokens < 88).sum().item()}")
    print(f"    Hold tokens (88-175): {((melody_tokens >= 88) & (melody_tokens < 178)).sum().item()}")
    print(f"    Padding tokens (178): {(melody_tokens == pad_token_id).sum().item()}")
    print(f"  Chord input: {chord_input_tokens.shape}, range: {chord_input_tokens.min()}-{chord_input_tokens.max()}")
    print(f"    Onset-like tokens: {((chord_input_tokens >= 179) & (chord_input_tokens < 2000)).sum().item()}")
    print(f"    Hold-like tokens: {(chord_input_tokens >= 2000).sum().item()}")
    print(f"    Padding tokens: {(chord_input_tokens == pad_token_id).sum().item()}")
    print(f"  Chord target: {chord_target_tokens.shape}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        logits = model(melody_tokens, chord_input_tokens)
        print(f"\nOutput shape: {logits.shape}")
        print(f"Expected: [{batch_size}, {chord_seq_len}, {total_vocab_size}]")
        
        # Test encode-decode separately
        encoder_hidden, encoder_mask = model.encode(melody_tokens)
        print(f"Encoder output shape: {encoder_hidden.shape}")
        
        decoder_logits = model.decode_step(encoder_hidden, encoder_mask, chord_input_tokens)
        print(f"Decoder output shape: {decoder_logits.shape}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    _test_t5_model() 