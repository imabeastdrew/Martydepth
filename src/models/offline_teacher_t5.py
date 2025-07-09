import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import T5Config, T5ForConditionalGeneration
from typing import Optional

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
                 pad_token_id: int = 178):
        super().__init__()
        
        self.melody_vocab_size = melody_vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        
        # Create T5 config
        self.config = T5Config(
            vocab_size=max(melody_vocab_size, chord_vocab_size),  # Use larger vocab
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
        
        # Create the T5 model
        self.t5_model = T5ForConditionalGeneration(config=self.config)
        
        # Separate input/output embeddings for melody and chords
        # We'll map our vocabulary to T5's vocabulary space
        self.melody_input_projection = nn.Linear(melody_vocab_size, embed_dim, bias=False)
        self.chord_input_projection = nn.Linear(chord_vocab_size, embed_dim, bias=False)
        self.chord_output_projection = nn.Linear(embed_dim, chord_vocab_size, bias=False)
        
        # Initialize T5 with proper scaling
        self._init_t5_weights()
        
    def _init_t5_weights(self):
        """Initialize T5 weights with proper scaling"""
        # T5 uses specific initialization
        self.t5_model.apply(self._init_weights)
        
        # Initialize our projection layers
        std = 1.0 / math.sqrt(self.embed_dim)
        nn.init.normal_(self.melody_input_projection.weight, mean=0, std=std)
        nn.init.normal_(self.chord_input_projection.weight, mean=0, std=std)
        nn.init.normal_(self.chord_output_projection.weight, mean=0, std=std)
    
    def _init_weights(self, module):
        """Initialize T5 module weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor * 1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def create_one_hot_embeddings(self, tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Convert token indices to one-hot embeddings"""
        batch_size, seq_len = tokens.shape
        # Create one-hot vectors
        one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=tokens.device, dtype=torch.float32)
        one_hot.scatter_(2, tokens.unsqueeze(-1), 1.0)
        return one_hot
    
    def forward(self,
                melody_tokens: torch.Tensor,
                chord_tokens: torch.Tensor,
                melody_mask: Optional[torch.Tensor] = None,
                chord_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through T5 model
        
        Args:
            melody_tokens: [batch, melody_seq_len] - encoder input
            chord_tokens: [batch, chord_seq_len] - decoder input (shifted)
            melody_mask: [batch, melody_seq_len] - encoder attention mask
            chord_mask: [batch, chord_seq_len] - decoder attention mask
        
        Returns:
            logits: [batch, chord_seq_len, chord_vocab_size]
        """
        batch_size = melody_tokens.shape[0]
        
        # Create attention masks if not provided
        if melody_mask is None:
            melody_mask = (melody_tokens != self.pad_token_id).long()
        if chord_mask is None:
            chord_mask = (chord_tokens != self.pad_token_id).long()
        
        # Convert tokens to one-hot embeddings then project to T5 embedding space
        melody_one_hot = self.create_one_hot_embeddings(melody_tokens, self.melody_vocab_size)
        chord_one_hot = self.create_one_hot_embeddings(chord_tokens, self.chord_vocab_size)
        
        # Project to T5 embedding dimension
        melody_embeds = self.melody_input_projection(melody_one_hot)  # [batch, seq, embed_dim]
        chord_embeds = self.chord_input_projection(chord_one_hot)     # [batch, seq, embed_dim]
        
        # Run through T5
        outputs = self.t5_model(
            inputs_embeds=melody_embeds,
            attention_mask=melody_mask,
            decoder_inputs_embeds=chord_embeds,
            decoder_attention_mask=chord_mask,
            return_dict=True
        )
        
        # T5ForConditionalGeneration already provides logits, but they're for the T5 vocab
        # We need to get the decoder hidden states and project to our chord vocab
        decoder_outputs = self.t5_model.decoder(
            inputs_embeds=chord_embeds,
            attention_mask=chord_mask,
            encoder_hidden_states=outputs.encoder_last_hidden_state,
            encoder_attention_mask=melody_mask,
            return_dict=True
        )
        
        # Project T5 decoder output back to chord vocabulary
        logits = self.chord_output_projection(decoder_outputs.last_hidden_state)
        
        return logits
    
    def encode(self, melody_tokens: torch.Tensor, melody_mask: Optional[torch.Tensor] = None):
        """Encode melody using T5 encoder only"""
        if melody_mask is None:
            melody_mask = (melody_tokens != self.pad_token_id).long()
        
        melody_one_hot = self.create_one_hot_embeddings(melody_tokens, self.melody_vocab_size)
        melody_embeds = self.melody_input_projection(melody_one_hot)
        
        encoder_outputs = self.t5_model.encoder(
            inputs_embeds=melody_embeds,
            attention_mask=melody_mask,
            return_dict=True
        )
        
        return encoder_outputs.last_hidden_state, melody_mask
    
    def decode_step(self, 
                    encoder_hidden: torch.Tensor,
                    encoder_mask: torch.Tensor,
                    chord_tokens: torch.Tensor) -> torch.Tensor:
        """Single decoder step for generation"""
        chord_one_hot = self.create_one_hot_embeddings(chord_tokens, self.chord_vocab_size)
        chord_embeds = self.chord_input_projection(chord_one_hot)
        
        decoder_mask = (chord_tokens != self.pad_token_id).long()
        
        # Create encoder outputs structure for T5
        encoder_outputs = type('EncoderOutputs', (), {
            'last_hidden_state': encoder_hidden,
            'hidden_states': None,
            'attentions': None
        })()
        
        outputs = self.t5_model.decoder(
            inputs_embeds=chord_embeds,
            attention_mask=decoder_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            return_dict=True
        )
        
        logits = self.chord_output_projection(outputs.last_hidden_state)
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
    # Note: vocab_size must be > max_token_id, so pad_token_id=178 requires vocab_size >= 179
    melody_vocab_size = 179  # Must include PAD token 178 (0-178 = 179 tokens)
    chord_vocab_size = 4601  # Must include PAD token 178 (178 + 4600 chord tokens = 4601)
    pad_token_id = 178       # Actual PAD token from dataset
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
        pad_token_id=pad_token_id
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
        print(f"Expected: [{batch_size}, {chord_seq_len}, {chord_vocab_size}]")
        
        # Test encode-decode separately
        encoder_hidden, encoder_mask = model.encode(melody_tokens)
        print(f"Encoder output shape: {encoder_hidden.shape}")
        
        decoder_logits = model.decode_step(encoder_hidden, encoder_mask, chord_input_tokens)
        print(f"Decoder output shape: {decoder_logits.shape}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    _test_t5_model() 