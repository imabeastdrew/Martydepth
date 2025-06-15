"""
Training configuration for OnlineTransformer model
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Data parameters
    data_dir: str
    batch_size: int = 32
    num_workers: int = 4
    
    # Model parameters (autoregressive, paper spec)
    vocab_size: Optional[int] = None  # Will be loaded from data
    embed_dim: int = 512    # Paper spec
    num_layers: int = 8     # Paper spec  
    num_heads: int = 8      # Changed to be compatible with embed_dim
    feedforward_dim: int = 2048  # 4 * embed_dim
    max_sequence_length: int = 512  # For interleaved sequences
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 100
    
    # Weights & Biases parameters
    wandb_project: str = "martydepth"
    wandb_entity: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    
    def to_dict(self):
        """Convert config to dictionary for wandb logging"""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "warmup_steps": self.warmup_steps,
            "gradient_clip_val": self.gradient_clip_val,
            "max_sequence_length": self.max_sequence_length,
            "vocab_size": self.vocab_size
        } 