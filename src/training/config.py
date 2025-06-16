"""
Training configuration for OnlineTransformer model
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    # Data parameters
    data_dir: str
    batch_size: int = 256
    num_workers: int = 4
    
    # Model parameters
    embed_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    feedforward_dim: int = field(init=False) # Will be calculated from embed_dim
    max_sequence_length: int = 512
    dropout: float = 0.1
    
    # Vocabulary sizes (will be loaded from data)
    vocab_size: Optional[int] = None
    melody_vocab_size: Optional[int] = None
    chord_vocab_size: Optional[int] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    total_steps: Optional[int] = 500000
    warmup_steps: int = 2000
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 100
    
    # Weights & Biases parameters
    wandb_project: str = "martydepth"
    wandb_entity: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    
    def __post_init__(self):
        """Calculate dependent fields after initialization."""
        self.feedforward_dim = 4 * self.embed_dim
    
    def to_dict(self):
        """Convert config to dictionary for wandb logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')} 