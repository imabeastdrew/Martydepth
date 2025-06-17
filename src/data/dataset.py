#!/usr/bin/env python3
"""
PyTorch Dataset for loading and processing frame sequences
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass

from src.data.datastructures import FrameSequence
from src.config.tokenization_config import SILENCE_TOKEN

class FrameDataset(Dataset):
    """PyTorch Dataset for loading frame sequences"""
    
    def __init__(self, 
                 data_dir: Path,
                 split: str = 'train',
                 sequence_length: int = 256,
                 mode: str = 'online'):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to the data directory containing train/valid/test splits
            split: Which split to load ('train', 'valid', or 'test')
            sequence_length: Length of each sequence
            mode: 'online' for causal training, 'offline' for full context, or 'contrastive' for reward model training.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        if mode not in ['online', 'offline', 'contrastive']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'online', 'offline', 'contrastive'.")
        self.mode = mode
        
        # Point to the directory for the correct split
        self.split_dir = self.data_dir / self.split
        
        # Discover all sequence files in the directory
        self.sequence_files = sorted(list(self.split_dir.glob("sequence_*.pkl")))
        if not self.sequence_files:
            raise FileNotFoundError(
                f"No sequence .pkl files found in {self.split_dir}. "
                f"Please run preprocessing to generate the data for the '{self.split}' split."
            )
        self.num_sequences = len(self.sequence_files)

        # Load tokenizer info
        self.tokenizer_info = self._load_tokenizer_info()
        
        # Set vocabulary sizes
        self.melody_vocab_size = self.tokenizer_info['melody_vocab_size']
        self.chord_vocab_size = self.tokenizer_info['chord_vocab_size']
        self.vocab_size = self.tokenizer_info['total_vocab_size']  # Expose total vocab size
        
    def _load_tokenizer_info(self) -> Dict:
        """Load tokenizer information"""
        tokenizer_file = self.split_dir / 'tokenizer_info.json'
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer info not found: {tokenizer_file}\n"
                f"Please run preprocessing to generate tokenizer info first."
            )
        
        print(f"Loading tokenizer info from {tokenizer_file}")
        with open(tokenizer_file, 'r') as f:
            info = json.load(f)
        print(f"Loaded tokenizer info with {info['total_vocab_size']} total tokens")
        return info
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return self.num_sequences
    
    def _interleave_sequences(self, melody_tokens: np.ndarray, chord_tokens: np.ndarray) -> np.ndarray:
        """Interleave melody and chord tokens in paper format: [chord_1, melody_1, chord_2, melody_2, ...]"""
        # Create interleaved sequence: [chord_0, melody_0, chord_1, melody_1, ...]
        interleaved = np.empty(len(melody_tokens) * 2, dtype=melody_tokens.dtype)
        interleaved[1::2] = melody_tokens  # Odd indices: melody tokens
        interleaved[0::2] = chord_tokens   # Even indices: chord tokens
        return interleaved
    
    def _get_online_format(self, sequence: FrameSequence) -> Dict[str, torch.Tensor]:
        """Standard autoregressive format"""
        # Create full interleaved sequence: [chord_0, melody_0, chord_1, melody_1, ...]
        full_interleaved = self._interleave_sequences(
            sequence.melody_tokens,
            sequence.chord_tokens
        )
        
        # Standard autoregressive split
        input_tokens = torch.tensor(full_interleaved[:-1], dtype=torch.long)
        target_tokens = torch.tensor(full_interleaved[1:], dtype=torch.long)

        # Create padding mask: True for padding tokens (silence), False otherwise
        # The online model uses a single, shared silence token for melody and chords.
        padding_mask = (input_tokens == SILENCE_TOKEN)
        
        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'padding_mask': padding_mask,
            'song_id': sequence.song_id,
            'start_frame': sequence.start_frame
        }
    
    def _get_offline_format(self, sequence: FrameSequence) -> Dict[str, torch.Tensor]:
        """Get full context training format for offline teacher model"""
        # Convert numpy arrays to tensors (no device placement)
        melody_tokens = torch.tensor(sequence.melody_tokens, dtype=torch.long)
        chord_tokens = torch.tensor(sequence.chord_tokens, dtype=torch.long)
        
        return {
            'melody_tokens': melody_tokens,           # [T] - full melody sequence
            'chord_input': chord_tokens[:-1],         # [T-1] - causal chord input
            'chord_target': chord_tokens[1:],         # [T-1] - next chord targets
            'song_id': sequence.song_id,
            'start_frame': sequence.start_frame
        }
    
    def _get_contrastive_format(self, sequence: FrameSequence) -> Dict[str, torch.Tensor]:
        """Get format for training the contrastive reward model."""
        melody_tokens = torch.tensor(sequence.melody_tokens, dtype=torch.long)
        chord_tokens = torch.tensor(sequence.chord_tokens, dtype=torch.long)
        
        # For the contrastive loss, we just need the melody and chord sequences.
        return {
            'melody_tokens': melody_tokens,
            'chord_tokens': chord_tokens,
            'song_id': sequence.song_id,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence by index by loading its individual pickle file.
        """
        # Get the file path for the requested index
        sequence_path = self.sequence_files[idx]
        
        # Load the single FrameSequence object from its pickle file
        with open(sequence_path, 'rb') as f:
            sequence = pickle.load(f)
        
        if self.mode == 'online':
            return self._get_online_format(sequence)
        elif self.mode == 'offline':
            return self._get_offline_format(sequence)
        else:  # contrastive mode
            return self._get_contrastive_format(sequence)

def create_dataloader(data_dir: Path,
                     split: str = 'train',
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     sequence_length: int = 256,
                     mode: str = 'online') -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the frame sequences
    
    Args:
        data_dir: Path to the data directory
        split: Which split to load ('train', 'valid', or 'test')
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle the sequences
        num_workers: Number of worker processes for loading
        sequence_length: Length of each sequence
        mode: 'online' for causal training, 'offline' for full context, or 'contrastive' for reward model training.
        
    Returns:
        DataLoader for the specified split
    """
    print(f"\n Creating {split} dataloader:")
    print(f"  Data directory: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Mode: {mode}")
    
    dataset = FrameDataset(
        data_dir=data_dir,
        split=split,
        sequence_length=sequence_length,
        mode=mode
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def main():
    """Test the dataset and dataloader"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "interim"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test all modes
    for mode in ['online', 'offline', 'contrastive']:
        print(f"\nTesting {mode} mode...")
        dataloader = create_dataloader(
            data_dir=data_dir,
            split='valid', # Use smaller validation split for local testing
            batch_size=4,
            shuffle=True,
            num_workers=0, # Use 0 for local testing to avoid memory issues
            mode=mode
        )
        
        # Print dataset info
        dataset = dataloader.dataset
        print(f"Split: {dataset.split}")
        print(f"Number of sequences: {len(dataset)}")
        print(f"Sequence length: {dataset.sequence_length}")
        print(f"Vocabulary sizes:")
        print(f"  Melody tokens: {dataset.melody_vocab_size}")
        print(f"  Chord tokens: {dataset.chord_vocab_size}")
        print(f"  Total tokens: {dataset.vocab_size}")
        
        # Test loading a batch and moving to device
        print("\nTesting batch loading...")
        batch = next(iter(dataloader))
        # Move tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        if mode == 'online':
            print(f"Input tokens shape: {batch['input_tokens'].shape}")
            print(f"Target tokens shape: {batch['target_tokens'].shape}")
        elif mode == 'offline':  # offline mode
            print(f"Melody tokens shape: {batch['melody_tokens'].shape}")
            print(f"Chord input shape: {batch['chord_input'].shape}")
            print(f"Chord target shape: {batch['chord_target'].shape}")
        else: # contrastive
            print(f"Melody tokens shape: {batch['melody_tokens'].shape}")
            print(f"Chord tokens shape: {batch['chord_tokens'].shape}")

        print(f"Device: {next(iter(batch.values())).device if isinstance(next(iter(batch.values())), torch.Tensor) else 'cpu'}")

if __name__ == "__main__":
    main() 