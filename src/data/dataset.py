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
from src.config.tokenization_config import PAD_TOKEN

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
            mode: 'online', 'offline', 'contrastive', or 'discriminator'.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.mode = mode
        
        # Point to the directory for the correct split
        self.split_dir = self.data_dir / self.split
        
        # Discover all sequence files in the directory
        self.sequence_files = sorted(list(self.split_dir.glob("sequence_*.pkl")))
        self.num_sequences = len(self.sequence_files)

        # Load tokenizer info
        self.tokenizer_info = self._load_tokenizer_info()
        self.pad_token_id = self.tokenizer_info['pad_token_id']
        
        # Set vocabulary sizes
        self.melody_vocab_size = self.tokenizer_info['melody_vocab_size']
        self.chord_vocab_size = self.tokenizer_info['chord_vocab_size']
        self.vocab_size = self.tokenizer_info['total_vocab_size']  # Expose total vocab size
        
    def _load_tokenizer_info(self) -> Dict:
        """Load tokenizer information"""
        tokenizer_file = self.split_dir / 'tokenizer_info.json'
        print(f"Loading tokenizer info from {tokenizer_file}")
        with open(tokenizer_file, 'r') as f:
            info = json.load(f)
        print(f"Loaded tokenizer info with {info['total_vocab_size']} total tokens")
        return info
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return self.num_sequences
    
    def _interleave_sequences(self, melody_tokens: np.ndarray, chord_tokens: np.ndarray) -> np.ndarray:
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

        # Create padding mask: True for padding tokens, False otherwise
        padding_mask = (input_tokens == self.pad_token_id)
        
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
        
        # Use PAD token as start token for teacher forcing (T5 standard).
        # The model receives the full melody, PAD token + chord sequence up to the
        # second-to-last token, and predicts the chord sequence from the
        # first token onwards.
        chord_input = torch.cat([torch.tensor([self.pad_token_id]), chord_tokens[:-1]])

        return {
            'melody_tokens': melody_tokens,           # [T] - full melody sequence
            'chord_input': chord_input,               # [T] - shifted chord sequence for decoder input
            'chord_target': chord_tokens,             # [T] - original chord sequence for target
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

    def _get_discriminator_format(self, sequence: FrameSequence) -> Dict[str, torch.Tensor]:
        """Get format for training the discriminative reward model."""
        interleaved_tokens = self._interleave_sequences(
            sequence.melody_tokens,
            sequence.chord_tokens
        )
        return {
            'interleaved_tokens': torch.tensor(interleaved_tokens, dtype=torch.long),
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

        # The pickle file contains a FrameSequence object that may be longer
        # than the desired sequence_length. Truncate it if necessary.
        if len(sequence.melody_tokens) > self.sequence_length:
            sequence.melody_tokens = sequence.melody_tokens[:self.sequence_length]
            sequence.chord_tokens = sequence.chord_tokens[:self.sequence_length]
        
        # The pickle file contains a FrameSequence object
        # that already has the correct sequence length.
        
        if self.mode == 'online':
            return self._get_online_format(sequence)
        elif self.mode == 'offline':
            return self._get_offline_format(sequence)
        elif self.mode == 'contrastive':
            return self._get_contrastive_format(sequence)
        elif self.mode == 'discriminator':
            return self._get_discriminator_format(sequence)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

def create_dataloader(data_dir: Path,
                     split: str,
                     batch_size: int,
                     shuffle: bool,
                     num_workers: int,
                     sequence_length: int,
                     mode: str) -> Tuple[torch.utils.data.DataLoader, Dict]:
    """
    Create a DataLoader for the frame sequences
    
    Args:
        data_dir: Path to the data directory
        split: Which split to load ('train', 'valid', or 'test')
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle the sequences
        num_workers: Number of worker processes for loading
        sequence_length: Length of each sequence
        mode: Mode for data loading, must be one of: 'online', 'offline', 'contrastive', or 'discriminator'
        
    Returns:
        A tuple containing the DataLoader and the tokenizer_info dictionary.
    """
    print(f"\n Creating {split} dataloader:")
    print(f"  Data directory: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Mode: {mode}")
    
    # Validate mode
    valid_modes = ['online', 'offline', 'contrastive', 'discriminator']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")
    
    dataset = FrameDataset(
        data_dir=data_dir,
        split=split,
        sequence_length=sequence_length,
        mode=mode
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader, dataset.tokenizer_info

def main():
    """Test the dataset and dataloader"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "interim"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configuration
    test_config = {
        'batch_size': 4,
        'num_workers': 0,  # Use 0 for local testing to avoid memory issues
        'sequence_length': 256,
        'shuffle': True
    }
    
    # Test all modes
    for mode in ['online', 'offline', 'contrastive', 'discriminator']:
        print(f"\nTesting {mode} mode...")
        dataloader, tokenizer_info = create_dataloader(
            data_dir=data_dir,
            split='valid',  # Use smaller validation split for local testing
            batch_size=test_config['batch_size'],
            shuffle=test_config['shuffle'],
            num_workers=test_config['num_workers'],
            sequence_length=test_config['sequence_length'],
            mode=mode
        )
        
        # Print dataset info
        dataset = dataloader.dataset
        print(f"Split: {dataset.split}")
        print(f"Number of sequences: {len(dataset)}")
        print(f"Sequence length: {dataset.sequence_length}")
        print(f"Vocabulary sizes:")
        print(f"  Melody tokens: {tokenizer_info['melody_vocab_size']}")
        print(f"  Chord tokens: {tokenizer_info['chord_vocab_size']}")
        print(f"  Total tokens: {tokenizer_info['total_vocab_size']}")
        
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
        elif mode == 'contrastive':
            print(f"Melody tokens shape: {batch['melody_tokens'].shape}")
            print(f"Chord tokens shape: {batch['chord_tokens'].shape}")
        else:  # discriminator
            print(f"Interleaved tokens shape: {batch['interleaved_tokens'].shape}")

        print(f"Device: {next(iter(batch.values())).device if isinstance(next(iter(batch.values())), torch.Tensor) else 'cpu'}")

if __name__ == "__main__":
    main() 