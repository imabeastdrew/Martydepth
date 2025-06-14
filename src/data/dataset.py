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

@dataclass
class FrameSequence:
    """Container for a sequence of chord frames"""
    melody_tokens: np.ndarray  # Shape: (sequence_length,)
    chord_tokens: np.ndarray   # Shape: (sequence_length,)
    key_context: np.ndarray    # Shape: (sequence_length,)
    meter_context: np.ndarray  # Shape: (sequence_length,)
    song_id: str
    start_frame: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'melody_tokens': self.melody_tokens,
            'chord_tokens': self.chord_tokens,
            'key_context': self.key_context,
            'meter_context': self.meter_context,
            'song_id': self.song_id,
            'start_frame': self.start_frame
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FrameSequence':
        """Create from dictionary after deserialization"""
        return cls(
            melody_tokens=data['melody_tokens'],
            chord_tokens=data['chord_tokens'],
            key_context=data['key_context'],
            meter_context=data['meter_context'],
            song_id=data['song_id'],
            start_frame=data['start_frame']
        )

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
            mode: 'online' for causal training or 'offline' for full context
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.mode = mode
        
        # Load sequences
        self.sequences = self._load_sequences()
        
        # Load tokenizer info
        self.tokenizer_info = self._load_tokenizer_info()
        
        # Set vocabulary sizes
        self.melody_vocab_size = self.tokenizer_info['melody_vocab_size']
        self.chord_vocab_size = self.tokenizer_info['chord_vocab_size']
        self.total_vocab_size = self.tokenizer_info['total_vocab_size']
        
    def _load_sequences(self) -> List[FrameSequence]:
        """Load sequences from pickle file"""
        with open(self.data_dir / self.split / 'frame_sequences.pkl', 'rb') as f:
            sequence_dicts = pickle.load(f)
        return [FrameSequence.from_dict(d) for d in sequence_dicts]
    
    def _load_tokenizer_info(self) -> Dict:
        """Load tokenizer information"""
        with open(self.data_dir / self.split / 'tokenizer_info.json', 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset"""
        return len(self.sequences)
    
    def _interleave_sequences(self, melody_tokens: np.ndarray, chord_tokens: np.ndarray) -> np.ndarray:
        """Interleave melody and chord tokens"""
        # Create interleaved sequence: [melody_0, chord_0, melody_1, chord_1, ...]
        interleaved = np.empty(len(melody_tokens) * 2, dtype=melody_tokens.dtype)
        interleaved[0::2] = melody_tokens  # Even indices: melody tokens
        interleaved[1::2] = chord_tokens   # Odd indices: chord tokens
        return interleaved
    
    def _get_online_format(self, sequence: FrameSequence) -> Dict[str, torch.Tensor]:
        """Get causal training format with interleaved sequences"""
        # Convert numpy arrays to tensors (no device placement)
        melody_tokens = torch.tensor(sequence.melody_tokens, dtype=torch.long)
        chord_tokens = torch.tensor(sequence.chord_tokens, dtype=torch.long)
        
        # Create interleaved sequence
        input_tokens = self._interleave_sequences(
            sequence.melody_tokens[:-1],  # [0:T-1]
            sequence.chord_tokens[:-1]    # [0:T-1]
        )
        input_tokens = torch.tensor(input_tokens, dtype=torch.long)
        
        # Target is next chord token
        target_chord = torch.tensor(sequence.chord_tokens[1:], dtype=torch.long)
        
        return {
            'input_tokens': input_tokens,  # [2*(T-1)]
            'target_chord': target_chord,  # [T-1]
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence by index
        
        Returns:
            Dictionary containing:
            For online mode:
            - input_tokens: Interleaved melody and chord tokens [2*(T-1)]
            - target_chord: Target chord tokens [T-1]
            For offline mode:
            - melody_tokens: Full melody sequence [T]
            - chord_input: Causal chord input [T-1]
            - chord_target: Next chord targets [T-1]
            Common:
            - song_id: String identifier for the song
            - start_frame: Integer indicating the start frame
        """
        sequence = self.sequences[idx]
        
        if self.mode == 'online':
            return self._get_online_format(sequence)
        else:  # offline mode
            return self._get_offline_format(sequence)

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
        mode: 'online' for causal training or 'offline' for full context
        
    Returns:
        DataLoader for the specified split
    """
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
        pin_memory=True  # Enable pin_memory for faster GPU transfer
    )

def main():
    """Test the dataset and dataloader"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "interim"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test both modes
    for mode in ['online', 'offline']:
        print(f"\nTesting {mode} mode...")
        dataloader = create_dataloader(
            data_dir=data_dir,
            split='train',
            batch_size=4,
            shuffle=True,
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
        print(f"  Total tokens: {dataset.total_vocab_size}")
        
        # Test loading a batch and moving to device
        print("\nTesting batch loading...")
        batch = next(iter(dataloader))
        # Move tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        if mode == 'online':
            print(f"Input tokens shape: {batch['input_tokens'].shape}")
            print(f"Target chord shape: {batch['target_chord'].shape}")
        else:  # offline mode
            print(f"Melody tokens shape: {batch['melody_tokens'].shape}")
            print(f"Chord input shape: {batch['chord_input'].shape}")
            print(f"Chord target shape: {batch['chord_target'].shape}")
        print(f"Device: {next(iter(batch.values())).device if isinstance(next(iter(batch.values())), torch.Tensor) else 'cpu'}")

if __name__ == "__main__":
    main() 