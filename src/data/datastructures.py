#!/usr/bin/env python3
"""
Shared data structures for the project.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np

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