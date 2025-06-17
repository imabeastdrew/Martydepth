from torch.optim.lr_scheduler import LambdaLR
import torch

def get_warmup_schedule(optimizer: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    """Create warmup schedule"""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))  # Linear warmup
        return 1.0  # Full learning rate
    return LambdaLR(optimizer, lr_lambda) 