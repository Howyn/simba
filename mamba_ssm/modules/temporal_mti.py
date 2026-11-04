"""MTI adaptation of TSkel-Mamba (arXiv:2512.11503, 2025).

Use the parallel residual sum in Eq. (4), rather than the sequential updates
in Algorithm 1. Cycle offsets are centered: c % K - K // 2. This extends the
paper's K=3 example to K=1 and K=5; Eq. (3) prints a constant -1 for all K.
Out-of-range time positions are zero, never wrapped to the other sequence end.
Only temporal/channel mixing is performed; joints are processed independently.
"""

import torch
from torch import nn


class TemporalCycleFC(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        if channels < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('channels must be positive and kernel_size positive and odd')
        self.channels = channels
        self.kernel_size = kernel_size
        self.proj = nn.Linear(channels, channels)
        offsets = torch.arange(channels) % kernel_size - kernel_size // 2
        self.register_buffer('offsets', offsets, persistent=False)

    def forward(self, x):
        # x: (B, C, T, V). Gather x[c, t + offset(c), v], then mix C.
        if x.ndim != 4 or x.shape[1] != self.channels or x.shape[2] < 1:
            raise ValueError(f'expected (B, {self.channels}, T>=1, V) input')
        batch, channels, length, joints = x.shape
        values = x.permute(0, 3, 2, 1)  # B, V, T, C
        times = torch.arange(length, device=x.device)[:, None] + self.offsets[None, :]
        valid = (times >= 0) & (times < length)
        index = times.clamp(0, length - 1)[None, None].expand(batch, joints, -1, -1)
        shifted = values.gather(2, index) * valid[None, None].to(dtype=x.dtype)
        return self.proj(shifted).permute(0, 3, 2, 1).contiguous()


class MultiScaleTemporalInteraction(nn.Module):
    def __init__(self, channels, kernel_sizes=(1, 3, 5)):
        super().__init__()
        if not kernel_sizes or len(set(kernel_sizes)) != len(kernel_sizes):
            raise ValueError('kernel_sizes must be nonempty and unique')
        self.branches = nn.ModuleList(
            TemporalCycleFC(channels, kernel_size) for kernel_size in kernel_sizes
        )

    def forward(self, x):
        result = x
        for branch in self.branches:
            result = result + branch(x)
        return result
