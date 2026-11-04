from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
import torch
import torch.nn as nn


class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout_level=0, use_mti=False, num_point=25):
        super().__init__()
        self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_level)
        self.use_mti = use_mti
        self.num_point = num_point
        if use_mti:
            if num_point < 1 or embed_dim % num_point:
                raise ValueError('embed_dim must be divisible by num_point for MTI')
            from mamba_ssm.modules.temporal_mti import MultiScaleTemporalInteraction
            # Extra initialization must not change the baseline weights in later
            # layers, nor the training/data RNG stream for the paired run.
            with torch.random.fork_rng(devices=[]):
                self.mti = MultiScaleTemporalInteraction(embed_dim // num_point)

    def forward(self, x):
        if self.use_mti:
            batch, length, width = x.shape
            features = x.reshape(batch, length, self.num_point, width // self.num_point)
            features = self.mti(features.permute(0, 3, 1, 2))
            enhanced = features.permute(0, 2, 3, 1).reshape(batch, length, width)
            return self.dropout(self.norm(self.mamba(enhanced) + x))
        return self.dropout(self.norm(self.mamba(x) + x))
