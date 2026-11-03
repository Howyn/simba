from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layernorm import RMSNorm
import torch.nn as nn


class MambaBlock(nn.Module):
    def __init__(self, embed_dim, dropout_level=0):
        super().__init__()
        self.mamba = Mamba(d_model=embed_dim, d_state=16, d_conv=4, expand=2)
        self.norm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_level)

    def forward(self, x):
        return self.dropout(self.norm(self.mamba(x) + x))
