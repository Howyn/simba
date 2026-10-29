"""Channel-wise topology refinement for Simba's spatial block entrance.

Implements the CTR-GC mechanism from Chen et al., ICCV 2021:
https://github.com/Uason-Chen/CTR-GCN/blob/main/model/ctrgcn.py
The upstream project is licensed CC BY-NC 4.0:
https://github.com/Uason-Chen/CTR-GCN/blob/main/LICENSE

Adaptation: expose the complete spatial unit as a drop-in replacement for
Simba's initial Shift_gcn, while keeping the rest of its U-shaped block.
"""

import torch
from torch import nn


class ChannelTopology(nn.Module):
    """One adjacency subset with sample- and channel-dependent relations."""

    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        rel_channels = 8 if in_channels in (3, 9) else max(1, in_channels // rel_reduction)
        self.query = nn.Conv2d(in_channels, rel_channels, 1)
        self.key = nn.Conv2d(in_channels, rel_channels, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = nn.Conv2d(rel_channels, out_channels, 1)

        for layer in (self.query, self.key, self.value, self.refine):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out")
            nn.init.zeros_(layer.bias)

    def forward(self, x, adjacency, alpha):
        # Input: N,C,T,V. The learned relations use the time-averaged features.
        query = self.query(x).mean(dim=2)
        key = self.key(x).mean(dim=2)
        relation = torch.tanh(query.unsqueeze(-1) - key.unsqueeze(-2))
        relation = self.refine(relation) * alpha + adjacency[None, None, :, :]
        return torch.einsum("ncuv,nctv->nctu", relation, self.value(x))


class CTRGCUnit(nn.Module):
    """CTR-GC spatial unit with learned topology, residual, BN and ReLU."""

    def __init__(self, in_channels, out_channels, adjacency):
        super().__init__()
        graph = torch.as_tensor(adjacency, dtype=torch.float32)
        if graph.ndim != 3 or graph.shape[-1] != graph.shape[-2]:
            raise ValueError("adjacency must have shape (subsets, joints, joints)")
        self.topology = nn.Parameter(graph.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.branches = nn.ModuleList(
            ChannelTopology(in_channels, out_channels) for _ in range(graph.shape[0])
        )
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn.weight, 1e-6)
        nn.init.zeros_(self.bn.bias)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            conv = nn.Conv2d(in_channels, out_channels, 1)
            nn.init.kaiming_normal_(conv.weight, mode="fan_out")
            nn.init.zeros_(conv.bias)
            self.residual = nn.Sequential(conv, nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim != 4 or x.size(-1) != self.topology.size(-1):
            raise ValueError("input must be N,C,T,V with V matching the adjacency")
        result = sum(branch(x, self.topology[i], self.alpha)
                     for i, branch in enumerate(self.branches))
        return self.relu(self.bn(result) + self.residual(x))
