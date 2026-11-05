"""Hyper-GCN (ICCV 2025) spatial / DeGCN (TIP 2024) temporal adapters.

Sources and adaptation details: COMBINATION1_EXPERIMENT.md. These are paper
submodules, not reproductions of either complete network. Geometry and sampling
arithmetic run in FP32 under AMP; ordinary projections still use autocast.
"""
import torch
from torch import nn


def virtual_graph(adjacency, virtual_nodes=3, heads=8):
    """Upstream get_virtual_graph_ensemble: undirected physical + virtual edges."""
    a = torch.as_tensor(adjacency, dtype=torch.float32)
    if a.ndim != 3 or a.shape[-1] != a.shape[-2]:
        raise ValueError('expected [subsets, joints, joints] adjacency')
    vertices = a.shape[-1]
    physical = (a.sum(0) != 0).float()
    physical.fill_diagonal_(1)
    graph = torch.zeros(vertices + virtual_nodes, vertices + virtual_nodes)
    graph[:vertices, :vertices] = physical
    graph[:vertices, vertices:] = 1
    graph[vertices:, :vertices] = 1
    for i in range(vertices, vertices + virtual_nodes):
        graph[i, i] = 1
    return graph.unsqueeze(0).repeat(heads, 1, 1)


class HyperGraphSpatial(nn.Module):
    """HYPERGC: virtual joints + nearest-neighbour incidence + hypergraph mixing."""
    def __init__(self, channels, adjacency, virtual_nodes=3, heads=8, neighbours=9):
        super().__init__()
        if channels % heads or channels // heads < 4 or virtual_nodes < 1:
            raise ValueError('Hyper-GCN needs divisible channels, >=4 per head, virtual nodes')
        self.vertices = int(adjacency.shape[-1])
        self.virtual_nodes, self.heads, self.neighbours = virtual_nodes, heads, neighbours
        if not 1 <= neighbours <= self.vertices + virtual_nodes:
            raise ValueError('invalid hyperedge neighbourhood size')
        self.hidden = channels // heads // 4
        self.to_V = nn.Conv1d(channels, heads * self.hidden, 1, groups=heads)
        self.to_W = nn.Sequential(
            nn.Conv1d(channels, heads * self.hidden, 1, groups=heads),
            nn.LeakyReLU(), nn.Conv1d(heads * self.hidden, heads, 1), nn.Tanh())
        self.hyper_joint = nn.Parameter(torch.randn(1, channels).repeat(virtual_nodes, 1))
        self.alpha = nn.Parameter(torch.ones(1))
        self.conv_d = nn.Conv2d(channels, channels, 1, groups=heads)
        self.register_buffer('PA', virtual_graph(adjacency, virtual_nodes, heads))
        self.edge_importance = nn.Parameter(torch.ones_like(self.PA))
        self.bn = nn.BatchNorm2d(channels)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.constant_(self.bn.weight, 1e-6)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        n, c, t, v = x.shape
        if v != self.vertices:
            raise ValueError('Hyper-GCN input joint count differs from adjacency')
        # Virtual nodes are internal only: the caller always receives V=25.
        virtual = self.hyper_joint.T[None, :, None, :].expand(n, -1, t, -1)
        augmented = torch.cat((x, virtual.to(x.dtype)), dim=-1)
        total = augmented.shape[-1]
        descriptor = augmented.mean(2)
        embedding = self.to_V(descriptor).reshape(n, self.heads, self.hidden, total)
        weights = self.to_W(descriptor)
        with torch.autocast(device_type=x.device.type, enabled=False):
            points = embedding.float().transpose(-1, -2).contiguous()
            # Direct Euclidean distances avoid cancellation in the GEMM cdist path.
            distances = torch.cdist(points, points, compute_mode='donot_use_mm_for_euclid_dist')
            nearest, indices = distances.topk(self.neighbours, largest=False, dim=-1)
            incidence = torch.zeros_like(distances).scatter(-1, indices, (-nearest).softmax(-1))
            diagonal = torch.diag_embed(weights.float())
            edge_norm = incidence.abs().sum(-2, keepdim=True) + 1e-8
            weighted = incidence @ diagonal
            vertex_norm = weighted.abs().sum(-1, keepdim=True) + 1e-8
            dynamic = (weighted / vertex_norm) @ (diagonal / edge_norm) @ incidence.transpose(-1, -2)
            fixed = self.PA.float() * self.edge_importance.float()
            fixed = fixed / (fixed.abs().sum(-1, keepdim=True) + 1e-8)
            graph = fixed + self.alpha.float().relu() * dynamic
        projected = self.conv_d(augmented).reshape(n, self.heads, c // self.heads, t, total)
        # FP32 reduction prevents signed hyperedge weights overflowing under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            result = torch.einsum('nhvu,nhctu->nhctv', graph, projected.float())
        result = result.reshape(n, c, t, total)[..., :v].to(x.dtype)
        return torch.relu(self.bn(result) + x)


class DeformableTemporalConv(nn.Module):
    """DeGCN DeTGC: learned shared temporal offsets, linear interpolation + Conv3d."""
    def __init__(self, channels, *, stride=1, dilation=1, samples=4, kernel_size=5):
        super().__init__()
        if min(channels, stride, dilation, samples) < 1 or kernel_size % 2 != 1:
            raise ValueError('invalid DeTGC dimensions')
        self.stride, self.samples = stride, samples
        radius = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.tr = nn.Parameter(torch.linspace(-radius, radius, samples))
        self.conv_out = nn.Sequential(
            nn.Conv3d(channels, channels, (samples, 1, 1)), nn.BatchNorm3d(channels))

    def forward(self, x):
        n, c, length, v = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            centers = torch.arange(0, length, self.stride, device=x.device, dtype=torch.float32)
            locations = self.tr.float()[:, None] + centers[None, :]
            lower = locations.detach().floor()
            upper = lower + 1
            fraction = (upper - locations).reshape(1, 1, -1, 1)
            lo = lower.clamp(0, length - 1).long().reshape(1, 1, -1, 1).expand(n, c, -1, v)
            hi = upper.clamp(0, length - 1).long().reshape(1, 1, -1, 1).expand(n, c, -1, v)
            sampled = x.float().gather(2, lo) * fraction + x.float().gather(2, hi) * (1 - fraction)
            sampled = sampled.reshape(n, c, self.samples, centers.numel(), v)
        return self.conv_out(sampled.to(x.dtype)).squeeze(2)


class MultiScaleDeformableTemporal(nn.Module):
    """The upstream four-branch MultiScale_TemporalModeling, with eta=4."""
    def __init__(self, channels, stride=1, samples=4):
        super().__init__()
        if channels % 4:
            raise ValueError('DeGCN temporal branches require channels divisible by four')
        width = channels // 4

        def pointwise(step=1):
            return nn.Sequential(nn.Conv2d(channels, width, 1, stride=(step, 1)), nn.BatchNorm2d(width))

        self.branches = nn.ModuleList([
            nn.Sequential(pointwise(), nn.LeakyReLU(0.1),
                          DeformableTemporalConv(width, stride=stride, dilation=dilation, samples=samples))
            for dilation in (1, 2)
        ])
        self.branches.append(nn.Sequential(
            pointwise(), nn.LeakyReLU(0.1), nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(width)))
        self.branches.append(pointwise(stride))
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)
