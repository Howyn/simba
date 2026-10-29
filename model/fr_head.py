"""Multi-level FR-Head adapted to Simba's skeleton feature maps.

The spatial/temporal branches and V0 contrastive refinement follow
Zhou et al., CVPR 2023, https://arxiv.org/abs/2303.03729, and the official
implementation https://github.com/zhysora/FR-Head (model/lib.py). The
adaptation registers prototype memories as buffers so checkpoints restore
them and accepts Simba's [N*M, C, T, V] intermediate features directly.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _RefinementLoss(nn.Module):
    """Prototype contrastive objective with FN/FP calibration (V0)."""

    def __init__(self, in_features, hidden_channels, num_classes,
                 momentum=0.9, temperature=0.125, alpha=0.125):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("FR-Head momentum must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("FR-Head temperature must be positive")
        self.num_classes = num_classes
        self.momentum = momentum
        self.temperature = temperature
        self.alpha = alpha
        self.projection = nn.Linear(in_features, hidden_channels)
        # Like the reference implementation, initialize class memories at random.
        # Registering as a buffer makes resume and device moves deterministic.
        self.register_buffer("prototypes", torch.randn(num_classes, hidden_channels))

    def forward(self, feature, logits, labels):
        # Disable outer CUDA autocast: matmul would otherwise downcast even
        # explicit float inputs and may destabilize prototype statistics.
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_float(feature.float(), logits, labels)

    def _forward_float(self, feature, logits, labels):
        projected = self.projection(feature)
        labels = labels.detach().long().reshape(-1)
        logits = logits.detach().float()
        if projected.shape[0] != labels.shape[0] or logits.shape != (labels.shape[0], self.num_classes):
            raise ValueError("FR-Head features, labels and logits have incompatible batch sizes")

        predictions = logits.argmax(dim=1)
        probabilities = logits.softmax(dim=1)
        true_mask = F.one_hot(labels, self.num_classes).to(projected.dtype)
        pred_mask = F.one_hot(predictions, self.num_classes).to(projected.dtype)
        tp = true_mask * pred_mask
        fn = true_mask - tp
        fp = pred_mask - tp

        def class_means(mask):
            count = mask.sum(dim=0)
            return mask.transpose(0, 1) @ projected / count.clamp_min(1).unsqueeze(1), count

        tp_mean, tp_count = class_means(tp)
        fn_mean, fn_count = class_means(fn)
        fp_mean, fp_count = class_means(fp)

        # Clone before the in-place buffer update: autograd needs this old value.
        old = self.prototypes.detach().clone().to(projected.dtype)
        has_tp = tp_count.gt(0).unsqueeze(1)
        memory = torch.where(has_tp, self.momentum * old + (1 - self.momentum) * tp_mean, old)
        with torch.no_grad():
            self.prototypes.copy_(memory.detach().to(self.prototypes.dtype))

        sample = F.normalize(projected, dim=1, eps=1e-12)
        memory = F.normalize(memory, dim=1, eps=1e-12)
        fn_mean = F.normalize(fn_mean, dim=1, eps=1e-12)
        fp_mean = F.normalize(fp_mean, dim=1, eps=1e-12)
        base_scores = sample @ memory.transpose(0, 1)
        # Reference V0 applies confidence-weighted FN/FP adjustments only to
        # the true-class column, and omits them when a class has no such sample.
        p_map = (1 - probabilities) * true_mask * self.alpha
        fn_adjust = (sample @ fn_mean.transpose(0, 1) - 1) * p_map * fn_count.gt(0)
        fp_adjust = (-sample @ fp_mean.transpose(0, 1) - 1) * p_map * fp_count.gt(0)
        fn_scores = (base_scores + fn_adjust) / self.temperature
        fp_scores = (base_scores + fp_adjust) / self.temperature
        return F.cross_entropy(fn_scores, labels) + F.cross_entropy(fp_scores, labels)


class _SpatialTemporalHead(nn.Module):
    def __init__(self, in_channels, num_frames, num_joints, num_classes,
                 hidden_channels, momentum, temperature, alpha):
        super().__init__()
        spatial_channels = hidden_channels // num_joints
        temporal_channels = hidden_channels // num_frames
        if min(spatial_channels, temporal_channels) < 1:
            raise ValueError("FR-Head hidden_channels must be at least num_frames and num_joints")
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.spatial = nn.Sequential(nn.Conv2d(in_channels, spatial_channels, 1),
                                     nn.BatchNorm2d(spatial_channels), nn.ReLU(inplace=True))
        self.temporal = nn.Sequential(nn.Conv2d(in_channels, temporal_channels, 1),
                                      nn.BatchNorm2d(temporal_channels), nn.ReLU(inplace=True))
        self.spatial_loss = _RefinementLoss(spatial_channels * num_joints, hidden_channels,
                                            num_classes, momentum, temperature, alpha)
        self.temporal_loss = _RefinementLoss(temporal_channels * num_frames, hidden_channels,
                                             num_classes, momentum, temperature, alpha)

    def forward(self, raw, logits, labels, persons):
        if raw.ndim != 4 or raw.shape[0] != labels.shape[0] * persons:
            raise ValueError("FR-Head expects [batch * persons, channels, frames, joints]")
        if raw.shape[2:] != (self.num_frames, self.num_joints):
            raise ValueError(f"FR-Head expected (T,V)={(self.num_frames, self.num_joints)}, "
                             f"got {tuple(raw.shape[2:])}; check num_frames/window_size")
        raw = raw.reshape(labels.shape[0], persons, *raw.shape[1:]).mean(dim=1)
        spatial = self.spatial(raw.mean(dim=2, keepdim=True)).flatten(1)
        temporal = self.temporal(raw.mean(dim=3, keepdim=True)).flatten(1)
        return self.spatial_loss(spatial, logits, labels) + self.temporal_loss(temporal, logits, labels)


class MultiLevelFRHead(nn.Module):
    """Train-only auxiliary loss for selected Simba layers.

    ``features`` must correspond to ``levels`` in order. Pass input frame count
    matching the feeder window_size. Level outputs 5 and 8 halve time length.
    """

    def __init__(self, in_channels=216, num_classes=60, num_person=2,
                 levels=(5, 8, 10), level_weights=(0.2, 0.5, 1.0),
                 hidden_channels=256, num_frames=64, num_joints=25,
                 momentum=0.9, temperature=0.125, alpha=0.125):
        super().__init__()
        self.levels = tuple(int(level) for level in levels)
        self.level_weights = tuple(float(weight) for weight in level_weights)
        self.num_person = int(num_person)
        if not self.levels or len(self.levels) != len(self.level_weights):
            raise ValueError("FR-Head levels and weights must be nonempty and have equal length")
        if len(set(self.levels)) != len(self.levels) or any(level < 1 or level > 10 for level in self.levels):
            raise ValueError("FR-Head levels must be distinct integers from 1 to 10")
        if self.num_person < 1:
            raise ValueError("FR-Head num_person must be positive")
        # Simba's l5 and l8 ShiftTCN use stride 2 and round odd T upward.
        frames_by_level = []
        for level in self.levels:
            frames = int(num_frames)
            if level >= 5:
                frames = (frames + 1) // 2
            if level >= 8:
                frames = (frames + 1) // 2
            frames_by_level.append(frames)
        self.heads = nn.ModuleList([
            _SpatialTemporalHead(in_channels, frames, num_joints, num_classes,
                                 hidden_channels, momentum, temperature, alpha)
            for frames in frames_by_level
        ])

    def forward(self, features, logits, labels, num_person=None):
        if not self.training:
            return logits.new_zeros(())
        if len(features) != len(self.heads):
            raise ValueError("FR-Head feature count must match configured levels")
        persons = self.num_person if num_person is None else int(num_person)
        losses = [weight * head(feature, logits, labels, persons)
                  for head, feature, weight in zip(self.heads, features, self.level_weights)]
        return torch.stack(losses).sum()
