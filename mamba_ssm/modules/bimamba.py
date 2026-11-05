"""Vim v2 bidirectional SSM, adapted to Simba's existing temporal tokens.

Source: Vision Mamba, ICML 2024, https://proceedings.mlr.press/v235/zhu24f.html
Author implementation: hustvl/Vim, mamba-1p1p1/mamba_ssm/modules/mamba_simple.py
(bimamba_type='v2', if_divide_out=True).

Input/output projections are shared; convolution, x/dt projections and SSM
parameters are direction-specific. Existing local fused kernels are reused.
Applying the shared linear output projection to each direction before averaging
is algebraically equivalent to projecting the averaged hidden states. It avoids
installing Vim's different CUDA extension, at the cost of a second projection.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_ref


class BiMambaV2(Mamba):
    def __init__(self, *args, **kwargs):
        # Construct exactly the baseline parameters first, under the same names.
        super().__init__(*args, **kwargs)
        device = self.in_proj.weight.device
        devices = [device.index] if device.type == 'cuda' else []
        # Extra initialization must not shift later baseline weights/data RNG.
        with torch.random.fork_rng(devices=devices):
            reverse = Mamba(*args, **kwargs)
        self.conv1d_b = reverse.conv1d
        self.x_proj_b = reverse.x_proj
        self.dt_proj_b = reverse.dt_proj
        self.A_b_log = reverse.A_log
        self.D_b = reverse.D
        # reverse.in_proj/out_proj are deliberately discarded, not registered.

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise ValueError('BiMambaV2 requires a complete offline temporal sequence, not a decoding cache')
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1 or hidden_states.shape[-1] != self.d_model:
            raise ValueError(f'expected (B,T,{self.d_model}) input')
        if not self.use_fast_path:
            return self.forward_reference(hidden_states)
        length = hidden_states.shape[1]
        # Same projection/layout as the original Simba Mamba fast path.
        xz = rearrange(self.in_proj.weight @ rearrange(hidden_states, 'b t d -> d (b t)'),
                       'd (b t) -> b d t', t=length)
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.to(xz.dtype)[None, :, None]

        def direction(values, conv, x_proj, dt_proj, a_log, skip):
            return mamba_inner_fn(
                values, conv.weight, conv.bias, x_proj.weight, dt_proj.weight,
                self.out_proj.weight, self.out_proj.bias, -torch.exp(a_log.float()),
                None, None, skip.float(), delta_bias=dt_proj.bias.float(), delta_softplus=True)

        forward = direction(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        backward = direction(xz.flip(-1), self.conv1d_b, self.x_proj_b,
                             self.dt_proj_b, self.A_b_log, self.D_b)
        # xz is B,2*inner,T, while projected outputs are B,T,d_model.
        return (forward + backward.flip(1)) * 0.5

    def allocate_inference_cache(self, *args, **kwargs):
        raise ValueError('bidirectional temporal modeling requires the complete sequence')

    def step(self, *args, **kwargs):
        raise ValueError('bidirectional temporal modeling does not support causal decoding steps')

    def forward_reference(self, hidden_states):
        """Independent unfused Torch recurrence for output/gradient verification."""
        xz = self.in_proj(hidden_states).transpose(1, 2)
        length = hidden_states.shape[1]

        def direction(values, conv, x_proj, dt_proj, a_log, skip):
            x, z = values.chunk(2, dim=1)
            x = F.silu(F.conv1d(x, conv.weight, conv.bias,
                               padding=self.d_conv - 1, groups=self.d_inner)[..., :length])
            projected = F.linear(x.transpose(1, 2), x_proj.weight)
            dt, b, c = projected.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.linear(dt, dt_proj.weight).transpose(1, 2)
            y = selective_scan_ref(x, dt, -torch.exp(a_log.float()),
                                   b.transpose(1, 2), c.transpose(1, 2), skip.float(),
                                   z=z, delta_bias=dt_proj.bias.float(), delta_softplus=True)
            return self.out_proj(y.transpose(1, 2))

        forward = direction(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
        backward = direction(xz.flip(-1), self.conv1d_b, self.x_proj_b,
                             self.dt_proj_b, self.A_b_log, self.D_b)
        return (forward + backward.flip(1)) * 0.5
