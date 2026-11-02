"""Shared-projection, dual-time-scale state updates for offline Simba clips.

This is an adaptation of the multi-Delta idea in ms-Mamba, not a reproduction
of that paper's forecasting architecture:
https://arxiv.org/abs/2504.07654

The original Mamba parameter names are inherited unchanged. A second scan
shares its encoded inputs and selective parameters but uses half the positive
step size. D, z and out_proj are applied once after mixing state readouts.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class DualScaleMamba(Mamba):
    """Maintain a baseline state and a slower state with shared projections.

    ``slow_scale`` multiplies softplus(raw_delta + bias), not raw_delta.
    ``dual_scale_enabled=False`` takes the unchanged Mamba forward path.
    ``state_enabled=False`` is an inference diagnostic that removes both
    state readouts while retaining the original D skip and z gate.

    All state starts at zero for every independent clip. Cached/token-by-token
    inference is intentionally unsupported while the new path is enabled.
    """

    def __init__(self, *args, slow_scale=0.5, gate_init=0.1, **kwargs):
        if isinstance(slow_scale, bool) or not math.isfinite(slow_scale) or not 0 < slow_scale <= 1:
            raise ValueError('slow_scale must be finite and in (0, 1]')
        if isinstance(gate_init, bool) or not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError('gate_init must be finite and in (0, 1)')
        super().__init__(*args, **kwargs)
        self.slow_scale = float(slow_scale)
        # Constant initialization consumes no RNG, preserving the initialization
        # of subsequent shared backbone layers under a common random seed.
        logit = math.log(gate_init / (1 - gate_init))
        self.dual_scale_logit = nn.Parameter(self.D.new_full((self.d_inner,), logit))
        self.dual_scale_logit._no_weight_decay = True
        self.dual_scale_enabled = True
        self.state_enabled = True
        # Keep the enhanced operator path but remove its actual correction.
        # Used to calibrate fused-vs-explicit numerical differences.
        self.correction_enabled = True

    def _encoded_inputs(self, hidden_states):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model or hidden_states.shape[1] < 1:
            raise ValueError('expected non-empty (batch, time, d_model) hidden_states')
        length = hidden_states.shape[1]
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'),
            'd (b l) -> b d l', l=length,
        )
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.to(xz.dtype)[None, :, None]
        u, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            u = self.act(self.conv1d(u)[..., :length])
        else:
            u = causal_conv1d_fn(
                x=u, weight=rearrange(self.conv1d.weight, 'd 1 w -> d w'),
                bias=self.conv1d.bias, activation=self.activation,
            )
        parameters = self.x_proj(rearrange(u, 'b d l -> (b l) d'))
        raw_rank, B, C = torch.split(parameters, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        raw_delta = self.dt_proj.weight @ raw_rank.t()
        raw_delta = rearrange(raw_delta, 'd (b l) -> b d l', l=length)
        # Explicit fp32 softplus matches the positive-step parameterization.
        # The CUDA scan requires delta/B/C to share u's dtype. Under BF16 this
        # introduces one positive-step rounding; FP32 is the screening default.
        positive_delta = F.softplus(raw_delta.float() + self.dt_proj.bias.float()[None, :, None])
        delta_base = positive_delta.to(dtype=u.dtype).contiguous()
        delta_slow = (positive_delta * self.slow_scale).to(dtype=u.dtype).contiguous()
        B = rearrange(B, '(b l) n -> b n l', l=length).to(u.dtype).contiguous()
        C = rearrange(C, '(b l) n -> b n l', l=length).to(u.dtype).contiguous()
        return u, z, delta_base, delta_slow, B, C

    def _finish(self, state_readout, u, z):
        # Match Mamba's fp32 D/z accumulation. The CUDA state scan itself
        # returns u.dtype, so BF16 parity is tolerance-based, not bitwise.
        y = (state_readout.float() + self.D.float()[None, :, None] * u.float()) * F.silu(z.float())
        return self.out_proj(rearrange(y.to(u.dtype), 'b d l -> b l d'))

    def _compute(self, hidden_states, components=False):
        u, z, delta_base, delta_slow, B, C = self._encoded_inputs(hidden_states)
        A = -self.A_log.float().exp()
        state_base = selective_scan_fn(
            u, delta_base, A, B, C, D=None, z=None,
            delta_bias=None, delta_softplus=False,
        )
        state_slow = selective_scan_fn(
            u, delta_slow, A, B, C, D=None, z=None,
            delta_bias=None, delta_softplus=False,
        )
        gate = self.dual_scale_logit.float().sigmoid()[None, :, None]
        correction = gate * (state_slow.float() - state_base.float())
        if not self.correction_enabled:
            correction = torch.zeros_like(correction)
        mixed = state_base.float() + correction
        if not self.state_enabled:
            mixed = torch.zeros_like(mixed)
            correction = torch.zeros_like(correction)
        if not self.dual_scale_enabled:
            mixed = state_base.float() if self.state_enabled else torch.zeros_like(mixed)
            correction = torch.zeros_like(correction)
        output = self._finish(mixed, u, z)
        if not components:
            return output
        return {
            'output': output,
            'output_base': self._finish(state_base, u, z),
            'state_base': state_base,
            'state_slow': state_slow,
            'state_correction': correction,
            'delta_base': delta_base,
            'delta_slow': delta_slow,
            'retention_base': torch.exp(delta_base.float().unsqueeze(-1) * A[None, :, None, :]),
            'retention_slow': torch.exp(delta_slow.float().unsqueeze(-1) * A[None, :, None, :]),
            'gate': gate,
        }

    def forward(self, hidden_states, inference_params=None):
        if not self.dual_scale_enabled and self.state_enabled:
            return super().forward(hidden_states, inference_params=inference_params)
        if inference_params is not None:
            raise ValueError('DualScaleMamba requires full offline clips; inference caches are unsupported')
        return self._compute(hidden_states)

    def probe_components(self, hidden_states):
        """Return differentiable intermediates without storing graphs on self.

        Call under no_grad for a signal report; leave gradients enabled for
        task-gradient diagnostics. Retention arrays are produced only here.
        """
        return self._compute(hidden_states, components=True)

    def step(self, hidden_states, conv_state, ssm_state):
        if self.dual_scale_enabled or not self.state_enabled:
            raise ValueError('DualScaleMamba requires full offline clips; step decoding is unsupported')
        return super().step(hidden_states, conv_state, ssm_state)
