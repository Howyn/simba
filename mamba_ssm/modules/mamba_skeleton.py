"""Optional offline temporal SSM variants for Simba.

The unmodified Mamba remains the default.  These variants share the input
projection, local encoder and output projection; the optional changes affect
the state scan, low-rank delta generator or state read/write parameters. They reuse the
existing selective-scan CUDA kernel without changing its recurrence.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class SkeletonMamba(Mamba):
    """Optional state scans, motion delta and normalized state read/write.

    ``bidirectional`` adds a reverse scan over the SAME encoded sequence and
    selective parameters.  It is intentionally not two complete Mambas.
    ``motion_delta`` adds a bounded correction in ``dt_rank`` space, based on
    adjacent encoded-feature differences.  The zero-initialized final motion
    projection starts this branch at the original delta generator.
    ``bc_norm`` applies per-token RMS normalization and controlled gain/bias
    to B and C along the state dimension before the existing scan. Its
    gain/bias values are adaptation settings, not Mamba-3 paper defaults.

    The enhanced path handles full offline clips only, not decoding caches.
    """

    def __init__(self, *args, bidirectional=False, motion_delta=False,
                 motion_rank=16, motion_scale=0.1, bidir_init=0.1,
                 bc_norm=False, bc_norm_gain=0.1, bc_norm_bias=0.1,
                 bc_norm_eps=1e-6, **kwargs):
        if not all(isinstance(value, bool) for value in (bidirectional, motion_delta, bc_norm)):
            raise ValueError('bidirectional, motion_delta and bc_norm must be booleans')
        if not isinstance(motion_rank, int) or isinstance(motion_rank, bool) or motion_rank < 1:
            raise ValueError('motion_rank must be a positive integer')
        if not math.isfinite(motion_scale) or motion_scale <= 0:
            raise ValueError('motion_scale must be finite and positive')
        if not math.isfinite(bidir_init) or not 0 < bidir_init < 1:
            raise ValueError('bidir_init must be between 0 and 1')
        if isinstance(bc_norm_gain, bool) or not math.isfinite(bc_norm_gain) or bc_norm_gain <= 0:
            raise ValueError('bc_norm_gain must be finite and positive')
        if isinstance(bc_norm_bias, bool) or not math.isfinite(bc_norm_bias):
            raise ValueError('bc_norm_bias must be finite')
        if isinstance(bc_norm_eps, bool) or not math.isfinite(bc_norm_eps) or bc_norm_eps <= 0:
            raise ValueError('bc_norm_eps must be finite and positive')
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.motion_delta = motion_delta
        self.motion_scale = float(motion_scale)
        self.bc_norm = bc_norm
        self.bc_norm_eps = float(bc_norm_eps)

        if bc_norm:
            # Four state-size vectors: 64 parameters for d_state=16. Constant
            # initialization leaves later backbone RNG draws unchanged.
            self.bc_B_gain = nn.Parameter(self.D.new_full((self.d_state,), float(bc_norm_gain)))
            self.bc_C_gain = nn.Parameter(self.D.new_full((self.d_state,), float(bc_norm_gain)))
            self.bc_B_bias = nn.Parameter(self.D.new_full((self.d_state,), float(bc_norm_bias)))
            self.bc_C_bias = nn.Parameter(self.D.new_full((self.d_state,), float(bc_norm_bias)))

        if bidirectional:
            gate_logit = math.log(bidir_init / (1 - bidir_init))
            self.bidir_logit = nn.Parameter(self.D.new_full((self.d_inner,), gate_logit))

        if motion_delta:
            # Add-on initialization must not change the random initialization
            # of subsequent backbone layers under the existing baseline seed.
            device = self.in_proj.weight.device
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            factory = {'device': device, 'dtype': self.in_proj.weight.dtype}
            with torch.random.fork_rng(devices=devices):
                self.motion_norm = nn.LayerNorm(self.d_inner, elementwise_affine=False, **factory)
                self.motion_down = nn.Linear(self.d_inner, motion_rank, bias=False, **factory)
                self.motion_up = nn.Linear(motion_rank, self.dt_rank, bias=False, **factory)
                nn.init.zeros_(self.motion_up.weight)

    def _normalize_bc(self, values, gain, bias):
        # Input layout is (batch*time, state). No statistics are shared across
        # frames, examples or B/C. Keep reductions and affine arithmetic FP32.
        original_dtype = values.dtype
        values = values.float()
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.bc_norm_eps)
        return (values * inverse_rms * gain.float() + bias.float()).to(dtype=original_dtype)

    def forward(self, hidden_states, inference_params=None):
        if not (self.bidirectional or self.motion_delta or self.bc_norm):
            return super().forward(hidden_states, inference_params=inference_params)
        if inference_params is not None:
            raise ValueError('SkeletonMamba enhancements require full offline clips; inference caches are unsupported')
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.d_model or hidden_states.shape[1] < 1:
            raise ValueError('expected non-empty hidden_states with shape (batch, time, d_model)')

        seqlen = hidden_states.shape[1]
        # Match vanilla Mamba's projection layout and autocast behavior.
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, 'b l d -> d (b l)'),
            'd (b l) -> b d l', l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + self.in_proj.bias.to(dtype=xz.dtype)[None, :, None]
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x, weight=rearrange(self.conv1d.weight, 'd 1 w -> d w'),
                bias=self.conv1d.bias, activation=self.activation,
            )

        x_dbl = self.x_proj(rearrange(x, 'b d l -> (b l) d'))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        if self.bc_norm:
            B = self._normalize_bc(B, self.bc_B_gain, self.bc_B_bias)
            C = self._normalize_bc(C, self.bc_C_gain, self.bc_C_bias)
        if self.motion_delta:
            tokens = rearrange(x, 'b d l -> b l d')
            # The first frame has no preceding observation (including T=1).
            motion = torch.cat((torch.zeros_like(tokens[:, :1]), tokens[:, 1:] - tokens[:, :-1]), dim=1)
            correction = self.motion_up(F.silu(self.motion_down(self.motion_norm(motion))))
            dt = dt + self.motion_scale * rearrange(torch.tanh(correction), 'b l r -> (b l) r')
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, 'd (b l) -> b d l', l=seqlen)
        B = rearrange(B, '(b l) n -> b n l', l=seqlen).contiguous()
        C = rearrange(C, '(b l) n -> b n l', l=seqlen).contiguous()
        A = -torch.exp(self.A_log.float())

        if not self.bidirectional:
            # Keep the one-direction scan's D/z accumulation/cast order. When
            # BC norm is off, zero-initialized motion starts at the baseline.
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            )
            return self.out_proj(rearrange(y, 'b d l -> b l d'))

        # With D/z outside the scan, they are applied exactly once after
        # direction fusion. The original dt bias and softplus stay in scan.
        y = selective_scan_fn(x, dt, A, B, C, D=None, z=None,
                              delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        if self.bidirectional:
            backward = selective_scan_fn(
                x.flip(-1), dt.flip(-1), A, B.flip(-1), C.flip(-1),
                D=None, z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            ).flip(-1)
            gate = torch.sigmoid(self.bidir_logit.float()).to(dtype=y.dtype)[None, :, None]
            y = y + gate * (backward - y)
        # Match the scan kernel's fp32 D skip accumulation before its cast.
        y = (y.float() + self.D.float()[None, :, None] * x.float()) * F.silu(z.float())
        y = y.to(dtype=x.dtype)
        return self.out_proj(rearrange(y, 'b d l -> b l d'))

    def step(self, hidden_states, conv_state, ssm_state):
        if self.bidirectional or self.motion_delta or self.bc_norm:
            raise ValueError('SkeletonMamba enhancements require full offline clips; step decoding is unsupported')
        return super().step(hidden_states, conv_state, ssm_state)
