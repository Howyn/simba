"""Multi-scale context write and decoupled retain/write gains for Simba.

Two independent, composable modifications of the same state update:

    h_t = exp(r_t * delta_t * A) h_{t-1} + w_t * delta_t * B_t^context u_t
    m_t = C_t h_t
    y_t = (m_t + D * u_t) * SiLU(z_t)

``context_write`` (candidate P) replaces the write matrix ``B_t`` produced by
the shared ``x_proj`` with ``B_t + scale * tanh(residual_t)``, where the
residual is built from a bounded, multi-scale neighbourhood of the encoded
input. It borrows the multi-scale temporal interaction idea from TSkel-Mamba
(https://arxiv.org/abs/2512.11503) but applies the context to the state write
parameter; it is a Simba adaptation, not a reproduction of that paper.

``retain_write`` (candidate Q) decouples the two roles of the positive step
size: ``r`` sets how fast history decays and ``w`` sets the strength of the
current write. It borrows the independent-memory-gate idea associated with
Naju, implemented here as our own bounded per-channel adaptation.

Both gains are bounded and their initial values are exactly the requested
ones. The scan kernel is reused unchanged: scaling ``u`` by ``w / r`` keeps
the write term at ``w * delta * B * u`` while the scan's own ``delta * r``
produces the longer retention. ``D`` and the ``z`` gate are applied once,
outside the scan, on the original ``u`` so the skip path is untouched.

The unmodified Mamba path remains the default. New parameters are created
inside a forked RNG scope, so adding these modules does not change later
backbone initialization under the same seed. States start at zero for every
offline clip; cached/token-by-token decoding is unsupported.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def _positive_odd_kernels(values):
    try:
        kernels = tuple(int(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError('context_kernels must be an iterable of odd positive integers') from None
    if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError('context_kernels must be a nonempty tuple of odd positive integers')
    if len(set(kernels)) != len(kernels):
        raise ValueError('context_kernels must not repeat a scale')
    return kernels


class ContextWriteMamba(Mamba):
    """Optional multi-scale write context and independent retain/write gains.

    ``context_write`` adds a bounded multi-scale correction to ``B``.
    ``retain_write`` multiplies the positive step by a bounded per-channel
    ``retain`` in ``[retain_floor, 1)`` and the write by a bounded per-channel
    ``write`` in ``(0, write_max]``. They may be enabled together.

    Diagnostic toggles are in-memory only and are never checkpointed:
    ``context_enabled`` / ``retain_write_enabled`` remove one mechanism,
    ``correction_enabled`` removes both while keeping the explicit scan path
    (for calibrating the fused-versus-explicit numerical floor), and
    ``state_enabled=False`` removes the state readout while retaining ``D``
    and the ``z`` gate.
    """

    def __init__(self, *args, context_write=False, retain_write=False,
                 context_rank=16, context_kernels=(3, 5, 7), context_scale=0.1,
                 retain_init=0.5, retain_floor=0.25,
                 write_init=1.0, write_max=2.0, **kwargs):
        if not isinstance(context_write, bool) or not isinstance(retain_write, bool):
            raise ValueError('context_write and retain_write must be booleans')
        if not context_write and not retain_write:
            raise ValueError('ContextWriteMamba needs context_write and/or retain_write enabled')
        if isinstance(context_rank, bool) or not isinstance(context_rank, int) or context_rank < 1:
            raise ValueError('context_rank must be a positive integer')
        if not math.isfinite(context_scale) or context_scale <= 0:
            raise ValueError('context_scale must be finite and positive')
        if isinstance(retain_floor, bool) or not math.isfinite(retain_floor) or not 0 < retain_floor < 1:
            raise ValueError('retain_floor must be finite and in (0, 1)')
        if (isinstance(retain_init, bool) or not math.isfinite(retain_init)
                or not retain_floor < retain_init < 1):
            raise ValueError('retain_init must be finite and strictly between retain_floor and 1')
        if isinstance(write_max, bool) or not math.isfinite(write_max) or write_max <= 0:
            raise ValueError('write_max must be finite and positive')
        if (isinstance(write_init, bool) or not math.isfinite(write_init)
                or not 0 < write_init < write_max):
            raise ValueError('write_init must be finite and strictly between 0 and write_max')

        super().__init__(*args, **kwargs)
        self.context_write = context_write
        self.retain_write = retain_write
        self.context_rank = int(context_rank)
        self.context_kernels = _positive_odd_kernels(context_kernels)
        self.context_scale = float(context_scale)
        self.retain_floor = float(retain_floor)
        self.write_max = float(write_max)
        self.context_enabled = True
        self.retain_write_enabled = True
        self.correction_enabled = True
        self.state_enabled = True

        if retain_write:
            # Constant initialization consumes no RNG and keeps sigmoid(logit)
            # exactly at the requested initial gain.
            retain_logit = math.log((retain_init - self.retain_floor)
                                    / (1 - retain_init))
            write_logit = math.log(write_init / (write_max - write_init))
            self.retain_logit = nn.Parameter(self.D.new_full((self.d_inner,), retain_logit))
            self.write_logit = nn.Parameter(self.D.new_full((self.d_inner,), write_logit))
            for parameter in (self.retain_logit, self.write_logit):
                # A decaying gain logit would drift away from the published
                # initial gain even with zero task gradient.
                parameter._no_weight_decay = True

        if context_write:
            # Add-on initialization must not change later backbone draws.
            device = self.in_proj.weight.device
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                       if device.type == 'cuda' else [])
            factory = {'device': device, 'dtype': self.in_proj.weight.dtype}
            scale = 1.0 / math.sqrt(self.d_inner)
            with torch.random.fork_rng(devices=devices):
                self.context_norm = nn.LayerNorm(self.d_inner, elementwise_affine=False, **factory)
                self.context_down = nn.Linear(self.d_inner, self.context_rank, bias=False, **factory)
                self.context_convs = nn.ModuleList(
                    nn.Conv1d(self.context_rank, self.context_rank, kernel_size=kernel,
                              groups=self.context_rank, bias=True, **factory)
                    for kernel in self.context_kernels)
                self.context_fuse = nn.Linear(self.context_rank * len(self.context_kernels),
                                              self.context_rank, bias=True, **factory)
                self.context_up = nn.Linear(self.context_rank, self.d_state, bias=False, **factory)
                # Small non-zero initialization: a zero last layer would block
                # the branch at step one and hide its signal for several steps.
                nn.init.normal_(self.context_up.weight, std=scale)

    def retain_gain(self):
        """Bounded per-channel retention multiplier in (retain_floor, 1)."""
        return (self.retain_floor
                + (1 - self.retain_floor) * torch.sigmoid(self.retain_logit.float()))

    def write_gain(self):
        """Bounded per-channel write multiplier in (0, write_max)."""
        return self.write_max * torch.sigmoid(self.write_logit.float())

    def _context_residual(self, u):
        """Bounded, two-sided multi-scale residual in state space.

        Layout conversion is explicit: ``u`` is (batch, inner, frames) while
        the temporal convolutions consume (batch, rank, frames). Zero padding
        is used at both clip boundaries; clips are never wrapped circularly.
        """
        hidden = self.context_norm(rearrange(u, 'b d l -> b l d'))
        hidden = rearrange(self.context_down(hidden), 'b l r -> b r l')
        branches = []
        for conv in self.context_convs:
            padding = (conv.kernel_size[0] - 1) // 2
            branches.append(conv(F.pad(hidden, (padding, padding))))
        fused = F.silu(self.context_fuse(
            rearrange(torch.cat(branches, dim=1), 'b c l -> b l c')))
        # (batch, frames, d_state) -> (batch, d_state, frames) to match B itself.
        return rearrange(self.context_up(fused), 'b l n -> b n l')

    def _encode(self, hidden_states):
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
        delta = F.softplus(raw_delta.float() + self.dt_proj.bias.float()[None, :, None])
        delta = delta.to(dtype=u.dtype).contiguous()
        B = rearrange(B, '(b l) n -> b n l', l=length).to(dtype=u.dtype).contiguous()
        C = rearrange(C, '(b l) n -> b n l', l=length).to(dtype=u.dtype).contiguous()
        return u, z, delta, B, C

    def _scan(self, u, delta, B, C):
        return selective_scan_fn(u, delta, -self.A_log.float().exp(), B, C,
                                 D=None, z=None, delta_bias=None, delta_softplus=False)

    def _finish(self, state_readout, u, z):
        # Match Mamba's fp32 D/z accumulation; the CUDA scan returns u.dtype.
        y = (state_readout.float() + self.D.float()[None, :, None] * u.float()) * F.silu(z.float())
        return self.out_proj(rearrange(y.to(u.dtype), 'b d l -> b l d'))

    def _compute(self, hidden_states, components=False):
        u, z, delta, B_base, C = self._encode(hidden_states)
        active = self.correction_enabled
        write_shape = (1, self.d_inner, 1)

        context_active = active and self.context_write and self.context_enabled
        gains_active = active and self.retain_write and self.retain_write_enabled

        context_delta = None
        if context_active:
            context_delta = self.context_scale * torch.tanh(self._context_residual(u))
            B = B_base + context_delta.to(dtype=B_base.dtype)
        else:
            B = B_base

        retain = torch.ones(self.d_inner, dtype=torch.float32, device=u.device)
        write = torch.ones(self.d_inner, dtype=torch.float32, device=u.device)
        if gains_active:
            retain = self.retain_gain()
            write = self.write_gain()
            delta_write = delta * retain.view(write_shape).to(dtype=delta.dtype)
            # Fold the independent write strength into the scan input: the
            # kernel computes delta * B * u, so this keeps the write at
            # w * delta * B * u while delta * r sets retention.
            u_write = u * (write / retain).view(write_shape).to(dtype=u.dtype)
        else:
            delta_write, u_write = delta, u

        readout = self._scan(u_write, delta_write, B, C)
        if not self.state_enabled:
            readout = torch.zeros_like(readout)
        output = self._finish(readout, u, z)
        if not components:
            return output

        base_readout = self._scan(u, delta, B_base, C)
        with torch.no_grad():
            A = -self.A_log.float().exp()
            retention = torch.exp(delta_write.float().unsqueeze(-1) * A[None, :, None, :])
            retention_base = torch.exp(delta.float().unsqueeze(-1) * A[None, :, None, :])
            state_rms = readout.detach().float().square().mean().sqrt()
            base_state_rms = base_readout.detach().float().square().mean().sqrt()
        return {
            'output': output,
            'output_base': self._finish(base_readout, u, z),
            'state_readout': readout,
            'state_readout_base': base_readout,
            'delta': delta,
            'delta_write': delta_write,
            'retain': retain,
            'write': write,
            'u_write_scale': (write / retain),
            'retention': retention,
            'retention_base': retention_base,
            'B': B,
            'B_base': B_base,
            'B_context': context_delta,
            'state_rms': state_rms,
            'state_rms_base': base_state_rms,
            'state_energy_ratio': state_rms / base_state_rms.clamp_min(1e-12),
            'context_relative': (0.0 if context_delta is None
                                 else (context_delta.float().norm()
                                       / B_base.float().norm().clamp_min(1e-12)).item()),
            'A': A,
        }

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise ValueError('ContextWriteMamba requires full offline clips; inference caches are unsupported')
        active = self.correction_enabled and (
            (self.context_write and self.context_enabled)
            or (self.retain_write and self.retain_write_enabled))
        if not active and self.state_enabled:
            if not self.correction_enabled:
                # Explicit no-op path: keeps the new operator wiring but
                # removes the actual correction, so the caller can calibrate
                # fused-versus-explicit numerical differences.
                return self._compute(hidden_states)
            # Both mechanisms switched off at the module toggles: take the
            # unchanged fused Mamba path, exactly as the baseline does.
            return super().forward(hidden_states)
        return self._compute(hidden_states)

    def probe_components(self, hidden_states):
        """Return differentiable intermediates without keeping graphs on self."""
        return self._compute(hidden_states, components=True)

    def step(self, hidden_states, conv_state, ssm_state):
        raise ValueError('ContextWriteMamba requires full offline clips; step decoding is unsupported')
