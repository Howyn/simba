"""Shared per-joint temporal state branch for Simba's compressed features.

The branch borrows the pure temporal, per-joint sequence organization from
TSkel-Mamba (https://arxiv.org/abs/2512.11503). It is a Simba adaptation,
not a reproduction of that complete model. The global Simba Mamba, outer
residual, and normalization are owned by the caller.

This module has no D*u shortcut and no outer residual. Its only output is
the projected, gated C_t*h_t state readout. States start at zero for every
call and remain independent across examples, persons, and joint sequences.
"""

import torch
from torch.nn import functional as F
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba, causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def joint_state_scan_ref(u, delta, A, B, C, delta_bias=None,
                         delta_softplus=True):
    """Differentiable real-valued reference recurrence with no D or z.

    Intended for isolated CPU checks and very short diagnostic inputs, not
    full CUDA training. Accumulation keeps float64 reference inputs in
    float64 and promotes half/bfloat16 inputs to float32.
    """
    output_dtype = u.dtype
    compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, A, B, C = [value.to(compute_dtype) for value in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.to(compute_dtype)[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for index in range(u.shape[-1]):
        dt = delta[:, :, index, None]
        state = (dt * A[None]).exp() * state
        state = state + dt * B[:, None, :, index] * u[:, :, index, None]
        outputs.append((state * C[:, None, :, index]).sum(dim=-1))
    return torch.stack(outputs, dim=-1).to(output_dtype)


class JointStateMamba(Mamba):
    """A small shared Mamba that independently scans every joint in time.

    Input/output: [batch_times_persons, frames, num_joints*joint_channels].
    The flattened feature order must be joint-major then channel-major,
    matching Simba's ``permute(0, 2, 3, 1).reshape(B,T,V*C)``.

    Initialization runs in a forked RNG scope so adding this branch does not
    alter later common backbone initialization or the data-loader RNG stream.
    ``state_disabled`` is an in-memory diagnostic toggle, never checkpointed.
    ``diagnostics_enabled`` retains only detached summary tensors.
    """

    def __init__(self, num_joints=25, joint_channels=25, d_state=16,
                 d_conv=4, expand=2, device=None, dtype=None):
        for name, value in (('num_joints', num_joints), ('joint_channels', joint_channels),
                            ('d_state', d_state), ('d_conv', d_conv), ('expand', expand)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        target = torch.device(device) if device is not None else torch.device('cpu')
        devices = []
        if target.type == 'cuda':
            devices = [target.index if target.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices):
            super().__init__(d_model=joint_channels, d_state=d_state,
                             d_conv=d_conv, expand=expand, use_fast_path=False,
                             device=device, dtype=dtype)
        # Do not leave an unused trainable D parameter in the optimizer or in
        # task-gradient checks. This branch is deliberately state-only.
        del self.D
        self.num_joints = num_joints
        self.joint_channels = joint_channels
        self.state_disabled = False
        self.force_reference_scan = False
        self.diagnostics_enabled = False
        self.last_diagnostics = {}

    def forward(self, hidden_states, inference_params=None, *, return_components=False):
        if inference_params is not None:
            raise ValueError('JointStateMamba accepts complete clips only; caches are unsupported')
        expected = self.num_joints * self.joint_channels
        if (hidden_states.ndim != 3 or hidden_states.shape[-1] != expected
                or hidden_states.shape[1] < 1):
            raise ValueError(f'expected non-empty [batch, time, {expected}] input')
        batch, frames, _ = hidden_states.shape
        tokens = rearrange(hidden_states, 'b t (v c) -> (b v) t c',
                           v=self.num_joints, c=self.joint_channels)
        # Match the vanilla projection layout and CUDA autocast behavior.
        xz = rearrange(self.in_proj.weight @ rearrange(tokens, 'b t c -> c (b t)'),
                       'c (b t) -> b c t', t=frames)
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None or not x.is_cuda:
            x = self.act(self.conv1d(x)[..., :frames])
        else:
            x = causal_conv1d_fn(
                x=x, weight=rearrange(self.conv1d.weight, 'd 1 w -> d w'),
                bias=self.conv1d.bias, activation=self.activation)

        parameters = self.x_proj(rearrange(x, 'b d t -> (b t) d'))
        rank, B, C = torch.split(parameters,
                                 [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = rearrange(self.dt_proj.weight @ rank.t(),
                          'd (b t) -> b d t', t=frames)
        B = rearrange(B, '(b t) n -> b n t', t=frames).contiguous()
        C = rearrange(C, '(b t) n -> b n t', t=frames).contiguous()
        A = -torch.exp(self.A_log.float())
        if x.is_cuda and not self.force_reference_scan:
            readout = selective_scan_fn(
                x, delta, A, B, C, D=None, z=None,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        else:
            readout = joint_state_scan_ref(
                x, delta, A, B, C, delta_bias=self.dt_proj.bias.float())
        if self.state_disabled:
            readout = torch.zeros_like(readout)
        # Preserve the scan result's precision, with FP32 z gating before
        # casting back for the output projection, just as the native path.
        gated = (readout.float() * F.silu(z.float())).to(x.dtype)
        projected = self.out_proj(rearrange(gated, 'b d t -> b t d'))
        output = rearrange(projected, '(b v) t c -> b t (v c)',
                           b=batch, v=self.num_joints)
        if self.diagnostics_enabled:
            with torch.no_grad():
                effective_delta = F.softplus(delta.float() + self.dt_proj.bias.float()[None, :, None])
                self.last_diagnostics = {
                    'state_readout_rms': readout.detach().float().square().mean().sqrt(),
                    'output_rms': output.detach().float().square().mean().sqrt(),
                    'delta_mean': effective_delta.detach().mean(),
                    'retention_mean': torch.exp(
                        effective_delta.detach()[..., None] * A.detach()[None, :, None, :]).mean(),
                }
        if return_components:
            return {
                'output': output,
                'state_output': output,
                'state_readout': readout,
                'local_encoded': x,
                'effective_delta': F.softplus(
                    delta.float() + self.dt_proj.bias.float()[None, :, None]),
                'A': A,
            }
        return output

    def probe_components(self, hidden_states):
        """Return this call's tensors without retaining their graph on self.

        ``output`` and ``state_output`` are the joint-state contribution in
        the original flattened layout. ``state_readout`` and local features
        retain [batch*joints, inner_channels, frames] for state diagnostics.
        The caller can detach summaries or differentiate the returned tensors.
        """
        return self.forward(hidden_states, return_components=True)

    def step(self, hidden_states, conv_state, ssm_state):
        raise ValueError('JointStateMamba accepts complete clips only; step decoding is unsupported')
