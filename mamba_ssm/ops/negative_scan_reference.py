"""Differentiable reference for the real, input-dependent Simba scan."""
import torch
import torch.nn.functional as F


def negative_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=True):
    if A.is_complex() or B.ndim != 3 or C.ndim != 3:
        raise ValueError('Reference supports real A and B,C of shape (batch,state,time)')
    dtype = u.dtype
    u, delta, A, B, C = (x.float() for x in (u, delta, A, B, C))
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for t in range(u.shape[-1]):
        dt = delta[:, :, t, None]
        a = 2 * torch.exp(dt * A[None]) - 1
        state = a * state + dt * B[:, None, :, t] * u[:, :, t, None]
        ys.append((state * C[:, None, :, t]).sum(-1))
    out = torch.stack(ys, -1)
    if D is not None:
        out = out + D.float()[None, :, None] * u
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)
