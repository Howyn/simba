"""Plain PyTorch reference for the selective scan, with the negative-eigenvalue variant.

The compiled kernel computes

    h[t] = a[t] * h[t-1] + delta[t] * B[t] * u[t],      a[t] = exp(delta[t] * A)

and the negative-eigenvalue expansion (Grazzi et al., "Unlocking State-Tracking in
Linear RNNs Through Negative Eigenvalues", ICLR 2025) only changes the transition

    a[t] = 2 * exp(delta[t] * A) - 1

so that a[t] may lie in (-1, 0) instead of (0, 1).  The scan itself is a general
first-order linear recurrence and does not assume the transition is a single
exponential decay, which is why no other part of the operator changes.

This module depends on torch only, so it can be imported and unit tested without
the compiled CUDA extension.
"""

import torch
import torch.nn.functional as F


def _working_dtype(dtype):
    """float64 stays float64 so the reference can be gradchecked; halves go to float32."""
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def selective_scan_reference(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                             delta_softplus=False, return_last_state=False,
                             negative=False):
    """Reference implementation of the selective scan.

    Args:
        u: (batch, dim, length)
        delta: (batch, dim, length)
        A: (dim, dstate), real weights (already negated, i.e. the SSM decay)
        B: (batch, dstate, length) or (dim, dstate)
        C: (batch, dstate, length) or (dim, dstate)
        D: (dim,) skip connection, optional
        z: (batch, dim, length) output gate, optional
        delta_bias: (dim,) added to delta before softplus, optional
        delta_softplus: apply softplus to delta + delta_bias
        return_last_state: also return the final state
        negative: use the transition 2 * exp(delta * A) - 1 instead of exp(delta * A)

    Returns:
        out: (batch, dim, length), same dtype as ``u`` (plus the last state).
    """
    if A.is_complex():
        raise NotImplementedError('the reference scan only supports a real A')
    if B.dim() not in (2, 3) or C.dim() not in (2, 3):
        raise NotImplementedError('the reference scan supports (dim, dstate) or '
                                  '(batch, dstate, length) for B and C')

    dtype_in = u.dtype
    work = _working_dtype(dtype_in)
    u = u.to(work)
    delta = delta.to(work)
    A = A.to(work)
    batch, dim, length = u.shape
    dstate = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.to(work)[:, None]
    if delta_softplus:
        delta = F.softplus(delta)

    B = B.to(work)
    C = C.to(work)
    if B.dim() == 3:
        deltaB_u = delta[:, :, None, :] * B[:, None, :, :] * u[:, :, None, :]
    else:
        deltaB_u = delta[:, :, None, :] * B[None, :, :, None] * u[:, :, None, :]

    # (batch, dim, dstate, length)
    delta_a = torch.exp(delta[:, :, None, :] * A[None, :, :, None])
    if negative:
        delta_a = 2.0 * delta_a - 1.0

    state = u.new_zeros((batch, dim, dstate))
    outputs = []
    for t in range(length):
        state = delta_a[:, :, :, t] * state + deltaB_u[:, :, :, t]
        if C.dim() == 3:
            outputs.append((state * C[:, None, :, t]).sum(dim=-1))
        else:
            outputs.append((state * C[None, :, :]).sum(dim=-1))
    out = torch.stack(outputs, dim=2)

    if D is not None:
        out = out + u * D.to(work)[None, :, None]
    if z is not None:
        out = out * F.silu(z.to(work))

    out = out.to(dtype_in)
    if return_last_state:
        return out, state
    return out
