"""CPU-only finite-difference check of the backward formula used by the CUDA kernel.

This runs without a GPU and without PyTorch. It implements the recurrence

    h[t] = a[t] * h[t-1] + delta[t] * u[t] * B[t],      y[t] = C[t] . h[t]

twice: once as a plain forward pass, and once with the analytic gradients that
``selective_scan_bwd_kernel`` computes. Central differences then decide whether the
analytic backward is right, for both

    a[t] = exp(delta[t] * A)                  (baseline)
    a[t] = 2 * exp(delta[t] * A) - 1          (negative eigenvalues)

The interesting line is the coefficient of the delta/A gradients: the kernel forms
``h[t] - write[t] == a[t] * h[t-1]`` and, in negative mode, adds ``h[t-1]`` so the
coefficient becomes ``2 * exp(delta * A) * h[t-1]``. Dividing by ``a[t]`` instead would
be wrong, because ``a[t]`` can be arbitrarily close to zero.

    python tools/check_backward_derivation.py
    python tools/check_backward_derivation.py --seed 7 --length 11
"""

import argparse

import numpy as np


def transition(delta, decay, negative):
    q = np.exp(delta[:, :, None, :] * decay[None, :, :, None])  # (B, D, N, L)
    return 2.0 * q - 1.0 if negative else q


def forward(decay, delta, u, B, C, negative):
    a = transition(delta, decay, negative)
    write = delta[:, :, None, :] * u[:, :, None, :] * B[:, None, :, :]
    state = np.zeros((delta.shape[0], delta.shape[1], decay.shape[1]))
    outputs, states = [], []
    for t in range(delta.shape[2]):
        state = a[:, :, :, t] * state + write[:, :, :, t]
        states.append(state.copy())
        outputs.append((state * C[:, None, :, t]).sum(-1))
    return np.stack(outputs, 2), np.stack(states, 3), a


def analytic(decay, delta, u, B, C, weight, negative):
    y, hs, a = forward(decay, delta, u, B, C, negative)
    length = delta.shape[2]
    adjoint = np.zeros_like(hs[:, :, :, 0])
    du = np.zeros_like(u)
    ddelta = np.zeros_like(delta)
    ddecay = np.zeros_like(decay)
    dB = np.zeros_like(B)
    dC = np.zeros_like(C)
    for t in range(length - 1, -1, -1):
        # dL/dh[t] = w[t] C[t] + a[t+1] * dL/dh[t+1]
        step = t + 1 if t + 1 < length else t
        adjoint = adjoint * a[:, :, :, step] + weight[:, :, t][:, :, None] * C[:, None, :, t]
        previous = np.zeros_like(hs[:, :, :, 0]) if t == 0 else hs[:, :, :, t - 1]
        write = delta[:, :, None, t] * u[:, :, None, t] * B[:, None, :, t]
        # (transition * h[t-1]) is what the kernel already has; negative mode needs 2q h[t-1].
        coefficient = (hs[:, :, :, t] - write) + previous if negative else (hs[:, :, :, t] - write)
        du[:, :, t] = (adjoint * B[:, None, :, t]).sum(-1) * delta[:, :, t]
        ddelta[:, :, t] = ((adjoint * B[:, None, :, t] * u[:, :, None, t]).sum(-1)
                           + (adjoint * decay[None] * coefficient).sum(-1))
        ddecay += (adjoint * delta[:, :, None, t] * coefficient).sum(0)
        dB[:, :, t] = (adjoint * delta[:, :, None, t] * u[:, :, None, t]).sum(1)
        dC[:, :, t] = (weight[:, :, t][:, :, None] * hs[:, :, :, t]).sum(1)
    return du, ddelta, ddecay, dB, dC


def numeric(tensor, loss, eps=1e-6):
    gradient = np.zeros_like(tensor)
    for index in np.ndindex(*tensor.shape):
        original = tensor[index]
        tensor[index] = original + eps
        high = loss()
        tensor[index] = original - eps
        low = loss()
        tensor[index] = original
        gradient[index] = (high - low) / (2 * eps)
    return gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--states', type=int, default=4)
    parser.add_argument('--length', type=int, default=7)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    decay = -(rng.random((args.dim, args.states)) * 2.0 + 0.05)
    delta = rng.random((args.batch, args.dim, args.length)) * 1.5 + 0.01
    u = rng.standard_normal((args.batch, args.dim, args.length))
    B = rng.standard_normal((args.batch, args.states, args.length))
    C = rng.standard_normal((args.batch, args.states, args.length))
    weight = rng.standard_normal((args.batch, args.dim, args.length))

    failures = 0
    for negative in (False, True):
        def loss():
            return float((forward(decay, delta, u, B, C, negative)[0] * weight).sum())

        expected = analytic(decay, delta, u, B, C, weight, negative)
        actual = (numeric(u, loss), numeric(delta, loss), numeric(decay, loss),
                  numeric(B, loss), numeric(C, loss))
        tag = 'negative' if negative else 'baseline'
        for name, analytic_grad, numeric_grad in zip(('du', 'ddelta', 'dA', 'dB', 'dC'),
                                                     expected, actual):
            scale = max(1e-8, float(np.abs(numeric_grad).max()))
            error = float(np.abs(analytic_grad - numeric_grad).max()) / scale
            status = 'ok' if error < args.tolerance else 'MISMATCH'
            print(f'{tag:9s} {name:7s} max relative error {error:.3e}  {status}')
            failures += status != 'ok'
        transitions = transition(delta, decay, negative)
        print(f'{tag:9s} transition range [{transitions.min():.4f}, {transitions.max():.4f}], '
              f'negative share {(transitions < 0).mean():.3f}')

    if failures:
        raise SystemExit(f'{failures} gradient check(s) failed')
    print('\nanalytic backward matches central differences for both modes')


if __name__ == '__main__':
    main()
