"""CPU tests for the negative-eigenvalue scan reference and the CUDA patch surface.

They run without a GPU, without the compiled extension and without NTU data:

    python -m unittest discover -s tests -t .

The numerical operator is checked against an explicit recurrence, and the compiled halves
are checked for the presence of the branch so a partial revert cannot go unnoticed.
"""

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_scan_reference():
    """Import mamba_ssm/ops/scan_reference.py without importing the mamba_ssm package.

    The package __init__ imports the compiled extension, which does not exist on a CPU-only
    machine, and the reference deliberately depends on torch only.
    """
    path = ROOT / 'mamba_ssm' / 'ops' / 'scan_reference.py'
    spec = importlib.util.spec_from_file_location('scan_reference_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan_reference = load_scan_reference()


def manual_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                negative=False):
    """Textbook recurrence, written independently of the module under test."""
    dtype_in = u.dtype
    u, delta, A = u.double(), delta.double(), A.double()
    B, C = B.double(), C.double()
    if delta_bias is not None:
        delta = delta + delta_bias.double()[:, None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    batch, dim, length = u.shape
    dstate = A.shape[1]
    state = torch.zeros(batch, dim, dstate, dtype=torch.float64)
    outputs = []
    for t in range(length):
        transition = torch.exp(delta[:, :, t].unsqueeze(-1) * A[None])
        if negative:
            transition = 2.0 * transition - 1.0
        delta_t = delta[:, :, t].unsqueeze(-1)
        u_t = u[:, :, t].unsqueeze(-1)
        if B.dim() == 3:
            write = delta_t * u_t * B[:, :, t][:, None, :]
        else:
            write = delta_t * u_t * B[None]
        state = transition * state + write
        if C.dim() == 3:
            outputs.append((state * C[:, :, t][:, None, :]).sum(dim=-1))
        else:
            outputs.append((state * C[None]).sum(dim=-1))
    out = torch.stack(outputs, dim=2)
    if D is not None:
        out = out + u * D.double()[None, :, None]
    if z is not None:
        out = out * torch.nn.functional.silu(z.double())
    return out.to(dtype_in)


def sample_inputs(batch=2, dim=3, dstate=2, length=6, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, dim, length, generator=generator, dtype=dtype)
    delta = torch.rand(batch, dim, length, generator=generator, dtype=dtype)
    A = -torch.rand(dim, dstate, generator=generator, dtype=dtype) * 2.0
    B = torch.randn(batch, dstate, length, generator=generator, dtype=dtype)
    C = torch.randn(batch, dstate, length, generator=generator, dtype=dtype)
    D = torch.randn(dim, generator=generator, dtype=dtype)
    z = torch.randn(batch, dim, length, generator=generator, dtype=dtype)
    delta_bias = torch.randn(dim, generator=generator, dtype=dtype) * 0.1
    return u, delta, A, B, C, D, z, delta_bias


class ReferenceMatchesRecurrence(unittest.TestCase):

    def test_both_modes(self):
        inputs = sample_inputs()
        for negative in (False, True):
            with self.subTest(negative=negative):
                expected = manual_scan(*inputs, delta_softplus=True, negative=negative)
                actual = scan_reference.selective_scan_reference(
                    *inputs, delta_softplus=True, negative=negative)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)

    def test_baseline_ignores_the_flag_default(self):
        inputs = sample_inputs()
        explicit = scan_reference.selective_scan_reference(*inputs, delta_softplus=True,
                                                           negative=False)
        implicit = scan_reference.selective_scan_reference(*inputs, delta_softplus=True)
        self.assertTrue(torch.equal(explicit, implicit))

    def test_negative_changes_the_output(self):
        inputs = sample_inputs()
        baseline = scan_reference.selective_scan_reference(*inputs, delta_softplus=True)
        negative = scan_reference.selective_scan_reference(*inputs, delta_softplus=True,
                                                           negative=True)
        self.assertGreater(float((baseline - negative).abs().max()), 1e-3)

    def test_constant_b_produces_the_same_result(self):
        u, delta, A, _, _, D, z, delta_bias = sample_inputs()
        Bconst = torch.randn(A.shape[0], A.shape[1], dtype=torch.float64)
        Cconst = torch.randn(A.shape[0], A.shape[1], dtype=torch.float64)
        expected = manual_scan(u, delta, A, Bconst, Cconst, D=D, z=z,
                               delta_bias=delta_bias, delta_softplus=True, negative=True)
        actual = scan_reference.selective_scan_reference(
            u, delta, A, Bconst, Cconst, D=D, z=z, delta_bias=delta_bias,
            delta_softplus=True, negative=True)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)


class TransitionBounds(unittest.TestCase):

    def test_negative_transition_is_inside_the_open_unit_interval(self):
        delta = torch.linspace(0.0, 5.0, 501, dtype=torch.float64)
        decay = -torch.cat([torch.linspace(0.001, 5.0, 250, dtype=torch.float64),
                            torch.tensor([0.0, 1e-4], dtype=torch.float64)])
        transitions = 2.0 * torch.exp(delta[:, None] * decay[None, :]) - 1.0
        self.assertLess(float(transitions.max()), 1.0)
        self.assertGreaterEqual(float(transitions.min()), -1.0)
        # It must actually reach the negative half for a plausible discretisation.
        self.assertLess(float(transitions.min()), -0.5)

    def test_baseline_transition_stays_positive(self):
        delta = torch.linspace(0.0, 5.0, 101, dtype=torch.float64)
        decay = -torch.linspace(0.001, 5.0, 101, dtype=torch.float64)
        transitions = torch.exp(delta[:, None] * decay[None, :])
        self.assertGreater(float(transitions.min()), 0.0)
        self.assertLessEqual(float(transitions.max()), 1.0)


class ReferenceGradients(unittest.TestCase):

    def test_gradcheck_both_modes(self):
        u, delta, A, B, C, D, z, delta_bias = sample_inputs(batch=2, dim=3, dstate=2, length=5)
        for negative in (False, True):
            with self.subTest(negative=negative):
                leaves = tuple(t.clone().requires_grad_(True)
                               for t in (u, delta, A, B, C, D, z, delta_bias))
                self.assertTrue(torch.autograd.gradcheck(
                    lambda *args: scan_reference.selective_scan_reference(
                        *args, delta_softplus=True, negative=negative),
                    leaves, eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_complex_a_is_rejected(self):
        u, delta, _, B, C, _, _, _ = sample_inputs()
        A = torch.zeros(3, 2, dtype=torch.complex64)
        with self.assertRaises(NotImplementedError):
            scan_reference.selective_scan_reference(u, delta, A, B, C, negative=True)


class CompiledBranchPresent(unittest.TestCase):
    """Cheap guard that the CUDA and binding halves of the patch are both in place."""

    def test_forward_kernel_carries_the_transition(self):
        source = (ROOT / 'csrc' / 'selective_scan' / 'selective_scan_fwd_kernel.cuh').read_text(
            encoding='utf-8')
        self.assertIn('2.f * delta_a_exp - 1.f', source)
        self.assertIn('params.negative', source)

    def test_backward_kernel_carries_the_gradient(self):
        source = (ROOT / 'csrc' / 'selective_scan' / 'selective_scan_bwd_kernel.cuh').read_text(
            encoding='utf-8')
        self.assertIn('2.f * delta_a_exp - 1.f', source)
        self.assertIn('a_coef += h_prev;', source)
        self.assertIn('smem_h_prev', source)

    def test_binding_exposes_the_flag(self):
        source = (ROOT / 'csrc' / 'selective_scan' / 'selective_scan.cpp').read_text(
            encoding='utf-8')
        self.assertIn('params.negative = negative;', source)
        self.assertIn('negative_scan_supported', source)
        header = (ROOT / 'csrc' / 'selective_scan' / 'selective_scan.h').read_text(
            encoding='utf-8')
        self.assertIn('bool negative;', header)


if __name__ == '__main__':
    unittest.main()
