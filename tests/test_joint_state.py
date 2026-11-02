"""CPU equation checks for the per-joint state-only Simba branch.

Run with PyTorch and einops installed:
    python -m unittest discover -s tests -p test_joint_state.py
Native Mamba/CUDA dependencies are isolated only while importing the source.
The real Mamba initializer and the new implementation are exercised.
"""

import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _source(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def isolated_subject():
    scan = types.ModuleType('mamba_ssm.ops.selective_scan_interface')

    def unavailable(*args, **kwargs):
        raise AssertionError('CPU branch must not call the native scan')

    scan.selective_scan_fn = unavailable
    scan.mamba_inner_fn = unavailable
    causal = types.ModuleType('causal_conv1d')
    causal.causal_conv1d_fn = None
    causal.causal_conv1d_update = None
    update = types.ModuleType('mamba_ssm.ops.triton.selective_state_update')
    update.selective_state_update = None
    norms = types.ModuleType('mamba_ssm.ops.triton.layernorm')
    norms.RMSNorm = None
    norms.layer_norm_fn = None
    norms.rms_norm_fn = None
    modules = {
        'mamba_ssm': _package('mamba_ssm'),
        'mamba_ssm.modules': _package('mamba_ssm.modules'),
        'mamba_ssm.ops': _package('mamba_ssm.ops'),
        'mamba_ssm.ops.triton': _package('mamba_ssm.ops.triton'),
        'mamba_ssm.ops.selective_scan_interface': scan,
        'mamba_ssm.ops.triton.selective_state_update': update,
        'mamba_ssm.ops.triton.layernorm': norms,
        'causal_conv1d': causal,
    }
    with mock.patch.dict(sys.modules, modules):
        _source('mamba_ssm.modules.mamba_simple', ROOT / 'mamba_ssm/modules/mamba_simple.py')
        module = _source('mamba_ssm.modules.mamba_joint_state',
                         ROOT / 'mamba_ssm/modules/mamba_joint_state.py')
    return module


subject = isolated_subject()
JointStateMamba = subject.JointStateMamba


def recurrence(u, delta, A, B, C):
    """Independent per-time formula, accepts already-positive delta."""
    h = torch.zeros(u.shape[0], u.shape[1], A.shape[1], dtype=u.dtype)
    results = []
    for t in range(u.shape[-1]):
        transition = torch.exp(torch.einsum('bd,dn->bdn', delta[:, :, t], A))
        write = torch.einsum('bd,bn,bd->bdn', delta[:, :, t], B[:, :, t], u[:, :, t])
        h = transition * h + write
        results.append(torch.einsum('bdn,bn->bd', h, C[:, :, t]))
    return torch.stack(results, dim=-1)


def one_joint_explicit(model, tokens):
    projected = F.linear(tokens, model.in_proj.weight).transpose(1, 2)
    u, z = projected.chunk(2, dim=1)
    u = F.silu(F.conv1d(u, model.conv1d.weight, model.conv1d.bias,
                       padding=model.d_conv - 1, groups=model.d_inner)[..., :tokens.shape[1]])
    coefficients = F.linear(u.transpose(1, 2), model.x_proj.weight)
    rank, B, C = coefficients.split((model.dt_rank, model.d_state, model.d_state), dim=-1)
    delta = F.softplus(F.linear(rank, model.dt_proj.weight, model.dt_proj.bias)).transpose(1, 2)
    A = -model.A_log.exp()
    readout = recurrence(u, delta, A, B.transpose(1, 2), C.transpose(1, 2))
    gated = readout * F.silu(z)
    return F.linear(gated.transpose(1, 2), model.out_proj.weight)


class JointStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def make_model(self):
        return JointStateMamba(num_joints=3, joint_channels=4,
                               d_state=3, d_conv=4, expand=2)

    def test_initializer_preserves_rng_and_has_no_d_parameter(self):
        torch.manual_seed(4321)
        before = torch.get_rng_state().clone()
        model = self.make_model()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(hasattr(model, 'D'))
        self.assertFalse(any(name == 'D' for name in model.state_dict()))
        self.assertEqual(model.d_model, 4)

    def test_output_matches_independent_joint_equations_and_gradients(self):
        model = self.make_model()
        inputs = torch.randn(2, 7, 12, requires_grad=True)
        expected = torch.cat([one_joint_explicit(model, inputs[..., index*4:(index+1)*4])
                              for index in range(3)], dim=-1)
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        expected_gradient = torch.autograd.grad(expected.square().sum(), inputs,
                                                retain_graph=True)[0]
        actual.square().sum().backward()
        torch.testing.assert_close(inputs.grad, expected_gradient, atol=2e-6, rtol=2e-5)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_no_cross_joint_or_cross_example_state(self):
        model = self.make_model().eval()
        inputs = torch.randn(3, 6, 12)
        original = model(inputs)
        changed = inputs.clone()
        changed[0, :, :4] += 1
        modified = model(changed)
        self.assertFalse(torch.equal(original[0, :, :4], modified[0, :, :4]))
        torch.testing.assert_close(original[0, :, 4:], modified[0, :, 4:], atol=0, rtol=0)
        torch.testing.assert_close(original[1:], modified[1:], atol=0, rtol=0)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(model(inputs[order])[torch.argsort(order)], original,
                                   atol=2e-7, rtol=2e-6)
        model(torch.randn_like(inputs))
        torch.testing.assert_close(model(inputs), original, atol=0, rtol=0)

    def test_causal_boundary_and_short_sequence(self):
        model = self.make_model()
        for length in (1, 2, 3):
            with self.subTest(length=length):
                inputs = torch.randn(2, length, 12)
                self.assertEqual(model(inputs).shape, inputs.shape)
        inputs = torch.randn(2, 8, 12)
        changed = inputs.clone()
        changed[:, 5:] += 2
        torch.testing.assert_close(model(inputs)[:, :5], model(changed)[:, :5], atol=0, rtol=0)

    def test_state_disable_zeroes_entire_branch_and_is_not_checkpointed(self):
        model = self.make_model()
        inputs = torch.randn(2, 5, 12)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        self.assertGreater(model(inputs).abs().sum().item(), 0)
        model.state_disabled = True
        self.assertTrue(torch.equal(model(inputs), torch.zeros_like(inputs)))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_probe_components_are_detached_only_when_cached(self):
        model = self.make_model()
        model.diagnostics_enabled = True
        result = model.probe_components(torch.randn(2, 5, 12, requires_grad=True))
        self.assertEqual(result['output'].shape, (2, 5, 12))
        self.assertEqual(result['local_encoded'].shape, (6, 8, 5))
        self.assertEqual(result['state_readout'].shape, (6, 8, 5))
        self.assertTrue(result['output'].requires_grad)
        self.assertTrue(all(not value.requires_grad for value in model.last_diagnostics.values()))
        self.assertFalse(any(isinstance(value, torch.Tensor) and value.requires_grad
                             for name, value in vars(model).items()))

    def test_state_dict_roundtrip(self):
        model = self.make_model().eval()
        inputs = torch.randn(2, 5, 12)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = self.make_model().eval()
        restored.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
        torch.testing.assert_close(restored(inputs), model(inputs), atol=0, rtol=0)

    def test_reference_scan_double_precision_gradient_check(self):
        tensors = (
            torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True),
            -torch.rand(2, 2, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 2, 3, dtype=torch.float64, requires_grad=True),
        )
        self.assertTrue(torch.autograd.gradcheck(subject.joint_state_scan_ref, tensors,
                                                eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_invalid_inputs_and_cached_decoding_are_rejected(self):
        for name in ('num_joints', 'joint_channels', 'd_state', 'd_conv', 'expand'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                JointStateMamba(**{name: 0})
        model = self.make_model()
        for inputs in (torch.randn(2, 4, 11), torch.randn(2, 0, 12), torch.randn(2, 12)):
            with self.assertRaises(ValueError):
                model(inputs)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 12), inference_params=object())


if __name__ == '__main__':
    unittest.main()
