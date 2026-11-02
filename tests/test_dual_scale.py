"""CPU equation/gradient tests; no native CUDA extension is needed.

Run: python -m unittest discover -s tests -p test_dual_scale.py
Requires torch and einops in the experiment environment.
"""

import importlib.util
import io
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    original_dtype = u.dtype
    u, delta, A, B, C = [value.float() for value in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        dt = delta[:, :, t, None]
        state = (dt * A[None]).exp() * state + dt * B[:, None, :, t] * u[:, :, t, None]
        outputs.append((state * C[:, None, :, t]).sum(dim=-1))
    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + D[None, :, None] * u
    if z is not None:
        output = output * F.silu(z)
    output = output.to(original_dtype)
    return (output, state) if return_last_state else output


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def isolated_sources():
    scan = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    scan.selective_scan_fn = reference_scan
    scan.mamba_inner_fn = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('native fused path'))
    causal = types.ModuleType('causal_conv1d')
    causal.causal_conv1d_fn = None
    causal.causal_conv1d_update = None
    update = types.ModuleType('mamba_ssm.ops.triton.selective_state_update')
    update.selective_state_update = None
    norms = types.ModuleType('mamba_ssm.ops.triton.layernorm')
    norms.RMSNorm = norms.layer_norm_fn = norms.rms_norm_fn = None
    modules = {
        'mamba_ssm': package('mamba_ssm'),
        'mamba_ssm.modules': package('mamba_ssm.modules'),
        'mamba_ssm.ops': package('mamba_ssm.ops'),
        'mamba_ssm.ops.triton': package('mamba_ssm.ops.triton'),
        'mamba_ssm.ops.selective_scan_interface': scan,
        'mamba_ssm.ops.triton.selective_state_update': update,
        'mamba_ssm.ops.triton.layernorm': norms,
        'causal_conv1d': causal,
    }
    with mock.patch.dict(sys.modules, modules):
        base = load_source('mamba_ssm.modules.mamba_simple', ROOT / 'mamba_ssm/modules/mamba_simple.py')
        subject = load_source('mamba_ssm.modules.mamba_dual_scale', ROOT / 'mamba_ssm/modules/mamba_dual_scale.py')
    return base.Mamba, subject


BaseMamba, subject = isolated_sources()
DualScaleMamba = subject.DualScaleMamba


def explicit_output(model, inputs, state_enabled=True):
    # Separate wiring expression; do not use the subject's helpers.
    xz = F.linear(inputs, model.in_proj.weight, model.in_proj.bias).transpose(1, 2)
    u, z = xz.chunk(2, dim=1)
    u = F.silu(F.conv1d(u, model.conv1d.weight, model.conv1d.bias,
                       padding=model.d_conv - 1, groups=model.d_inner)[..., :inputs.shape[1]])
    params = F.linear(u.transpose(1, 2), model.x_proj.weight)
    rank, B, C = torch.split(params, (model.dt_rank, model.d_state, model.d_state), dim=-1)
    delta = F.softplus(F.linear(rank, model.dt_proj.weight, model.dt_proj.bias)).transpose(1, 2)
    A = -model.A_log.float().exp()
    base = reference_scan(u, delta, A, B.transpose(1, 2), C.transpose(1, 2))
    slow = reference_scan(u, delta * model.slow_scale, A, B.transpose(1, 2), C.transpose(1, 2))
    gate = model.dual_scale_logit.sigmoid()[None, :, None]
    state = (1 - gate) * base + gate * slow
    if not state_enabled:
        state = torch.zeros_like(state)
    mixed = (state + model.D[None, :, None] * u) * F.silu(z)
    return F.linear(mixed.transpose(1, 2), model.out_proj.weight, model.out_proj.bias)


class DualScaleTest(unittest.TestCase):
    def make(self, **kwargs):
        return DualScaleMamba(d_model=12, d_state=4, d_conv=4, expand=2,
                              dt_rank=3, use_fast_path=False, **kwargs)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)

    def test_common_initialization_rng_and_no_decay_gate(self):
        torch.manual_seed(17)
        baseline = BaseMamba(d_model=12, d_state=4, d_conv=4, expand=2,
                             dt_rank=3, use_fast_path=False)
        expected_next = torch.rand(8)
        torch.manual_seed(17)
        model = self.make()
        actual_next = torch.rand(8)
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertTrue(torch.equal(expected_next, actual_next))
        self.assertTrue(model.dual_scale_logit._no_weight_decay)
        self.assert_close(model.dual_scale_logit.sigmoid(), torch.full_like(model.dual_scale_logit, 0.1))

    def test_equations_and_input_parameter_gradients(self):
        torch.manual_seed(31)
        model = self.make()
        inputs = torch.randn(2, 7, 12, requires_grad=True)
        actual, expected = model(inputs), explicit_output(model, inputs)
        self.assert_close(actual, expected)
        expected_grad = torch.autograd.grad(expected.square().sum(), inputs, retain_graph=True)[0]
        actual.square().sum().backward()
        self.assert_close(inputs.grad, expected_grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.dual_scale_logit.grad.abs().sum().item(), 0)

    def test_positive_delta_scaling_retention_and_kernel_arguments(self):
        model = self.make()
        calls = []
        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return reference_scan(*args, **kwargs)
        with mock.patch.object(subject, 'selective_scan_fn', side_effect=record):
            components = model.probe_components(torch.randn(2, 6, 12))
        self.assertEqual(len(calls), 2)
        self.assertTrue((components['delta_base'] > 0).all())
        self.assert_close(components['delta_slow'], components['delta_base'] * 0.5)
        self.assert_close(components['retention_slow'], components['retention_base'].sqrt())
        self.assertTrue((components['retention_slow'] >= components['retention_base']).all())
        for args, kwargs in calls:
            self.assertIsNone(kwargs['D'])
            self.assertIsNone(kwargs['z'])
            self.assertIsNone(kwargs['delta_bias'])
            self.assertFalse(kwargs['delta_softplus'])

    def test_scale_one_and_zero_gate_recover_baseline(self):
        for options in ({'slow_scale': 1.0}, {}):
            model = self.make(**options)
            if not options:
                with torch.no_grad():
                    model.dual_scale_logit.fill_(-100)
            baseline = BaseMamba(d_model=12, d_state=4, d_conv=4, expand=2,
                                 dt_rank=3, use_fast_path=False)
            baseline.load_state_dict({name: model.state_dict()[name] for name in baseline.state_dict()})
            x1 = torch.randn(2, 5, 12, requires_grad=True)
            x2 = x1.detach().clone().requires_grad_()
            y1, y2 = model(x1), baseline(x2)
            self.assert_close(y1, y2)
            y1.square().sum().backward()
            y2.square().sum().backward()
            self.assert_close(x1.grad, x2.grad)

    def test_runtime_disable_uses_vanilla_forward(self):
        model = self.make()
        model.dual_scale_enabled = False
        sentinel = object()
        with mock.patch.object(BaseMamba, 'forward', return_value=sentinel) as call:
            self.assertIs(model(torch.randn(1, 4, 12)), sentinel)
        call.assert_called_once()

    def test_correction_off_retains_explicit_path_for_noise_calibration(self):
        model = self.make()
        model.correction_enabled = False
        inputs = torch.randn(2, 6, 12)
        # Unlike dual_scale_enabled=False, this must not call vanilla/fused.
        with mock.patch.object(BaseMamba, 'forward', side_effect=AssertionError('vanilla path')):
            components = model.probe_components(inputs)
            self.assert_close(model(inputs), components['output_base'])
        self.assertTrue(torch.equal(components['state_correction'],
                                    torch.zeros_like(components['state_correction'])))
        self.assertNotIn('correction_enabled', model.state_dict())

    def test_state_off_retains_d_and_z_once(self):
        model = self.make()
        model.state_enabled = False
        inputs = torch.randn(2, 5, 12)
        self.assert_close(model(inputs), explicit_output(model, inputs, state_enabled=False))
        # Removing both B and C leaves precisely the same D-only output.
        with torch.no_grad():
            model.x_proj.weight[model.dt_rank:].zero_()
        model.state_enabled = True
        self.assert_close(model(inputs), explicit_output(model, inputs, state_enabled=False))

    def test_independent_clip_state_and_checkpoint_roundtrip(self):
        model = self.make()
        model.eval()
        for length in (1, 2, 3, 8):
            inputs = torch.randn(3, length, 12)
            output = model(inputs)
            self.assertEqual(output.shape, inputs.shape)
            self.assert_close(output, explicit_output(model, inputs))
            permutation = torch.tensor([2, 0, 1])
            self.assert_close(model(inputs[permutation]), output[permutation])
            self.assert_close(model(inputs[:1]), output[:1])
        memory = io.BytesIO()
        torch.save(model.state_dict(), memory)
        memory.seek(0)
        restored = self.make()
        restored.load_state_dict(torch.load(memory, map_location='cpu', weights_only=True))
        self.assert_close(restored(inputs), model(inputs))

    def test_validation_and_cache_rejection(self):
        for value in (True, 0, -0.5, math.nan, math.inf, 1.1):
            with self.subTest(scale=value), self.assertRaises((ValueError, TypeError)):
                self.make(slow_scale=value)
        for value in (True, 0, 1, math.nan, math.inf):
            with self.subTest(gate=value), self.assertRaises((ValueError, TypeError)):
                self.make(gate_init=value)
        model = self.make()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 5, 12), inference_params=object())
        with self.assertRaises(ValueError):
            model.step(torch.randn(1, 1, 12), None, None)


if __name__ == '__main__':
    unittest.main()
