"""CPU equation and recovery checks for B/C normalization inside Mamba.

Uses the real source and an independent differentiable scan, with only native
CUDA dependencies replaced. Run with PyTorch, einops and PyYAML installed:
    python -m unittest discover -s tests -p test_bc_norm.py -v
"""
import importlib.util
import io
import math
from pathlib import Path
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('bc_scan_reference', ROOT / 'tests/test_ssm_modules.py')
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
BaseMamba, subject = reference._isolated_sources()
SkeletonMamba = subject.SkeletonMamba
BC_NAMES = ('bc_B_gain', 'bc_C_gain', 'bc_B_bias', 'bc_C_bias')


def normalized(values, gain, bias, eps):
    """State-vector RMS equation, written separately from the implementation."""
    value = values.float()
    denominator = (torch.linalg.vector_norm(value, dim=-1, keepdim=True).square()
                   / value.shape[-1] + eps).sqrt()
    return (value / denominator * gain.float() + bias.float()).to(values.dtype)


def equations(model, inputs):
    projected = F.linear(inputs, model.in_proj.weight, model.in_proj.bias).transpose(1, 2)
    u, z = projected.chunk(2, dim=1)
    u = F.silu(F.conv1d(u, model.conv1d.weight, model.conv1d.bias,
                       groups=model.d_inner, padding=model.d_conv - 1)[..., :inputs.shape[1]])
    params = F.linear(u.transpose(1, 2), model.x_proj.weight)
    rank, B, C = params.split((model.dt_rank, model.d_state, model.d_state), dim=-1)
    B = normalized(B, model.bc_B_gain, model.bc_B_bias, model.bc_norm_eps)
    C = normalized(C, model.bc_C_gain, model.bc_C_bias, model.bc_norm_eps)
    delta = F.linear(rank, model.dt_proj.weight).transpose(1, 2)
    scanned = reference.reference_scan(
        u, delta, -model.A_log.float().exp(), B.transpose(1, 2), C.transpose(1, 2),
        model.D.float(), z=z, delta_bias=model.dt_proj.bias.float(), delta_softplus=True)
    return F.linear(scanned.transpose(1, 2), model.out_proj.weight, model.out_proj.bias)


class BCNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4321)

    def make_model(self, **options):
        args = dict(d_model=12, d_state=4, d_conv=4, dt_rank=3,
                    expand=2, use_fast_path=True, bc_norm=True)
        args.update(options)
        return SkeletonMamba(**args)

    def close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-4)

    def test_equations_outputs_input_and_parameter_gradients(self):
        model = self.make_model()
        inputs = torch.randn(2, 7, 12, requires_grad=True)
        expected, actual = equations(model, inputs), model(inputs)
        self.close(actual, expected)
        parameters = list(model.parameters())
        expected_grad = torch.autograd.grad(expected.square().sum(), [inputs] + parameters)
        actual_grad = torch.autograd.grad(actual.square().sum(), [inputs] + parameters)
        for name, a, b in zip(['input'] + [name for name, _ in model.named_parameters()],
                              actual_grad, expected_grad):
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(a).all())
                self.close(a, b)
        for name in BC_NAMES:
            index = next(i for i, (key, _) in enumerate(model.named_parameters()) if key == name)
            self.assertGreater(actual_grad[index + 1].abs().sum().item(), 0, name)

    def test_one_scan_and_state_axis_affine_even_when_fast_path_enabled(self):
        model = self.make_model()
        with torch.no_grad():
            model.bc_B_gain.copy_(torch.tensor([.04, .08, .12, .16]))
            model.bc_C_gain.copy_(torch.tensor([.03, .07, .11, .19]))
            model.bc_B_bias.copy_(torch.tensor([-.1, 0., .1, .2]))
            model.bc_C_bias.copy_(torch.tensor([.2, .1, 0., -.1]))
        generated, calls = [], []
        handle = model.x_proj.register_forward_hook(
            lambda module, args, output: generated.append(output.detach().clone()))

        def capture(*args, **kwargs):
            calls.append((args, kwargs))
            return reference.reference_scan(*args, **kwargs)

        data = torch.randn(2, 5, 12)
        data[1] *= 4
        try:
            with mock.patch.object(subject, 'selective_scan_fn', side_effect=capture):
                output = model(data)
        finally:
            handle.remove()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(len(calls), 1)
        _, raw_B, raw_C = generated[0].split((3, 4, 4), dim=-1)
        expected_B = normalized(raw_B, model.bc_B_gain, model.bc_B_bias, model.bc_norm_eps)
        expected_C = normalized(raw_C, model.bc_C_gain, model.bc_C_bias, model.bc_norm_eps)
        expected_B = expected_B.reshape(2, 5, 4).transpose(1, 2)
        expected_C = expected_C.reshape(2, 5, 4).transpose(1, 2)
        args, kwargs = calls[0]
        self.close(args[3], expected_B)
        self.close(args[4], expected_C)
        self.assertIsNotNone(args[5])
        self.assertIsNotNone(kwargs['z'])
        self.assertTrue(kwargs['delta_softplus'])

    def test_new_parameters_are_constant_and_do_not_consume_rng(self):
        args = dict(d_model=12, d_state=16, dt_rank=3, expand=2, d_conv=4)
        torch.manual_seed(9)
        baseline = BaseMamba(**args)
        next_expected = torch.rand(7)
        torch.manual_seed(9)
        model = SkeletonMamba(**args, bc_norm=True)
        next_actual = torch.rand(7)
        self.assertTrue(torch.equal(next_expected, next_actual))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        added = {name: value for name, value in model.named_parameters() if name in BC_NAMES}
        self.assertEqual(set(added), set(BC_NAMES))
        self.assertEqual(sum(value.numel() for value in added.values()), 64)
        for name, value in added.items():
            torch.testing.assert_close(value, torch.full_like(value, .1), atol=0, rtol=0)

    def test_normalization_uses_fp32_for_small_and_large_low_precision_values(self):
        model = self.make_model()
        # Squaring this row in FP16 would overflow; the zero/tiny rows check
        # epsilon placement and the learned offset without a nonzero RMS.
        values = torch.tensor([[10000., -10000., 20000., -20000.],
                               [0., 0., 0., 0.],
                               [1e-5, -1e-5, 2e-5, -2e-5]],
                              dtype=torch.float16, requires_grad=True)
        actual = model._normalize_bc(values, model.bc_B_gain, model.bc_B_bias)
        expected = normalized(values, model.bc_B_gain, model.bc_B_bias, model.bc_norm_eps)
        self.assertEqual(actual.dtype, values.dtype)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)
        self.assertTrue(torch.equal(actual[1], model.bc_B_bias.to(values.dtype)))
        actual.float().square().sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertTrue(torch.isfinite(model.bc_B_gain.grad).all())
        self.assertTrue(torch.isfinite(model.bc_B_bias.grad).all())

    def test_disabled_preserves_original_checkpoint_keys_and_delegation(self):
        model = self.make_model(bc_norm=False, use_fast_path=False)
        baseline = BaseMamba(d_model=12, d_state=4, d_conv=4, dt_rank=3,
                             expand=2, use_fast_path=False)
        baseline.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
        data = torch.randn(2, 4, 12)
        self.close(model(data), baseline(data))
        sentinel = object()
        with mock.patch.object(BaseMamba, 'forward', return_value=sentinel) as call:
            self.assertIs(model(data), sentinel)
        call.assert_called_once()

    def test_short_sequences_checkpoint_roundtrip_and_optimizer_step(self):
        for length in (1, 2, 3):
            with self.subTest(length=length):
                model = self.make_model()
                data = torch.randn(2, length, 12)
                optimizer = torch.optim.SGD(model.parameters(), lr=.05)
                before = {name: getattr(model, name).detach().clone() for name in BC_NAMES}
                model(data).square().sum().backward()
                optimizer.step()
                for name in BC_NAMES:
                    self.assertTrue(torch.isfinite(getattr(model, name)).all())
                    self.assertFalse(torch.equal(before[name], getattr(model, name)), name)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = self.make_model()
                restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
                self.close(restored(data), model(data))

    def test_bc_path_rejects_cached_decoding_and_invalid_inputs(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 12), inference_params=object())
        with self.assertRaises(ValueError):
            model.step(torch.randn(1, 1, 12), None, None)
        for shape in ((2, 0, 12), (2, 3, 11), (2, 12)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))

    def test_invalid_bc_hyperparameters_are_rejected(self):
        for name in ('bc_norm_gain', 'bc_norm_eps'):
            for value in (0., -.1, math.nan, math.inf):
                with self.subTest(name=name, value=value), self.assertRaises((ValueError, TypeError)):
                    self.make_model(**{name: value})
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(bias=value), self.assertRaises((ValueError, TypeError)):
                self.make_model(bc_norm_bias=value)
        for value in (0, 1, None, 'yes'):
            with self.subTest(bc_norm=value), self.assertRaises((ValueError, TypeError)):
                self.make_model(bc_norm=value)
        for name in ('bc_norm_gain', 'bc_norm_bias', 'bc_norm_eps'):
            with self.subTest(name=name, value=True), self.assertRaises((ValueError, TypeError)):
                self.make_model(**{name: True})


if __name__ == '__main__':
    unittest.main()
