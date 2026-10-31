"""CPU mathematical checks for the optional skeleton SSM mechanisms.

Run with PyTorch and einops installed (CUDA extensions are not required):
    python -m unittest discover -s tests -p test_ssm_modules.py

Only native scan/convolution dependencies are replaced during isolated source
loading. The real Mamba initializer and SkeletonMamba implementation are used.
No mocked modules remain registered after loading.
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
from torch import nn
from torch.nn import functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    """Independent, differentiable recurrence; variable B/C, one group."""
    original_dtype = u.dtype
    u, delta, A, B, C = [value.float() for value in (u, delta, A, B, C)]
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    values = []
    for t in range(u.shape[-1]):
        step = delta[:, :, t, None]
        state = (step * A[None]).exp() * state
        state = state + step * B[:, None, :, t] * u[:, :, t, None]
        values.append((state * C[:, None, :, t]).sum(dim=-1))
    output = torch.stack(values, dim=-1)
    if D is not None:
        output = output + D[None, :, None] * u
    if z is not None:
        output = output * F.silu(z)
    output = output.to(original_dtype)
    return (output, state) if return_last_state else output


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _isolated_sources():
    scan = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    scan.selective_scan_fn = reference_scan

    def fused_is_not_available(*args, **kwargs):
        raise AssertionError('CPU test unexpectedly entered native fused Mamba')

    scan.mamba_inner_fn = fused_is_not_available
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
        base = _load_source('mamba_ssm.modules.mamba_simple',
                            ROOT / 'mamba_ssm/modules/mamba_simple.py')
        subject = _load_source('mamba_ssm.modules.mamba_skeleton',
                               ROOT / 'mamba_ssm/modules/mamba_skeleton.py')
    return base.Mamba, subject


BaseMamba, subject = _isolated_sources()
SkeletonMamba = subject.SkeletonMamba


def explicit_output(model, inputs):
    """Express the intended wiring independently of SkeletonMamba.forward."""
    projected = F.linear(inputs, model.in_proj.weight,
                         model.in_proj.bias).transpose(1, 2)
    u, z = projected.chunk(2, dim=1)
    u = F.silu(F.conv1d(u, model.conv1d.weight, model.conv1d.bias,
                       padding=model.d_conv - 1,
                       groups=model.d_inner)[..., :inputs.shape[1]])
    parameters = F.linear(u.transpose(1, 2), model.x_proj.weight)
    rank, B, C = torch.split(parameters,
                             (model.dt_rank, model.d_state, model.d_state),
                             dim=-1)
    if model.motion_delta:
        tokens = u.transpose(1, 2)
        difference = torch.cat((torch.zeros_like(tokens[:, :1]),
                                tokens[:, 1:] - tokens[:, :-1]), dim=1)
        correction = model.motion_up(F.silu(model.motion_down(
            model.motion_norm(difference))))
        rank = rank + model.motion_scale * torch.tanh(correction)
    delta = F.linear(rank, model.dt_proj.weight).transpose(1, 2)
    B, C = B.transpose(1, 2), C.transpose(1, 2)
    A = -model.A_log.float().exp()
    forward = reference_scan(u, delta, A, B, C,
                             delta_bias=model.dt_proj.bias,
                             delta_softplus=True)
    if model.bidirectional:
        backward = reference_scan(
            u.flip(-1), delta.flip(-1), A, B.flip(-1), C.flip(-1),
            delta_bias=model.dt_proj.bias, delta_softplus=True).flip(-1)
        gate = model.bidir_logit.sigmoid().reshape(1, model.d_inner, 1)
        forward = (1 - gate) * forward + gate * backward
    output = (forward + model.D[None, :, None] * u) * F.silu(z)
    return F.linear(output.transpose(1, 2), model.out_proj.weight,
                    model.out_proj.bias)


class SkeletonSSMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def make_model(self, **options):
        return SkeletonMamba(d_model=12, d_state=4, d_conv=4,
                             dt_rank=3, expand=2, use_fast_path=False,
                             motion_rank=5, **options)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_disabled_is_original_forward_and_checkpoint_compatible(self):
        model = self.make_model()
        baseline = BaseMamba(d_model=12, d_state=4, d_conv=4,
                             dt_rank=3, expand=2, use_fast_path=False)
        baseline.load_state_dict(model.state_dict(), strict=True)
        inputs = torch.randn(2, 7, 12)
        self.assert_close(model(inputs), baseline(inputs))
        sentinel = object()
        with mock.patch.object(BaseMamba, 'forward', return_value=sentinel) as call:
            self.assertIs(model(inputs), sentinel)
        call.assert_called_once()

    def test_new_parameters_do_not_change_shared_initialization_or_rng(self):
        torch.manual_seed(123)
        baseline = BaseMamba(d_model=12, d_state=4, d_conv=4,
                             dt_rank=3, expand=2, use_fast_path=False)
        expected_next = torch.rand(8)
        torch.manual_seed(123)
        model = self.make_model(bidirectional=True, motion_delta=True)
        actual_next = torch.rand(8)
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]), name)
        self.assertTrue(torch.equal(expected_next, actual_next))
        self.assert_close(model.bidir_logit.sigmoid(),
                          torch.full_like(model.bidir_logit, 0.1))

    def test_zero_initialized_motion_matches_original_outputs_and_input_gradients(self):
        model = self.make_model(motion_delta=True)
        baseline = BaseMamba(d_model=12, d_state=4, d_conv=4,
                             dt_rank=3, expand=2, use_fast_path=False)
        baseline.load_state_dict({name: model.state_dict()[name]
                                  for name in baseline.state_dict()}, strict=True)
        x1 = torch.randn(2, 6, 12, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_()
        y1, y2 = model(x1), baseline(x2)
        self.assert_close(y1, y2)
        y1.square().sum().backward()
        y2.square().sum().backward()
        self.assert_close(x1.grad, x2.grad)

    def test_all_modes_match_explicit_equations_and_gradients(self):
        for bidirectional, motion_delta in ((True, False), (False, True), (True, True)):
            with self.subTest(bidirectional=bidirectional, motion_delta=motion_delta):
                model = self.make_model(bidirectional=bidirectional,
                                        motion_delta=motion_delta)
                if motion_delta:
                    nn.init.normal_(model.motion_up.weight, std=0.2)
                x = torch.randn(2, 7, 12, requires_grad=True)
                expected = explicit_output(model, x)
                actual = model(x)
                self.assert_close(actual, expected)
                expected_grad = torch.autograd.grad(expected.square().sum(), x,
                                                    retain_graph=True)[0]
                actual.square().sum().backward()
                self.assert_close(x.grad, expected_grad)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if bidirectional:
                    self.assertGreater(model.bidir_logit.grad.abs().sum().item(), 0)
                if motion_delta:
                    self.assertGreater(model.motion_up.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.motion_down.weight.grad.abs().sum().item(), 0)

    def test_reverse_scan_flips_all_temporal_arguments_and_skips_d_and_z(self):
        model = self.make_model(bidirectional=True, motion_delta=True)
        calls = []

        def record(*args, **kwargs):
            calls.append((args, kwargs))
            return reference_scan(*args, **kwargs)

        with mock.patch.object(subject, 'selective_scan_fn', side_effect=record):
            actual = model(torch.randn(2, 5, 12))
        self.assertEqual(len(calls), 2)
        forward, backward = calls
        for index in (0, 1, 3, 4):
            self.assert_close(backward[0][index], forward[0][index].flip(-1))
        self.assert_close(backward[0][2], forward[0][2])
        for args, kwargs in calls:
            self.assertIsNone(args[5] if len(args) > 5 else kwargs.get('D'))
            self.assertIsNone(kwargs.get('z'))
        self.assertTrue(torch.isfinite(actual).all())

    def test_first_motion_difference_is_zero_and_motion_can_learn(self):
        model = self.make_model(motion_delta=True)
        seen = []
        hook = model.motion_norm.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        inputs = torch.randn(2, 6, 12)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        first_up_gradient = None
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            output.square().sum().backward()
            if first_up_gradient is None:
                first_up_gradient = model.motion_up.weight.grad.detach().clone()
            optimizer.step()
        hook.remove()
        self.assertTrue(torch.equal(seen[0][:, 0], torch.zeros_like(seen[0][:, 0])))
        self.assertGreater(first_up_gradient.abs().sum().item(), 0)
        self.assertGreater(model.motion_down.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.motion_down.weight.grad).all())

    def test_short_sequences_and_checkpoint_roundtrip(self):
        for length in (1, 2, 3):
            with self.subTest(length=length):
                model = self.make_model(bidirectional=True, motion_delta=True)
                x = torch.randn(2, length, 12)
                output = model(x)
                self.assertEqual(output.shape, x.shape)
                self.assert_close(output, explicit_output(model, x))
                output.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.grad is not None))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = self.make_model(bidirectional=True, motion_delta=True)
                restored.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)
                self.assert_close(restored(x), model(x))

    def test_enhanced_modes_explicitly_reject_cached_inference(self):
        for options in ({'bidirectional': True}, {'motion_delta': True}):
            model = self.make_model(**options)
            with self.assertRaises((ValueError, NotImplementedError)):
                model(torch.randn(1, 3, 12), inference_params=object())

    def test_invalid_parameters_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(motion_rank=value), self.assertRaises((ValueError, TypeError)):
                SkeletonMamba(d_model=12, motion_delta=True, motion_rank=value)
        for value in (0.0, 1.0, -0.1, math.nan, math.inf):
            with self.subTest(bidir_init=value), self.assertRaises(ValueError):
                self.make_model(bidirectional=True, bidir_init=value)
        for value in (-0.1, math.nan, math.inf):
            with self.subTest(motion_scale=value), self.assertRaises(ValueError):
                self.make_model(motion_delta=True, motion_scale=value)


class ExperimentParityTest(unittest.TestCase):
    def test_new_configs_preserve_completed_baseline_training_settings(self):
        directory = ROOT / 'config/nturgbd-cross-subject'
        baseline = yaml.safe_load((directory / 'joint_baseline_10ep.yaml').read_text(encoding='utf-8'))
        expected_flags = {
            'ssm_bidir': ([4, 7, 10], []),
            'ssm_motion': ([], [4, 7, 10]),
            'ssm_bidir_motion': ([4, 7, 10], [4, 7, 10]),
        }
        outputs = {baseline['work_dir']}
        for mode, (bidir_layers, motion_layers) in expected_flags.items():
            with self.subTest(mode=mode):
                config = yaml.safe_load((directory / f'joint_{mode}_10ep.yaml').read_text(encoding='utf-8'))
                self.assertNotIn(config['work_dir'], outputs)
                outputs.add(config['work_dir'])
                self.assertEqual({key: value for key, value in config.items()
                                  if key not in ('work_dir', 'model_args')},
                                 {key: value for key, value in baseline.items()
                                  if key not in ('work_dir', 'model_args')})
                model_args = dict(config['model_args'])
                self.assertEqual(model_args.pop('ssm_bidir_layers'), bidir_layers)
                self.assertEqual(model_args.pop('ssm_motion_layers'), motion_layers)
                self.assertEqual(model_args.pop('ssm_motion_rank'), 16)
                self.assertEqual(model_args.pop('ssm_motion_scale'), 0.1)
                self.assertEqual(model_args.pop('ssm_bidir_init'), 0.1)
                self.assertEqual(model_args.pop('ctr_layers'), [])
                self.assertEqual(model_args.pop('fr_layers'), [])
                self.assertEqual(model_args, baseline['model_args'])


if __name__ == '__main__':
    unittest.main()
