"""CPU equation, invariant and gradient tests for the context/retain SSM.

Run: python -m unittest discover -s tests -p test_context_write.py
Requires torch and einops in the experiment environment; no CUDA extension is
needed because the scan is replaced by an explicit reference recurrence.
"""

import importlib.util
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
        subject = load_source('mamba_ssm.modules.mamba_context_write',
                              ROOT / 'mamba_ssm/modules/mamba_context_write.py')
    return base.Mamba, subject


BaseMamba, subject = isolated_sources()
ContextWriteMamba = subject.ContextWriteMamba


def explicit_output(model, inputs):
    """Separate wiring expression; the subject's helper methods are not used."""
    xz = F.linear(inputs, model.in_proj.weight, model.in_proj.bias).transpose(1, 2)
    u, z = xz.chunk(2, dim=1)
    u = F.silu(F.conv1d(u, model.conv1d.weight, model.conv1d.bias,
                        padding=model.d_conv - 1, groups=model.d_inner)[..., :inputs.shape[1]])
    params = F.linear(u.transpose(1, 2), model.x_proj.weight)
    rank, B, C = torch.split(params, (model.dt_rank, model.d_state, model.d_state), dim=-1)
    delta = F.softplus(F.linear(rank, model.dt_proj.weight, model.dt_proj.bias)).transpose(1, 2)
    B = B.transpose(1, 2)
    C = C.transpose(1, 2)
    if model.context_write:
        hidden = model.context_norm(u.transpose(1, 2))
        query = model.context_down(hidden).transpose(1, 2)
        branches = []
        for conv in model.context_convs:
            padding = (conv.kernel_size[0] - 1) // 2
            branches.append(conv(F.pad(query, (padding, padding))))
        fused = F.silu(model.context_fuse(torch.cat(branches, dim=1).transpose(1, 2)))
        residual = model.context_up(fused).transpose(1, 2)
        B = B + model.context_scale * torch.tanh(residual)
    if model.retain_write:
        retain = model.retain_floor + (1 - model.retain_floor) * torch.sigmoid(model.retain_logit)
        write = model.write_max * torch.sigmoid(model.write_logit)
        delta = delta * retain[None, :, None]
        u = u * (write / retain)[None, :, None]
    A = -model.A_log.float().exp()
    state = reference_scan(u, delta, A, B, C)
    gated = (state + model.D[None, :, None] * u) * F.silu(z)
    return F.linear(gated.transpose(1, 2), model.out_proj.weight, model.out_proj.bias)


class ContextWriteTest(unittest.TestCase):
    def make(self, **kwargs):
        options = dict(d_model=12, d_state=4, d_conv=4, expand=2, dt_rank=3, use_fast_path=False,
                       context_write=True, retain_write=True, context_rank=2,
                       context_kernels=(3, 5))
        options.update(kwargs)
        return ContextWriteMamba(**options)

    def inputs(self, batch=2, frames=9, width=12):
        torch.manual_seed(5)
        return torch.randn(batch, frames, width)

    def test_new_parameters_do_not_change_backbone_initialization_or_rng(self):
        torch.manual_seed(17)
        base = BaseMamba(d_model=12, d_state=4, d_conv=4, expand=2, dt_rank=3, use_fast_path=False)
        expected_cpu_rng = torch.get_rng_state().clone()
        torch.manual_seed(17)
        model = self.make()
        self.assertTrue(torch.equal(expected_cpu_rng, torch.get_rng_state()))
        base_state, state = base.state_dict(), model.state_dict()
        for name, value in base_state.items():
            self.assertIn(name, state)
            self.assertTrue(torch.equal(value, state[name]), name)
        extra = set(state) - set(base_state)
        self.assertTrue(extra)
        for name in extra:
            self.assertTrue(name.startswith(('context_', 'retain_', 'write_')), name)

    def test_default_toggles_off_take_the_unchanged_fused_path(self):
        model = self.make()
        base = BaseMamba(d_model=12, d_state=4, d_conv=4, expand=2, dt_rank=3, use_fast_path=False)
        base.load_state_dict({name: value for name, value in model.state_dict().items()
                              if name in base.state_dict()})
        model.context_enabled = False
        model.retain_write_enabled = False
        x = self.inputs()
        self.assertTrue(torch.equal(model(x), base(x)))

    def test_equations_match_an_independent_reference_wiring(self):
        model = self.make()
        x = self.inputs()
        self.assert_close(model(x), explicit_output(model, x))
        for frame in (1, 5, 9):
            sliced = model(x[:, :frame])
            self.assert_close(sliced, explicit_output(model, x[:, :frame]))

    def test_identity_gains_reproduce_the_original_recurrence(self):
        model = self.make()
        model.context_enabled = False
        x = self.inputs()
        model.retain_write_enabled = False
        reference = model(x)
        model.retain_write_enabled = True
        with torch.no_grad():
            # retain -> 1 from above, write exactly 1.0 (write_max / 2).
            model.retain_logit.fill_(20.0)
            model.write_logit.fill_(0.0)
        torch.testing.assert_close(model(x), reference, atol=2e-5, rtol=1e-5)

    def test_retention_is_the_published_power_of_the_baseline_and_write_is_independent(self):
        model = self.make()
        parts = model.probe_components(self.inputs())
        retain, write = parts['retain'], parts['write']
        torch.testing.assert_close(retain, torch.full_like(retain, 0.5), atol=1e-6, rtol=0)
        torch.testing.assert_close(write, torch.ones_like(write), atol=1e-6, rtol=0)
        expected = parts['retention_base'].pow(retain.double().view(1, -1, 1, 1))
        torch.testing.assert_close(parts['retention'], expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(parts['u_write_scale'], write / retain, atol=1e-6, rtol=1e-6)
        self.assertTrue(bool((parts['delta_write'] <= parts['delta']).all()))

    def test_context_only_changes_the_write_matrix(self):
        model = self.make(retain_write=False)
        parts = model.probe_components(self.inputs())
        self.assertIsNotNone(parts['B_context'])
        self.assertFalse(torch.equal(parts['B'], parts['B_base']))
        self.assertTrue(torch.equal(parts['delta_write'], parts['delta']))
        ratio = parts['B_context'].norm().item() / parts['B_base'].norm().item()
        self.assertTrue(0 < ratio < 0.5, ratio)
        self.assertAlmostEqual(parts['context_relative'], ratio, places=5)

    def test_context_neighbourhood_is_bounded_and_never_circular(self):
        model = self.make(retain_write=False)
        x = self.inputs(frames=12)
        reference = model.probe_components(x)['B_context']
        perturbed = x.clone()
        perturbed[:, 0] += 1.0
        after = model.probe_components(perturbed)['B_context']
        span = (3 - 1) // 2 + (5 - 1) // 2  # total reach of the two scales
        self.assertFalse(torch.equal(reference[:, :, :span], after[:, :, :span]))
        torch.testing.assert_close(reference[:, :, span:], after[:, :, span:])
        # The last frame must not reach the first frames either.
        perturbed = x.clone()
        perturbed[:, -1] += 1.0
        after = model.probe_components(perturbed)['B_context']
        torch.testing.assert_close(reference[:, :, :12 - span], after[:, :, :12 - span])

    def test_gains_stay_bounded_at_extreme_logits(self):
        model = self.make()
        with torch.no_grad():
            model.retain_logit.fill_(10.0)
            model.write_logit.fill_(10.0)
        self.assertTrue(bool((model.retain_gain() < 1).all()))
        self.assertTrue(bool((model.retain_gain() > model.retain_floor).all()))
        self.assertTrue(bool((model.write_gain() < model.write_max).all()))
        self.assertTrue(bool((model.write_gain() > 0).all()))
        with torch.no_grad():
            model.retain_logit.fill_(-10.0)
            model.write_logit.fill_(-10.0)
        self.assertGreater(model.retain_gain().min().item(), model.retain_floor)
        self.assertGreater(model.write_gain().min().item(), 0.0)

    def test_every_new_parameter_receives_a_finite_nonzero_task_gradient(self):
        model = self.make()
        x = self.inputs().requires_grad_(True)
        target = torch.zeros(2, 12)
        target[:, 3] = 1.0
        F.cross_entropy(model(x), target).backward()
        new_names = [name for name in dict(model.named_parameters())
                     if name.startswith(('context_', 'retain_', 'write_'))]
        self.assertTrue(new_names)
        for name in new_names:
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().max().item(), 0.0, name)

    def test_state_and_correction_toggles_isolate_the_readout(self):
        model = self.make()
        x = self.inputs()
        full = model(x)
        model.correction_enabled = False
        without_correction = model(x)
        self.assertFalse(torch.equal(full, without_correction))
        model.correction_enabled = True
        model.state_enabled = False
        disabled = model(x)
        parts = model.probe_components(x)
        self.assertFalse(bool(parts['state_readout'].abs().sum()))
        self.assertFalse(torch.equal(disabled, full))
        model.state_enabled = True
        torch.testing.assert_close(model(x), full, atol=0, rtol=0)

    def test_gain_logits_are_protected_from_weight_decay(self):
        sys.path.insert(0, str(ROOT))
        from tools.ssm_optimizer import algorithm_parameter_groups

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mamba = torch.nn.Module()
                self.mamba.mamba = torch.nn.Module()
                self.mamba.mamba.weight = torch.nn.Parameter(torch.ones(2))
                self.mamba.mamba.retain_logit = torch.nn.Parameter(torch.zeros(2))
                self.mamba.mamba.write_logit = torch.nn.Parameter(torch.zeros(2))

        wrapper = Wrapper()
        groups = algorithm_parameter_groups(wrapper, 0.0004)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]['weight_decay'], 0.0004)
        self.assertEqual(groups[1]['weight_decay'], 0.0)
        protected = {id(parameter) for parameter in groups[1]['params']}
        self.assertIn(id(wrapper.mamba.mamba.retain_logit), protected)
        self.assertIn(id(wrapper.mamba.mamba.write_logit), protected)
        self.assertNotIn(id(wrapper.mamba.mamba.weight), protected)

    def test_invalid_settings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'context_write and/or retain_write'):
            ContextWriteMamba(d_model=12, d_state=4, d_conv=4, expand=2, dt_rank=3,
                              use_fast_path=False)
        with self.assertRaisesRegex(ValueError, 'retain_init'):
            self.make(retain_init=0.2)
        with self.assertRaisesRegex(ValueError, 'context_kernels'):
            self.make(context_kernels=(2, 4))
        with self.assertRaisesRegex(ValueError, 'write_init'):
            self.make(write_init=3.0)

    def test_cached_inference_and_step_decoding_are_refused(self):
        model = self.make()
        with self.assertRaisesRegex(ValueError, 'offline clips'):
            model(self.inputs(), inference_params=object())
        with self.assertRaisesRegex(ValueError, 'step decoding'):
            model.step(self.inputs()[:, 0], None, None)

    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main()
