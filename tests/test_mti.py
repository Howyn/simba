"""CPU operator/wrapper checks; real fused-Mamba checks run when CUDA exists."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mti_module = load_file('standalone_temporal_mti', 'mamba_ssm/modules/temporal_mti.py')


def reference_cycle(x, projection, kernel_size):
    batch, channels, length, joints = x.shape
    outputs = []
    for time in range(length):
        inputs = []
        for channel in range(channels):
            source = time + channel % kernel_size - kernel_size // 2
            value = x[:, channel, source, :] if 0 <= source < length else x.new_zeros(batch, joints)
            inputs.append(value)
        outputs.append(projection(torch.stack(inputs, dim=-1)).permute(0, 2, 1))
    return torch.stack(outputs, dim=2)


class TemporalMTITests(unittest.TestCase):
    def test_forward_and_all_gradients_against_reference(self):
        torch.manual_seed(6)
        for kernel in (1, 3, 5):
            with self.subTest(kernel=kernel):
                op = mti_module.TemporalCycleFC(7, kernel).double()
                reference = copy.deepcopy(op)
                x = torch.randn(2, 7, 4, 3, dtype=torch.double, requires_grad=True)
                x_ref = x.detach().clone().requires_grad_()
                actual = op(x)
                expected = reference_cycle(x_ref, reference.proj, kernel)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                weight = torch.randn_like(actual)
                (actual * weight).sum().backward()
                (expected * weight).sum().backward()
                torch.testing.assert_close(x.grad, x_ref.grad)
                for got, want in zip(op.parameters(), reference.parameters()):
                    torch.testing.assert_close(got.grad, want.grad)

    def test_boundaries_do_not_wrap_or_mix_joints(self):
        op = mti_module.TemporalCycleFC(3, 3)
        with torch.no_grad():
            op.proj.weight.copy_(torch.eye(3))
            op.proj.bias.zero_()
        x = torch.ones(1, 3, 4, 2)
        x[..., 1] = 0
        y = op(x)
        self.assertEqual(y[0, 0, 0, 0].item(), 0)
        self.assertEqual(y[0, 2, -1, 0].item(), 0)
        self.assertEqual(y[..., 1].count_nonzero().item(), 0)
        torch.testing.assert_close(y[0, 1, :, 0], torch.ones(4))

    def test_parallel_residual_sum_and_finite_difference(self):
        torch.manual_seed(5)
        op = mti_module.MultiScaleTemporalInteraction(4).double()
        x = torch.randn(1, 4, 3, 2, dtype=torch.double, requires_grad=True)
        expected = x + sum(reference_cycle(x, branch.proj, branch.kernel_size) for branch in op.branches)
        torch.testing.assert_close(op(x), expected)
        self.assertTrue(torch.autograd.gradcheck(op, (x,)))
        self.assertEqual(sum(p.numel() for p in op.parameters()), 3 * (4 * 4 + 4))


class FakeMamba(nn.Module):
    """Only isolates the wrapper from unavailable CPU CUDA extensions."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(x)


def wrapper_stubs():
    names = ['mamba_ssm', 'mamba_ssm.ops', 'mamba_ssm.ops.triton',
             'mamba_ssm.ops.triton.layernorm', 'mamba_ssm.modules']
    modules = {name: types.ModuleType(name) for name in names}
    modules['mamba_ssm'].Mamba = FakeMamba
    modules['mamba_ssm.ops.triton.layernorm'].RMSNorm = nn.LayerNorm
    modules['mamba_ssm.modules.temporal_mti'] = mti_module
    return modules


class MambaWrapperTests(unittest.TestCase):
    def test_disabled_matches_committed_wrapper_and_enabled_preserves_rng(self):
        original = subprocess.check_output(
            ['git', '-c', f'safe.directory={ROOT.as_posix()}', 'show', '94478ab:mamba_blocks.py'],
            cwd=ROOT, text=True,
        )
        with patch.dict(sys.modules, wrapper_stubs()):
            old_namespace = {}
            exec(compile(original, 'baseline_mamba_blocks.py', 'exec'), old_namespace)
            new = load_file('isolated_mamba_blocks', 'mamba_blocks.py')
            torch.manual_seed(4321)
            old = old_namespace['MambaBlock'](15)
            after_old = torch.get_rng_state().clone()
            torch.manual_seed(4321)
            disabled = new.MambaBlock(15, use_mti=False, num_point=3)
            torch.testing.assert_close(torch.get_rng_state(), after_old)
            torch.manual_seed(4321)
            enabled = new.MambaBlock(15, use_mti=True, num_point=3)
            torch.testing.assert_close(torch.get_rng_state(), after_old)
            self.assertEqual(set(old.state_dict()), set(disabled.state_dict()))
            for key, value in old.state_dict().items():
                torch.testing.assert_close(disabled.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(enabled.state_dict()[key], value, rtol=0, atol=0)
            x = torch.randn(2, 7, 15, requires_grad=True)
            torch.testing.assert_close(disabled(x), old(x), rtol=0, atol=0)
            # Zero branches reduce MTI to identity, including the wrapper residual.
            with torch.no_grad():
                for parameter in enabled.mti.parameters():
                    parameter.zero_()
            torch.testing.assert_close(enabled(x), old(x), rtol=0, atol=0)
            (enabled(x) * torch.randn_like(x)).sum().backward()
            for branch in enabled.mti.branches:
                self.assertGreater(branch.proj.weight.grad.abs().sum().item(), 0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA fused Mamba requires an AutoDL GPU environment')
class CUDAMambaMTITests(unittest.TestCase):
    def test_real_mamba_parity_and_amp_backward(self):
        sys.path.insert(0, str(ROOT))
        from mamba_blocks import MambaBlock
        torch.manual_seed(13)
        baseline = MambaBlock(625, num_point=25).cuda()
        torch.manual_seed(13)
        improved = MambaBlock(625, use_mti=True, num_point=25).cuda()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(improved.state_dict()[key], value, rtol=0, atol=0)
        x = torch.randn(2, 16, 625, device='cuda')
        with torch.no_grad():
            expected = baseline.dropout(baseline.norm(baseline.mamba(x) + x))
            torch.testing.assert_close(baseline(x), expected, rtol=0, atol=0)
        scaler = torch.amp.GradScaler('cuda', init_scale=128)
        optimizer = torch.optim.SGD(improved.parameters(), lr=0.005)
        with torch.autocast('cuda', dtype=torch.float16):
            values = improved(x).float()
            loss = (values * torch.randn_like(values)).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for branch in improved.mti.branches:
            self.assertTrue(torch.isfinite(branch.proj.weight.grad).all())
            self.assertGreater(branch.proj.weight.grad.abs().sum().item(), 0)
        scaler.step(optimizer)
        scaler.update()

    def test_full_model_common_weights_and_amp_backward(self):
        sys.path.insert(0, str(ROOT))
        import yaml
        from model.Hyperformer import Model
        config = yaml.safe_load((ROOT / 'config/nturgbd-cross-subject/joint_mti_baseline15.yaml').read_text())
        args = config['model_args']
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        baseline = Model(**args).cuda()
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        improved = Model(**dict(args, use_mti=True)).cuda()
        state = improved.state_dict()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        extra = sum(p.numel() for p in improved.parameters()) - sum(p.numel() for p in baseline.parameters())
        self.assertEqual(extra, 19500)
        del baseline, state
        x = torch.randn(1, 3, 64, 25, 2, device='cuda')
        label = torch.tensor([3], device='cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            output, _ = improved(x, torch.nn.functional.one_hot(label, 60))
            loss = torch.nn.functional.cross_entropy(output, label)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for layer in (improved.l1, improved.l5, improved.l10):
            for branch in layer.mamba.mti.branches:
                self.assertIsNotNone(branch.proj.weight.grad)
                self.assertTrue(torch.isfinite(branch.proj.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
