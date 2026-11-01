"""CPU checks for temporary interventions; no CUDA/NTU dependencies."""
import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/ablate_ssm_checkpoint.py'
SPEC = importlib.util.spec_from_file_location('ssm_functional_ablation', SCRIPT)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


class Inner(torch.nn.Module):
    def __init__(self, enhanced=True):
        super().__init__()
        self.dt_rank, self.d_state = 2, 2
        self.use_fast_path = True
        self.x_proj = torch.nn.Linear(3, 6, bias=False)
        if enhanced:
            self.motion_delta = self.bidirectional = True

    def forward(self, x):
        return 2 * x.flip(-1)


class Outer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = Inner()
        self.norm = torch.nn.LayerNorm(3)

    def forward(self, x):
        return self.norm(self.mamba(x) + x)


class FunctionalAblationTest(unittest.TestCase):
    def test_flags_are_restored_and_ordinary_modules_are_untouched(self):
        ordinary, enhanced = Inner(False), Inner()
        for variant, motion, bidir in (('motion_off', False, True),
                                       ('bidir_off', True, False),
                                       ('addons_off', False, False)):
            with tool.ablation([ordinary, enhanced], variant):
                self.assertEqual(enhanced.motion_delta, motion)
                self.assertEqual(enhanced.bidirectional, bidir)
                self.assertFalse(hasattr(ordinary, 'motion_delta'))
                self.assertTrue(ordinary.use_fast_path)
                self.assertEqual(enhanced.use_fast_path, variant != 'addons_off')
            self.assertTrue(enhanced.motion_delta and enhanced.bidirectional)
            self.assertTrue(enhanced.use_fast_path)

    def test_flags_restore_when_forward_raises(self):
        module = Inner()
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            with tool.ablation([module], 'addons_off'):
                raise RuntimeError('forward failed')
        self.assertTrue(module.motion_delta and module.bidirectional)

    def test_only_c_readout_rows_are_zeroed_and_restore_after_exception(self):
        module = Inner()
        previous = module.x_proj.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            with tool.ablation([module], 'state_readout_off'):
                torch.testing.assert_close(module.x_proj.weight[:-2], previous[:-2], atol=0, rtol=0)
                self.assertEqual(torch.count_nonzero(module.x_proj.weight[-2:]), 0)
                raise RuntimeError('forward failed')
        torch.testing.assert_close(module.x_proj.weight, previous, atol=0, rtol=0)

    def test_output_bypass_preserves_outer_normalization_and_residual(self):
        module = Outer()
        data = torch.tensor([[1., -2., 4.]])
        full = module(data)
        with tool.ablation([module.mamba], 'mamba_output_off'):
            torch.testing.assert_close(module(data), module.norm(data), atol=0, rtol=0)
            self.assertFalse(torch.allclose(module(data), full))
        self.assertNotIn('forward', module.mamba.__dict__)
        torch.testing.assert_close(module(data), full, atol=0, rtol=0)

    def test_existing_forward_override_restores_on_exception(self):
        module = Inner()
        previous = lambda data: data + 7
        module.forward = previous
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            with tool.ablation([module], 'mamba_output_off'):
                self.assertEqual(torch.count_nonzero(module(torch.ones(2, 3))), 0)
                raise RuntimeError('forward failed')
        self.assertIs(module.forward, previous)

    def test_metrics_distinguish_prediction_changes_from_accuracy(self):
        full = torch.tensor([[3., 0., 0.], [0., 2., 0.], [0., 2., 0.]])
        changed = torch.tensor([[0., 3., 0.], [0., 2., 0.], [0., 0., 2.]])
        labels = torch.tensor([0, 1, 2])
        metrics = dict(samples=0, correct=0, changed=0, max_logit_delta=0.)
        for _ in range(2):
            tool.record(metrics, changed, labels, full)
        self.assertEqual(metrics, dict(samples=6, correct=4, changed=4, max_logit_delta=3.))

    def test_full_state_guard_checks_buffers_too(self):
        model = torch.nn.BatchNorm1d(3).eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        tool.assert_state_unchanged(model, before)
        model.running_mean.add_(1.)
        with self.assertRaisesRegex(RuntimeError, 'running_mean'):
            tool.assert_state_unchanged(model, before)


if __name__ == '__main__':
    unittest.main()
