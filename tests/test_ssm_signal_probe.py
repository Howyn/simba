"""CPU checks for the read-only SSM signal measurements and live hooks."""
import importlib.util
import math
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/probe_ssm_signal.py'
SPEC = importlib.util.spec_from_file_location('ssm_signal_probe', SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ToyDelta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dt_rank, self.motion_scale = 2, .1
        self.x_proj = torch.nn.Linear(3, 4, bias=False)
        self.motion_up = torch.nn.Linear(3, 2, bias=False)
        self.dt_proj = torch.nn.Linear(2, 3)

    def forward(self, data):
        rank = self.x_proj(data)[..., :self.dt_rank]
        correction = self.motion_up(data.reshape(2, -1, 3))
        return self.dt_proj(rank + self.motion_scale * correction.reshape_as(rank).tanh())


class SignalProbeTest(unittest.TestCase):
    def test_zero_correction_is_exactly_inactive(self):
        rank = torch.randn(6, 2)
        values = probe.delta_change_metrics(rank, torch.zeros(2, 3, 2),
                                            torch.randn(4, 2), torch.randn(4), .1)
        self.assertEqual(values['difference_square'], 0.)
        self.assertEqual(values['max_relative'], 0.)
        self.assertGreater(values['base_square'], 0.)

    def test_nonzero_correction_is_measured_after_projection_and_softplus(self):
        values = probe.delta_change_metrics(torch.zeros(1, 1), torch.ones(1, 1, 1),
                                            torch.ones(1, 1), torch.zeros(1), .5)
        expected = math.log1p(math.exp(.5 * math.tanh(1.))) - math.log(2.)
        self.assertAlmostEqual(values['difference_square'], expected ** 2, places=7)
        self.assertAlmostEqual(values['max_relative'], expected / math.log(2.), places=6)

    def test_live_hooks_observe_real_call_and_leave_parameters_unchanged(self):
        torch.manual_seed(4321)
        model = ToyDelta()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        observer = probe.MotionDeltaProbe(model)
        model(torch.randn(6, 3)).square().mean().backward()
        relative, maximum = observer.summary()
        self.assertGreater(relative, 0.)
        self.assertGreater(maximum, 0.)
        self.assertEqual(len(observer.measurements), 1)
        self.assertIsNone(observer.rank)
        probe.assert_parameters_unchanged(model, before)
        observer.close()
        model(torch.randn(6, 3))
        self.assertEqual(len(observer.measurements), 1)

    def test_task_and_decay_gradients_are_distinguished(self):
        parameter = torch.nn.Parameter(torch.full((4,), 2.))
        self.assertIsNone(probe.gradient_metrics(parameter, .1)['task_grad_rms'])
        parameter.grad = torch.full_like(parameter, .5)
        values = probe.gradient_metrics(parameter, .1)
        self.assertAlmostEqual(values['decay_grad_rms'], .2)
        self.assertAlmostEqual(values['task_grad_rms'], .5)
        self.assertAlmostEqual(values['ratio'], 2.5)
        parameter.grad[0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Non-finite task'):
            probe.gradient_metrics(parameter, .1)

    def test_parameter_change_guard_catches_updates(self):
        model = torch.nn.Linear(2, 1)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        with torch.no_grad():
            model.weight.add_(.01)
        with self.assertRaisesRegex(RuntimeError, 'Parameter unexpectedly changed'):
            probe.assert_parameters_unchanged(model, before)


if __name__ == '__main__':
    unittest.main()
