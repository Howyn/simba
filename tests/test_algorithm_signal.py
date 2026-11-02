"""Check that diagnostic gates cannot mistake path noise for a live module."""
import copy
import unittest

from tools.check_algorithm_signal import assess_signal


class SignalGateTest(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(step=step, correction_ratio=0.01, logit_relative=0.002,
                          numerical_floor=1e-7, mechanism_ok=True)
                     for step in (0, 10, 20, 40)]
        self.gradients = [dict(step=step, parameters={
            'state.weight': dict(task_grad_rms=1e-5, decay_grad_rms=1e-4)})
            for step in (10, 20, 40)]

    def test_live_signal_passes_even_when_decay_exceeds_task_gradient(self):
        self.assertTrue(assess_signal(self.rows, self.gradients)[0])

    def test_dead_or_nonfinite_task_gradient_fails(self):
        for value in (0.0, 1e-15, float('nan')):
            with self.subTest(value=value):
                gradients = copy.deepcopy(self.gradients)
                for row in gradients:
                    row['parameters']['state.weight']['task_grad_rms'] = value
                self.assertFalse(assess_signal(self.rows, gradients)[0])

    def test_operator_noise_alone_fails(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row['numerical_floor'] = row['logit_relative'] / 2
        self.assertFalse(assess_signal(rows, self.gradients)[0])

    def test_tiny_excessive_nonfinite_or_invalid_mechanism_fails(self):
        for field, value in (('correction_ratio', 1e-8), ('correction_ratio', 2),
                             ('logit_relative', float('inf')), ('mechanism_ok', False)):
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[-1][field] = value
                self.assertFalse(assess_signal(rows, self.gradients)[0])

    def test_missing_late_observation_or_changed_backbone_fails(self):
        self.assertFalse(assess_signal(self.rows[:-1], self.gradients)[0])
        self.assertFalse(assess_signal(self.rows, self.gradients, initial_parity=False)[0])
        self.assertFalse(assess_signal(self.rows, [])[0])


if __name__ == '__main__':
    unittest.main()
