"""Global RNG checkpoint recovery tests without requiring a CUDA device."""

import copy
import io
import random
import unittest
from unittest import mock

import numpy as np
import torch

from tools.training_rng import capture_rng_state, restore_rng_state


class TrainingRNGTest(unittest.TestCase):
    def setUp(self):
        self.original = capture_rng_state()
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)

    def tearDown(self):
        restore_rng_state(self.original)

    def draw(self):
        return (random.random(), random.gauss(0, 1), np.random.random(7),
                np.random.normal(size=7), torch.rand(7), torch.randn(7))

    def assert_draw_equal(self, actual, expected):
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        np.testing.assert_array_equal(actual[3], expected[3])
        torch.testing.assert_close(actual[4], expected[4], atol=0, rtol=0)
        torch.testing.assert_close(actual[5], expected[5], atol=0, rtol=0)

    def test_restore_replays_python_numpy_and_torch_draws(self):
        # Include nonempty Gaussian caches in both Python and NumPy states.
        random.gauss(0, 1)
        np.random.normal()
        state = capture_rng_state()
        expected = [self.draw() for _ in range(3)]
        self.assertTrue(restore_rng_state(state))
        actual = [self.draw() for _ in range(3)]
        for one, two in zip(actual, expected):
            self.assert_draw_equal(one, two)

    def test_weights_only_serialization_uses_plain_numpy_metadata(self):
        random.gauss(0, 1)
        np.random.normal()
        state = capture_rng_state()
        expected = self.draw()
        file = io.BytesIO()
        torch.save({'rng_state': state}, file)
        file.seek(0)
        restored = torch.load(file, map_location='cpu', weights_only=True)['rng_state']
        self.assertIs(type(restored['numpy']['algorithm']), str)
        self.assertTrue(all(type(value) is int for value in restored['numpy']['keys']))
        self.assertIs(type(restored['numpy']['cached_gaussian']), float)
        restore_rng_state(restored)
        self.assert_draw_equal(self.draw(), expected)

    def test_capture_does_not_consume_or_alias_torch_generator(self):
        before = torch.get_rng_state().clone()
        state = capture_rng_state()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        state['torch_cpu'].zero_()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cuda_capture_is_copied_and_restore_passes_cpu_byte_states(self):
        fake_cuda_state = torch.arange(12, dtype=torch.uint8)
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[fake_cuda_state]), \
                mock.patch.object(torch.cuda, 'device_count', return_value=1), \
                mock.patch.object(torch.cuda, 'set_rng_state_all') as restore:
            state = capture_rng_state()
            expected = state['torch_cuda'][0].clone()
            fake_cuda_state.zero_()
            self.assertTrue(torch.equal(state['torch_cuda'][0], expected))
            self.assertTrue(restore_rng_state(state))
            restore.assert_called_once()
            actual = restore.call_args.args[0][0]
            self.assertEqual(actual.device.type, 'cpu')
            self.assertEqual(actual.dtype, torch.uint8)
            self.assertTrue(torch.equal(actual, expected))

    def test_cuda_mismatch_is_explicit_and_rejects_before_mutating_cpu_rng(self):
        state = capture_rng_state()
        state['torch_cuda'] = [torch.ones(12, dtype=torch.uint8)]
        expected = self.draw()
        # Return to a known state first, then try the invalid restore.
        restore_rng_state({**state, 'torch_cuda': []})
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                restore_rng_state(state)
            self.assert_draw_equal(self.draw(), expected)
            with self.assertWarns(RuntimeWarning):
                self.assertTrue(restore_rng_state(state, strict_cuda=False))

    def test_invalid_metadata_and_legacy_absence(self):
        self.assertFalse(restore_rng_state(None))
        state = capture_rng_state()
        invalids = [None, [], {**state, 'format_version': 2},
                    {**state, 'torch_cpu': torch.ones(8)},
                    {**state, 'torch_cpu': torch.ones(8, dtype=torch.uint8)},
                    {**state, 'numpy': {**state['numpy'], 'keys': [-1]}}]
        for invalid in invalids[1:]:
            with self.subTest(invalid_type=type(invalid).__name__), self.assertRaises(ValueError):
                restore_rng_state(invalid)
        copied = copy.deepcopy(state)
        copied['python'] = [state['python'][0], list(state['python'][1]), state['python'][2]]
        self.assertTrue(restore_rng_state(copied))


if __name__ == '__main__':
    unittest.main()
