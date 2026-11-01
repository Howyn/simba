"""Saved-prediction diagnostics require no GPU, dataset, or NumPy."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/diagnose_ssm_results.py'
SPEC = importlib.util.spec_from_file_location('ssm_diagnostics', SCRIPT)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class UnsupportedCheckpointMetadata:
    pass


class PredictionDiagnosticsTest(unittest.TestCase):
    def test_paired_counts_and_class_changes(self):
        baseline = [(0, 0), (1, 1), (0, 2), (2, 1), (1, 0), (2, 2)]
        combined = [(0, 0), (0, 1), (2, 2), (1, 1), (1, 0), (2, 2)]
        result = diagnostics.compare_predictions(baseline, combined)
        self.assertEqual(result['samples'], 6)
        self.assertEqual(result['both_correct'], 2)
        self.assertEqual(result['both_wrong'], 1)
        self.assertEqual(result['gained'], 2)
        self.assertEqual(result['lost'], 1)
        self.assertEqual(result['net_correct'], 1)
        self.assertAlmostEqual(result['baseline_top1'], 50)
        self.assertAlmostEqual(result['combined_top1'], 100 * 4 / 6)
        self.assertAlmostEqual(result['delta_pp'], 100 / 6)
        self.assertEqual(result['classes'][2]['combined_correct'], 2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            diagnostics.print_comparison(result, 3)
        self.assertIn('class 2: +50.00 pp', output.getvalue())

    def test_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'counts differ'):
            diagnostics.compare_predictions([(0, 0)], [(0, 0), (1, 1)])

    def test_label_order_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'row 2'):
            diagnostics.compare_predictions([(0, 0), (1, 1)], [(0, 0), (2, 2)])

    def test_empty_predictions(self):
        with self.assertRaisesRegex(ValueError, 'No paired predictions'):
            diagnostics.compare_predictions([], [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-best_right.txt'
            path.touch()
            with self.assertRaisesRegex(ValueError, 'no predictions'):
                diagnostics.load_predictions(path)

    def test_saved_prediction_format_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-best_right.txt'
            path.write_text('0,0\n2,1\n', encoding='utf-8')
            self.assertEqual(diagnostics.load_predictions(path), [(0, 0), (2, 1)])
            for invalid in ('1\n', '1,2,3\n', 'a,1\n', '-1,0\n', '0,60\n', '\n'):
                with self.subTest(invalid=invalid):
                    path.write_text(invalid, encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, ':1: expected'):
                        diagnostics.load_predictions(path)


class CheckpointDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('optional checkpoint checks need PyTorch')
        cls.torch = torch

    def state(self):
        state = {}
        for layer in (4, 7, 10):
            prefix = f'l{layer}.mamba.mamba.'
            state[prefix + 'bidir_logit'] = self.torch.full((8,), math.log(.1 / .9))
            state[prefix + 'motion_up.weight'] = self.torch.zeros(3, 2)
            state[prefix + 'motion_down.weight'] = self.torch.ones(2, 8)
        return state

    def test_cpu_checkpoint_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-best.pt'
            self.torch.save({'model_state_dict': self.state()}, path)
            result = diagnostics.checkpoint_stats(path)
        self.assertEqual([row['layer'] for row in result], [4, 7, 10])
        self.assertAlmostEqual(result[0]['gate_mean'], .1, places=6)
        self.assertEqual(result[0]['motion_up.weight']['nonzero'], 0)
        self.assertEqual(result[0]['motion_down.weight']['rms'], 1)

    def test_missing_or_nonfinite_parameter_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-best.pt'
            state = self.state()
            del state['l4.mamba.mamba.bidir_logit']
            self.torch.save({'model_state_dict': state}, path)
            with self.assertRaisesRegex(ValueError, 'missing l4'):
                diagnostics.checkpoint_stats(path)
            state = self.state()
            state['l7.mamba.mamba.motion_up.weight'][0, 0] = float('nan')
            self.torch.save({'model_state_dict': state}, path)
            with self.assertRaisesRegex(ValueError, 'non-finite parameter l7'):
                diagnostics.checkpoint_stats(path)

    def test_real_numpy_metadata_and_both_scalar_module_names(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('legacy metadata checks need NumPy')
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / 'native.pt'
            self.torch.save({'model_state_dict': self.state(), 'best_accuracy': np.float64(.862),
                             'best_epoch': np.int64(19),
                             'optimizer_state_dict': {'param_groups': [{'lr': np.float64(.00025)}]}}, native)
            # Mirror the NumPy 1/2 GLOBAL-path rename using valid torch archives.
            for module in (b'numpy.core.multiarray', b'numpy._core.multiarray'):
                path = Path(directory) / ('legacy.pt' if b'._core' not in module else 'modern.pt')
                with zipfile.ZipFile(native) as source, zipfile.ZipFile(path, 'w') as target:
                    for name in source.namelist():
                        data = source.read(name)
                        if name.endswith('/data.pkl'):
                            data = data.replace(b'numpy._core.multiarray\n', module + b'\n')
                            data = data.replace(b'numpy.core.multiarray\n', module + b'\n')
                        target.writestr(name, data)
                with self.torch.serialization.safe_globals([np.dtype]):
                    before = set(self.torch.serialization.get_safe_globals())
                    with mock.patch.object(self.torch, 'load', wraps=self.torch.load) as load:
                        result = diagnostics.checkpoint_stats(path)
                    self.assertTrue(all(call.kwargs['weights_only'] is True for call in load.call_args_list))
                    self.assertEqual(set(self.torch.serialization.get_safe_globals()), before)
                self.assertAlmostEqual(result[0]['gate_mean'], .1, places=6)

    def test_unknown_metadata_is_rejected_without_unsafe_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unknown.pt'
            self.torch.save({'model_state_dict': self.state(),
                             'metadata': UnsupportedCheckpointMetadata()}, path)
            with mock.patch.object(self.torch, 'load', wraps=self.torch.load) as load:
                with self.assertRaisesRegex(ValueError, 'safe .*loading'):
                    diagnostics.checkpoint_stats(path)
            self.assertTrue(all(call.kwargs['weights_only'] is True for call in load.call_args_list))


if __name__ == '__main__':
    unittest.main()
