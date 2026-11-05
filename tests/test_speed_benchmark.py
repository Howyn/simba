"""CPU-only checks for timing comparisons and partial benchmark recovery."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_autodl import LoaderWindow, make_cases, summarize, write_reports


def row(name, seconds):
    return dict(name=name, status='ok', batch_size=16, test_batch_size=64, tf32=False,
                train_samples_per_second=80, eval_samples_per_second=300,
                train_peak_mib=6000, estimated_epoch_seconds=seconds)


class SpeedBenchmarkTests(unittest.TestCase):
    def test_time_reduction_is_not_throughput_increase(self):
        summary = summarize([row('reference_b16_eval16', 750), row('candidate', 600)])
        self.assertTrue(summary['target_20_percent_met'])
        self.assertEqual(summary['cases'][1]['estimated_time_saved_percent'], 20)
        summary = summarize([row('reference_b16_eval16', 750), row('candidate', 625)])
        self.assertFalse(summary['target_20_percent_met'])

    def test_warmup_and_measurement_consume_disjoint_batches(self):
        iterator = iter(range(12))
        warmup = LoaderWindow(iterator, 3, object())
        measure = LoaderWindow(iterator, 5, object())
        self.assertEqual(list(warmup), [0, 1, 2])
        self.assertEqual(list(measure), [3, 4, 5, 6, 7])
        self.assertEqual(len(measure), 5)

    def test_oom_does_not_win_and_partial_reports_are_readable(self):
        rows = [row('reference_b16_eval16', 750), dict(name='too_large', status='oom')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            summary = write_reports(path, rows)
            self.assertEqual(summary['fastest'], 'reference_b16_eval16')
            self.assertEqual(json.loads((path / 'results.json').read_text())['cases'][1]['status'], 'oom')
            self.assertIn('too_large', (path / 'summary.md').read_text())
            self.assertIn('--precision bf16', (path / 'recommended_cli.txt').read_text())
        self.assertEqual(len(make_cases()), 4)
        self.assertEqual(len(make_cases(include_batch32=True)), 5)


if __name__ == '__main__':
    unittest.main()
