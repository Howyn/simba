"""Baseline-reuse, comparison and shutdown controls; never shuts down a machine."""
from contextlib import nullcontext
import errno
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_combination1 as runner
from tools.experiment_results import SIGNATURE_KEYS, stable_window, training_signature, write_run_results


def config():
    values = dict.fromkeys(SIGNATURE_KEYS)
    values.update(seed=4321, num_epoch=10, screening_protocol='short10', model_args={
        'use_mti': False, 'use_bimamba': False}, base_lr=0.025, batch_size=24)
    return values


def result(values=None):
    values = [81.1, 81.2, 81.3] if values is None else values
    rows = [dict(epoch=e, top1_percent=50.0 if e < 8 else values[e-8],
                 learning_rate=0.0125 if e == 1 else 0.025 if e <= 6 else 0.0025)
            for e in range(1, 11)]
    return dict(completed=True, run_config=training_signature(config()), epochs=rows)


class RunnerTests(unittest.TestCase):
    def test_reuse_requires_matching_config_history_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'baseline_seed4321'
            work.mkdir()
            (root / 'source_revision.txt').write_text(runner.BASELINE_REVISION)
            (work / 'results.json').write_text(json.dumps(result()))
            self.assertTrue(runner.validate_baseline(work, config())['completed'])
            changed = config()
            changed['batch_size'] = 16
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                runner.validate_baseline(work, changed)
            (root / 'source_revision.txt').write_text('different-source')
            with self.assertRaisesRegex(ValueError, 'revision'):
                runner.validate_baseline(work, config())

    def test_stability_is_recomputed_not_trusted_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            base = result([81.2379, 81.8357, 81.7331])
            better = result([82.0, 82.1, 82.2])
            base['stability'] = {'status': 'stable'}
            runner.compare(Path(directory), 'combined', 4321, base, better)
            summary = runner.read_json(Path(directory) / 'comparison_combined_seed4321.json')
            self.assertEqual(summary['verdict'], 'inconclusive_unstable')
            self.assertIsNone(summary['delta_percentage_points'])

    def test_shutdown_cancellation_and_busy_gpu_never_call_poweroff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'KEEP_RUNNING').touch()
            with patch.object(runner.subprocess, 'run') as run:
                runner.wait_and_shutdown(root, 0)
                run.assert_not_called()
            (root / 'KEEP_RUNNING').unlink()
            with patch.object(runner.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '12345\n')) as run:
                runner.wait_and_shutdown(root, 0)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][0], 'nvidia-smi')

    def test_shutdown_waits_before_official_autodl_command(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
            calls = []
            def run(command, **kwargs):
                calls.append((clock[0], command))
                return subprocess.CompletedProcess(command, 0, '')
            with patch.object(runner.subprocess, 'run', side_effect=run), patch.object(runner.os, 'sync', create=True):
                runner.wait_and_shutdown(Path(directory), 300, sleep=sleep, monotonic=lambda: clock[0])
            self.assertEqual(calls[-1], (300.0, ['/usr/bin/shutdown']))

    def test_shutdown_enoexec_shell_wrapper_fallback(self):
        failure = OSError(errno.ENOEXEC, 'Exec format error')
        success = subprocess.CompletedProcess([], 0)
        with patch.object(runner.Path, 'read_bytes', return_value=b'echo stopping\n'), \
             patch.object(runner.subprocess, 'run', side_effect=[failure, success, success]) as run:
            self.assertEqual(runner.call_autodl_shutdown().returncode, 0)
        self.assertEqual([call.args[0] for call in run.call_args_list], [
            ['/usr/bin/shutdown'], ['/bin/bash', '-n', '/usr/bin/shutdown'],
            ['/bin/bash', '/usr/bin/shutdown']])

    def test_shutdown_does_not_fallback_for_permission_or_binary_errors(self):
        with patch.object(runner.subprocess, 'run', side_effect=OSError(errno.EACCES, 'denied')) as run:
            with self.assertRaises(OSError):
                runner.call_autodl_shutdown()
            self.assertEqual(run.call_count, 1)
        with patch.object(runner.subprocess, 'run', side_effect=OSError(errno.ENOEXEC, 'format')) as run, \
             patch.object(runner.Path, 'read_bytes', return_value=b'\x7fELF\x00'):
            with self.assertRaises(RuntimeError):
                runner.call_autodl_shutdown()
            self.assertEqual(run.call_count, 1)

    def test_variant_labels_and_invalid_lr(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, (a, b) in runner.VARIANTS.items():
                cfg = training_signature(config())
                cfg['model_args'].update(use_hypergcn=a, use_detgc=b)
                saved = write_run_results(directory, result()['epochs'], cfg, completed=True)
                self.assertEqual(saved['variant'], name)
        bad = result()
        bad['epochs'][-1]['learning_rate'] = 0.005
        with self.assertRaisesRegex(ValueError, 'learning-rate'):
            runner.checked_records(bad)

    def test_default_job_launches_only_combined_and_reuses_baseline(self):
        import argparse
        import yaml
        from tools import run_mti_experiment, benchmark_autodl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / 'old' / 'baseline_seed4321'
            baseline.mkdir(parents=True)
            (baseline.parent / 'source_revision.txt').write_text(runner.BASELINE_REVISION)
            data = root / 'data.npz'
            data.touch()
            cfg = yaml.safe_load((ROOT / 'config/nturgbd-cross-subject/joint_bimamba_baseline10.yaml').read_text())
            for split in ('train', 'test'):
                cfg[f'{split}_feeder_args']['data_path'] = str(data.resolve())
            old = result()
            old['run_config'] = training_signature(cfg)
            (baseline / 'results.json').write_text(json.dumps(old))
            before = (baseline / 'results.json').read_bytes()
            args = argparse.Namespace(output_root=root / 'new', baseline_dir=baseline, data_path=data,
                                      seed=4321, variant='combined', skip_preflight=True,
                                      shutdown_on_exit=False, shutdown_delay=300)
            launches = []
            def fake_run(command, **kwargs):
                if str(ROOT / 'main.py') in command:
                    launches.append(command)
                    launch = Path(command[command.index('--config') + 1])
                    actual = yaml.safe_load(launch.read_text())
                    self.assertTrue(actual['model_args']['use_hypergcn'])
                    self.assertTrue(actual['model_args']['use_detgc'])
                    completed = result()
                    completed['run_config'] = training_signature(actual)
                    (launch.parent / 'results.json').write_text(json.dumps(completed))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(run_mti_experiment, 'run_lock', return_value=nullcontext()), \
                 patch.object(benchmark_autodl, 'check_idle_gpu'), \
                 patch.object(runner.subprocess, 'run', side_effect=fake_run), \
                 patch.object(runner.subprocess, 'check_output', return_value=runner.BASELINE_REVISION):
                self.assertEqual(runner.run(args), 0)
            self.assertEqual(len(launches), 1)
            self.assertEqual((baseline / 'results.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
