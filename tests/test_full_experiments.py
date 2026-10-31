"""CPU-only behavioral checks for the detached experiment manager."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / 'tools/manage_full_experiments.py'
SPEC = importlib.util.spec_from_file_location('full_experiments', SCRIPT)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


class ExperimentManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.queue = self.root / manager.QUEUE_REL

    def complete(self, mode, best=0.85):
        directory = manager.group_dir(self.root, mode)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'log.txt').write_text(
            'Training epoch: 19\nEval epoch: 19\nTop1: 85.00%\nTop5: 97.00%\n'
            'Training epoch: 20\nEval epoch: 20\nTop1: 84.00%\nTop5: 96.00%\n'
            f'Best accuracy: {best}\nEpoch number: 19\n', encoding='utf-8')
        (directory / 'runs-best.pt').touch()
        return directory

    def run_fake(self, modes, fail=False):
        self.queue.mkdir(parents=True, exist_ok=True)
        calls = []

        def launch(command, **kwargs):
            calls.append((command, kwargs))
            mode = next(mode for mode in manager.MODES if str(manager.config_path(self.root, mode)) in command)
            if not fail:
                self.complete(mode)
            return mock.Mock(pid=12345, wait=mock.Mock(return_value=2 if fail else 0))

        with mock.patch.object(manager, 'training_processes', return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            code = manager.run_modes(self.root, self.queue, modes=modes, spawn=launch)
        return code, calls

    def test_parse_metrics_distinguishes_best_epoch_and_final_epoch(self):
        directory = self.complete('baseline', best=0.850123)
        result = manager.parse_result(directory)
        self.assertTrue(result['finished'])
        self.assertEqual(result['trained_epoch'], 20)
        self.assertEqual(result['best_epoch'], 19)
        self.assertAlmostEqual(result['best_top1'], 85.0123)
        self.assertEqual(result['best_top5'], 97.0)
        self.assertEqual(result['final_top1'], 84.0)

    def test_twenty_epochs_without_final_summary_are_not_complete(self):
        directory = self.complete('baseline')
        (directory / 'log.txt').write_text('Training epoch: 20\nEval epoch: 20\nTop1: 85.00%\n')
        self.assertFalse(manager.parse_result(directory)['finished'])

    def test_plan_skips_completed_and_resumes_only_latest_checkpoint(self):
        self.complete('baseline')
        self.assertEqual(manager.plan_mode(self.root, 'baseline'), ('skip', None))
        directory = manager.group_dir(self.root, 'ssm_bidir')
        directory.mkdir(parents=True)
        (directory / 'runs-best.pt').touch()
        with self.assertRaisesRegex(RuntimeError, 'without runs-latest'):
            manager.plan_mode(self.root, 'ssm_bidir')
        latest = directory / 'runs-latest.pt'
        latest.touch()
        self.assertEqual(manager.plan_mode(self.root, 'ssm_bidir'), ('resume', latest))

    def test_default_queue_runs_all_four_sequentially(self):
        code, calls = self.run_fake(manager.MODES)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 4)
        for mode, (command, kwargs) in zip(manager.MODES, calls):
            self.assertEqual(command[:3], [sys.executable, '-u', str(self.root / 'main.py')])
            self.assertIn(str(manager.config_path(self.root, mode)), command)
            self.assertNotIn('--resume', command)
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
            self.assertEqual(kwargs['env']['PYTHONUNBUFFERED'], '1')
        self.assertEqual(manager.read_state(self.queue)['status'], 'completed')

    def test_subset_saves_budget_and_keeps_requested_order(self):
        modes = ('baseline', 'ssm_bidir_motion')
        code, calls = self.run_fake(modes)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn(str(manager.config_path(self.root, 'baseline')), calls[0][0])
        self.assertIn(str(manager.config_path(self.root, 'ssm_bidir_motion')), calls[1][0])
        self.assertFalse(manager.group_dir(self.root, 'ssm_bidir').exists())
        self.assertEqual(manager.read_state(self.queue)['modes'], list(modes))

    def test_failure_stops_before_spending_on_next_group(self):
        code, calls = self.run_fake(manager.MODES, fail=True)
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(manager.group_dir(self.root, 'ssm_bidir').exists())
        state = manager.read_state(self.queue)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(state['exit_code'], 2)

    def test_restart_skips_finished_group_and_passes_resume(self):
        self.complete('baseline')
        directory = manager.group_dir(self.root, 'ssm_bidir_motion')
        directory.mkdir(parents=True)
        latest = directory / 'runs-latest.pt'
        latest.touch()
        (directory / 'console.log').write_text('old output\n')
        code, calls = self.run_fake(('baseline', 'ssm_bidir_motion'))
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0][-2:], ['--resume', str(latest)])
        self.assertEqual((directory / 'console.log').read_text(), 'old output\n')

    def test_read_only_status_and_results_do_not_create_queue(self):
        for command in ('status', 'results'):
            with mock.patch.object(manager, 'ROOT', self.root), mock.patch.object(sys, 'argv', [str(SCRIPT), command]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(), 0)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_detached_launcher_uses_new_session_and_inherits_lock(self):
        with mock.patch.object(manager.subprocess, 'Popen') as popen:
            manager.detached_process(['python', 'worker'], self.root, 'log-handle', 42)
        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(kwargs['pass_fds'], (42,))

    def test_source_preflight_rejects_silent_kwargs_fallback(self):
        model = self.root / 'model/Hyperformer.py'
        model.parent.mkdir()
        model.write_text('class Model:\n    def __init__(self, **kwargs): pass\n')
        with self.assertRaisesRegex(RuntimeError, 'explicitly accept ssm_bidir_layers'):
            manager.check_sources(self.root)

    def test_preflight_rejects_existing_checkout_training(self):
        with mock.patch.object(manager, 'check_sources'), mock.patch.object(manager, 'training_processes', return_value=[123]):
            with self.assertRaisesRegex(RuntimeError, 'Stop the old foreground'):
                manager.preflight(self.root)

    def test_source_preflight_rejects_old_training_program(self):
        model = self.root / 'model/Hyperformer.py'
        model.parent.mkdir()
        model.write_text('class Model:\n    def __init__(self, ssm_bidir_layers=None): pass\n'
                         'class TCN_ViT_unit:\n    def __init__(self, ssm_args=None): pass\n')
        (self.root / 'mamba_blocks.py').write_text(
            'class MambaBlock:\n    def __init__(self, bidirectional=False): pass\n')
        (self.root / 'main.py').write_text('print("old training program")\n')
        with self.assertRaisesRegex(RuntimeError, '--save-latest is missing'):
            manager.check_sources(self.root)


if __name__ == '__main__':
    unittest.main()
