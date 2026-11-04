"""Budget/supervisor checks without CUDA or signals to real processes."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import training_signature, validate_screening, SIGNATURE_KEYS

spec = importlib.util.spec_from_file_location('comparison_controller', ROOT / 'tools/run_mti_comparison.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class MTIBudgetTests(unittest.TestCase):
    def test_budget_is_not_a_training_setting(self):
        values = {key: None for key in SIGNATURE_KEYS}
        signature = training_signature(values)
        values['stop_after_epoch'] = 9
        self.assertEqual(signature, training_signature(values))
        args = argparse.Namespace(
            screening=True, fixed_lr=True, warm_up_epoch=0, step=[], base_lr=0.005,
            num_epoch=15, precision='bf16', device=[0], ema=False, weights=None,
            start_epoch=3, resume='runs-latest.pt', save_latest=True,
            strict_rng_resume=True, persistent_workers=False, stop_after_epoch=9,
        )
        validate_screening(args)
        args.stop_after_epoch = 16
        with self.assertRaises(ValueError):
            validate_screening(args)

    def test_unknown_training_child_is_never_signalled(self):
        with patch.object(controller, 'verified_launcher', return_value=11), \
             patch.object(controller.Path, 'iterdir', return_value=[]), \
             patch.object(controller.os, 'kill') as kill:
            with self.assertRaises(RuntimeError):
                controller.stop_verified_baseline(11, Path('/unused/baseline_seed4321'))
            kill.assert_not_called()

    def test_verified_queue_is_stopped_before_verified_child(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'baseline_seed4321'
            work.mkdir()
            (work / 'results.json').write_text(json.dumps({'epochs': [{'epoch': 9}]}))
            argv = {
                42: ['python', str(ROOT / 'main.py'), '--config', str(work / 'launch_config.yaml')],
                22: ['bash', '-c', f'set -e; tools/run_mti_experiment.py --variant baseline; '
                     f'tools/run_mti_experiment.py --variant mti; # {work.parent.name}'],
                11: [],
            }
            with patch.object(controller, 'verified_launcher', return_value=11), \
                 patch.object(controller.Path, 'iterdir', return_value=[types.SimpleNamespace(name='42')]), \
                 patch.object(controller, 'parent_pid', side_effect=lambda pid: {42: 11, 11: 22}.get(pid)), \
                 patch.object(controller, 'process_args', side_effect=lambda pid: argv.get(pid, [])), \
                 patch.object(controller.os, 'kill') as kill:
                controller.stop_verified_baseline(11, work)
                self.assertEqual([call.args for call in kill.call_args_list],
                                 [(22, signal.SIGTERM), (42, signal.SIGTERM)])
                self.assertEqual(json.loads((work / 'budget_stop.json').read_text())['completed_epoch'], 9)

    def test_controller_resumes_baseline_then_starts_mti_with_same_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            baseline = root / 'baseline_seed4321'
            baseline.mkdir()
            (baseline / 'results.json').write_text(json.dumps({'epochs': [{'epoch': 3}]}))
            (baseline / 'runs-latest.pt').touch()
            calls = []

            def launch(variant, output, seed, budget, resume=False):
                calls.append((variant, seed, budget, resume))
                target = output / f'{variant}_seed{seed}'
                target.mkdir(exist_ok=True)
                (target / 'results.json').write_text(json.dumps({'epochs': [{'epoch': budget}]}))

            fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
            with patch.dict(sys.modules, {'fcntl': fcntl}), \
                 patch.object(sys, 'argv', ['controller', '--output-root', str(root), '--stop-after-epoch', '9']), \
                 patch.object(controller, 'verified_launcher', return_value=None), \
                 patch.object(controller, 'launch', side_effect=launch):
                controller.main()
            self.assertEqual(calls, [('baseline', 4321, 9, True), ('mti', 4321, 9, False)])


if __name__ == '__main__':
    unittest.main()
