"""CPU-only checks for the isolated B/C Norm screening run and its reference."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/manage_full_experiments.py'
SPEC = importlib.util.spec_from_file_location('bc_experiment_manager', SCRIPT)
manager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manager)


class BCRunSetupTest(unittest.TestCase):
    def setUp(self):
        manager.configure_suite('bc10')
        self.addCleanup(manager.configure_suite, 'ssm20')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def baseline_log(self):
        directory = self.root / 'work_dir/ntu60/xsub/full_baseline_20ep'
        directory.mkdir(parents=True)
        (directory / 'log.txt').write_text(
            'Training epoch: 9\nEval epoch: 9\nTop1: 72.36%\nTop5: 94.00%\n'
            'Training epoch: 10\nEval epoch: 10\nTop1: 74.99%\nTop5: 95.00%\n'
            'Training epoch: 19\nEval epoch: 19\nTop1: 86.12%\nTop5: 97.62%\n'
            'Training epoch: 20\nEval epoch: 20\nTop1: 85.86%\nTop5: 97.00%\n'
            'Best accuracy: 0.861231884057971\nEpoch number: 19\n', encoding='utf-8')
        return directory

    def test_prefix_ignores_later_scores_and_final_summary_without_loading_weights(self):
        directory = self.baseline_log()
        result = manager.parse_result(directory, epoch_limit=10)
        self.assertEqual(result['best_top1'], 74.99)
        self.assertEqual(result['best_epoch'], 10)
        self.assertEqual(result['best_top5'], 95.0)
        self.assertEqual(result['final_top1'], 74.99)
        self.assertEqual(result['last_top1'], 74.99)
        self.assertEqual(result['trained_epoch'], 10)
        self.assertEqual(result['eval_epoch'], 10)
        self.assertTrue(result['finished'])
        self.assertEqual([path.name for path in directory.iterdir()], ['log.txt'])

    def test_missing_prefix_epoch_is_not_reported_complete(self):
        directory = self.baseline_log()
        log = directory / 'log.txt'
        log.write_text(log.read_text().replace('Eval epoch: 10\nTop1: 74.99%\nTop5: 95.00%\n', ''), encoding='utf-8')
        result = manager.parse_result(directory, epoch_limit=10)
        self.assertFalse(result['finished'])
        self.assertEqual(result['best_top1'], 72.36)
        self.assertIsNone(result['final_top1'])

    def test_results_show_reference_prefix_and_candidate_difference(self):
        self.baseline_log()
        directory = manager.group_dir(self.root, 'ssm_bc_norm')
        directory.mkdir(parents=True)
        (directory / 'log.txt').write_text(
            'Training epoch: 10\nEval epoch: 10\nTop1: 76.50%\nTop5: 96.00%\n'
            'Best accuracy: 0.765\nEpoch number: 10\n', encoding='utf-8')
        (directory / 'runs-best.pt').touch()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            manager.print_results(self.root)
        text = output.getvalue()
        self.assertIn('baseline_first10\treference\t10\t74.9900\t10', text)
        self.assertIn('ssm_bc_norm\tcompleted\t10\t76.5000\t10', text)
        self.assertIn('+1.5100 percentage points', text)
        self.assertNotIn('86.1232', text)
        self.assertNotIn('epoch20_top1', text)

    def test_profile_keeps_old_defaults_and_uses_independent_new_paths(self):
        self.assertEqual(manager.MODES, ('ssm_bc_norm',))
        self.assertEqual(manager.EPOCHS, 10)
        self.assertEqual(manager.group_dir(self.root, 'ssm_bc_norm').name, 'full_ssm_bc_norm_10ep')
        self.assertEqual(manager.config_path(self.root, 'ssm_bc_norm').name, 'joint_ssm_bc_norm_10ep.yaml')
        bc_queue = manager.QUEUE_REL
        manager.configure_suite('ssm20')
        self.assertEqual(manager.MODES, ('baseline', 'ssm_bidir', 'ssm_motion', 'ssm_bidir_motion'))
        self.assertEqual(manager.EPOCHS, 20)
        self.assertEqual(manager.QUEUE_REL, Path('work_dir/ntu60/xsub/ssm_full_20ep_queue'))
        self.assertNotEqual(manager.QUEUE_REL, bc_queue)

    def test_results_wait_for_equal_budgets_before_printing_difference(self):
        self.baseline_log()
        directory = manager.group_dir(self.root, 'ssm_bc_norm')
        directory.mkdir(parents=True)
        (directory / 'log.txt').write_text(
            'Training epoch: 4\nEval epoch: 4\nTop1: 66.00%\n', encoding='utf-8')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            manager.print_results(self.root)
        self.assertIn('ssm_bc_norm\tincomplete\t4', output.getvalue())
        self.assertNotIn('percentage points', output.getvalue())

    def test_start_passes_suite_and_only_one_mode_to_detached_worker(self):
        queue = self.root / manager.QUEUE_REL
        queue.mkdir(parents=True)
        handle = (queue / 'queue.lock').open('a')
        with mock.patch.object(manager, 'os') as fake_os, \
                mock.patch.object(manager.Path, 'is_dir', return_value=True), \
                mock.patch.object(manager, 'lock_queue', return_value=handle), \
                mock.patch.object(manager, 'preflight') as preflight, \
                mock.patch.object(manager, 'detached_process', return_value=mock.Mock(pid=123)) as detached, \
                contextlib.redirect_stdout(io.StringIO()):
            fake_os.name = 'posix'
            manager.start_queue(self.root, queue)
        preflight.assert_called_once_with(self.root, ('ssm_bc_norm',))
        command = detached.call_args.args[0]
        self.assertEqual(command[2:5], ['--suite', 'bc10', 'run'])
        self.assertEqual(command[command.index('--modes') + 1:], ['ssm_bc_norm'])
        self.assertEqual(manager.read_state(queue)['suite'], 'bc10')

    def test_cli_selects_profile_before_building_mode_choices(self):
        with mock.patch.object(manager, 'ROOT', self.root), \
                mock.patch.object(sys, 'argv', [str(SCRIPT), '--suite', 'bc10', 'start']), \
                mock.patch.object(manager, 'start_queue') as start:
            self.assertEqual(manager.main(), 0)
        start.assert_called_once_with(self.root, self.root / manager.QUEUE_REL, ('ssm_bc_norm',))

    def test_status_and_results_are_read_only(self):
        for command in ('status', 'results'):
            with mock.patch.object(manager, 'ROOT', self.root), \
                    mock.patch.object(sys, 'argv', [str(SCRIPT), '--suite', 'bc10', command]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(), 0)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_preflight_rejects_silent_kwargs_fallback_for_bc_model(self):
        model = self.root / 'model/Hyperformer.py'
        model.parent.mkdir()
        model.write_text('class Model:\n    def __init__(self, **kwargs): pass\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'explicitly accept ssm_bc_norm_layers'):
            manager.check_sources(self.root)

    def test_preflight_rejects_silent_kwargs_fallback_for_bc_block(self):
        model = self.root / 'model/Hyperformer.py'
        model.parent.mkdir()
        model.write_text('class Model:\n    def __init__(self, ssm_bc_norm_layers=None): pass\n'
                         'class TCN_ViT_unit:\n    def __init__(self, ssm_args=None): pass\n', encoding='utf-8')
        (self.root / 'mamba_blocks.py').write_text(
            'class MambaBlock:\n    def __init__(self, **kwargs): pass\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'explicitly accept bc_norm'):
            manager.check_sources(self.root)

    def test_training_config_reuses_old_first_ten_epoch_recipe(self):
        config_dir = ROOT / 'config/nturgbd-cross-subject'
        baseline = yaml.safe_load((config_dir / 'joint_baseline_20ep.yaml').read_text(encoding='utf-8'))
        candidate = yaml.safe_load((config_dir / 'joint_ssm_bc_norm_10ep.yaml').read_text(encoding='utf-8'))
        self.assertEqual(candidate['num_epoch'], 10)
        self.assertEqual(candidate['work_dir'], './work_dir/ntu60/xsub/full_ssm_bc_norm_10ep')
        self.assertTrue(candidate['ssm_protect_addons'])
        extras = {'ssm_bc_norm_layers': [4], 'ssm_bc_norm_gain': 0.1,
                  'ssm_bc_norm_bias': 0.1, 'ssm_bc_norm_eps': 1e-6}
        for key, value in extras.items():
            self.assertEqual(candidate['model_args'][key], value)
        normalized = dict(candidate)
        normalized['work_dir'] = baseline['work_dir']
        normalized['num_epoch'] = baseline['num_epoch']
        normalized.pop('ssm_protect_addons')
        normalized['model_args'] = {key: value for key, value in candidate['model_args'].items() if key not in extras}
        self.assertEqual(normalized, baseline)
        for key in ('ctr_layers', 'fr_layers', 'ssm_bidir_layers', 'ssm_motion_layers'):
            self.assertEqual(candidate['model_args'][key], [])
        self.assertEqual(candidate['step'], [14, 18])
        self.assertGreater(min(candidate['step']), candidate['num_epoch'])


if __name__ == '__main__':
    unittest.main()
