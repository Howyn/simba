"""Standard-library tests for algorithm budget decisions and safe launching."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import manage_algorithm_screening as manager


class AlgorithmResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def curve(self, scores, extra=''):
        directory = self.root / str(len(list(self.root.iterdir())))
        directory.mkdir()
        lines = []
        for epoch, score in enumerate(scores, 1):
            lines.extend((f'Training epoch: {epoch}',
                          f'Training: {100 + epoch} seconds, 5 samples/second',
                          'Peak allocated CUDA memory: 1000 MiB.',
                          f'Eval epoch: {epoch}', f'Top1: {score:.4f}%', f'Top5: {score + 10:.4f}%'))
        lines.extend(('Eval epoch: 1', 'Top1: 99.99%', 'Top5: 100.00%',
                      f'Best accuracy: {max(scores) / 100}',
                      f'Epoch number: {scores.index(max(scores)) + 1}', extra))
        (directory / 'log.txt').write_text('\n'.join(lines), encoding='utf-8')
        return manager.parse_curve(directory)

    def test_final_best_reevaluation_does_not_overwrite_first_epoch(self):
        result = self.curve(list(range(50, 60)))
        self.assertEqual(result['scores'][1], 50)
        self.assertEqual(result['scores'][10], 59)
        self.assertTrue(manager.complete(result, 10))

    def test_prefix_does_not_leak_later_best(self):
        curve = self.curve(list(range(50, 70)))
        first = manager.metrics(curve, 10)
        self.assertEqual(first['best'], 59)
        self.assertEqual(first['e10'], 59)
        self.assertEqual(first['last3'], 58)
        self.assertIsNone(first['e20'])
        full = manager.metrics(curve, 20)
        self.assertEqual(full['last3'], 68)
        self.assertEqual(full['last5'], 67)
        self.assertEqual(full['post_second_decay'], 68.5)

    def test_incomplete_prefix_does_not_produce_selection(self):
        baseline = self.curve([50] * 10)
        candidate = self.curve([51] * 9)
        result = manager.compare(candidate, baseline)
        self.assertEqual(manager.classify(result), '未完成')
        self.assertIsNone(manager.choose_winner({'joint_state': result}))

    def test_best_peak_alone_is_not_enough(self):
        baseline = self.curve([50] * 10)
        candidate = self.curve([49, 49, 49, 49, 49, 60, 49, 49, 49, 49])
        result = manager.compare(candidate, baseline)
        self.assertGreater(result['delta_best'], 0)
        self.assertEqual(manager.classify(result), '暂缓')

    def test_strong_and_borderline_rules(self):
        baseline = self.curve([50] * 10)
        strong = self.curve([50.1] * 7 + [50.5] * 3)
        self.assertEqual(manager.classify(manager.compare(strong, baseline)), '值得追加')
        borderline = self.curve([50.1] * 7 + [50.3] * 3)
        self.assertEqual(manager.classify(manager.compare(borderline, baseline)), '值得追加')
        inconsistent = self.curve([49.9] * 7 + [50.4] * 3)
        self.assertEqual(manager.classify(manager.compare(inconsistent, baseline)), '观察')
        self.assertEqual(manager.classify(manager.compare(strong, baseline), False), '先修模块')

    def test_only_one_winner_and_ties_use_cost(self):
        baseline = self.curve([50] * 10)
        lower = manager.compare(self.curve([50.6] * 10), baseline)
        higher = manager.compare(self.curve([50.8] * 10), baseline)
        self.assertEqual(manager.choose_winner({'joint_state': lower, 'dual_scale': higher}), 'dual_scale')
        higher['delta_last3'] = lower['delta_last3']
        higher['training_hours'] = lower['training_hours'] + 1
        self.assertEqual(manager.choose_winner({'joint_state': lower, 'dual_scale': higher}), 'joint_state')

    def test_readonly_commands_do_not_create_files(self):
        for command in ('status', 'results'):
            with mock.patch.object(manager, 'ROOT', self.root), \
                    mock.patch.object(sys, 'argv', ['manager', command]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(), 0)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_summary_is_required_to_call_a_run_complete(self):
        curve = self.curve([50] * 10)
        curve['summary_epoch'] = 9
        self.assertFalse(manager.complete(curve, 10))


class AlgorithmLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = 4321
        self.protocol = mock.Mock()
        self.protocol.current_identity.side_effect = self.identity
        self.protocol.load_manifest.side_effect = lambda path: json.loads((path / 'run_manifest.json').read_text())
        self.protocol.write_manifest.side_effect = self.write_manifest
        self.protocol.validate_identity.side_effect = self.validate_identity
        self.protocol.validate_baseline_manifest.side_effect = self.validate_baseline
        self.patch = mock.patch.object(manager, 'protocol_module', return_value=self.protocol)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def identity(self, root, mode, seed):
        return dict(schema_version=1, mode=mode, seed=seed, source_sha256='source',
                    config_sha256=mode + '-config', dataset_sha256='data',
                    recipe_fingerprint='recipe', device={'name': '4090'})

    def write_manifest(self, directory, identity):
        (directory / 'run_manifest.json').write_text(json.dumps(identity))

    def validate_identity(self, record, expected, kind='receipt'):
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f'{kind}: {key} mismatch')

    def validate_baseline(self, root, directory, seed, expected):
        record = self.protocol.load_manifest(directory)
        self.validate_identity(record, self.identity(root, 'baseline', seed), kind='baseline')

    def make_receipt(self, mode, passed=True, steps=40, **changes):
        record = dict(self.identity(self.root, mode, self.seed), passed=passed, checked_steps=steps)
        record.update(changes)
        path = manager.receipt_path(self.root, mode, self.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))

    def complete(self, mode, epochs=10, increment=0):
        directory = manager.group_dir(self.root, mode, self.seed)
        directory.mkdir(parents=True, exist_ok=True)
        self.write_manifest(directory, self.identity(self.root, mode, self.seed))
        lines = []
        for epoch in range(1, epochs + 1):
            lines.extend((f'Training epoch: {epoch}', f'Eval epoch: {epoch}',
                          f'Top1: {50 + epoch + increment}%', 'Top5: 95%'))
        lines.extend(('Best accuracy: .7', f'Epoch number: {epochs}'))
        (directory / 'log.txt').write_text('\n'.join(lines))
        (directory / 'runs-latest.pt').touch()
        return directory

    def test_receipt_requires_pass_40_updates_and_matching_hashes(self):
        expected = self.identity(self.root, 'joint_state', self.seed)
        with self.assertRaisesRegex(RuntimeError, 'Missing signal-check'):
            manager.validate_receipt(self.root, 'joint_state', self.seed, expected)
        self.make_receipt('joint_state', steps=39)
        with self.assertRaisesRegex(RuntimeError, '40 updates'):
            manager.validate_receipt(self.root, 'joint_state', self.seed, expected)
        self.make_receipt('joint_state', source_sha256='modified')
        with self.assertRaisesRegex(ValueError, 'source_sha256 mismatch'):
            manager.validate_receipt(self.root, 'joint_state', self.seed, expected)
        self.make_receipt('joint_state')
        manager.validate_receipt(self.root, 'joint_state', self.seed, expected)

    def test_baseline_must_precede_first_candidate(self):
        self.make_receipt('joint_state')
        with self.assertRaisesRegex(RuntimeError, 'Run baseline first'):
            manager.preflight(self.root, self.seed, ('joint_state',), 10, False)
        with self.assertRaisesRegex(RuntimeError, 'list baseline first'):
            manager.preflight(self.root, self.seed, ('joint_state', 'baseline'), 10, False)

    def test_existing_results_without_manifest_cannot_be_overwritten(self):
        directory = manager.group_dir(self.root, 'baseline', self.seed)
        directory.mkdir(parents=True)
        (directory / 'log.txt').write_text('some old run')
        with self.assertRaises(FileNotFoundError):
            manager.plan_mode(self.root, 'baseline', self.seed, 10,
                              self.identity(self.root, 'baseline', self.seed))

    def test_queue_runs_only_explicit_modes_sequentially(self):
        self.make_receipt('joint_state')
        calls = []
        def spawn(command, **kwargs):
            calls.append((command, kwargs))
            mode = 'baseline' if 'joint_algorithm_baseline.yaml' in ' '.join(command) else 'joint_state'
            self.complete(mode, increment=1 if mode == 'joint_state' else 0)
            return mock.Mock(pid=1234, wait=mock.Mock(return_value=0))
        with mock.patch.object(manager.legacy, 'training_processes', return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            code = manager.run_modes(self.root, self.seed, ('baseline', 'joint_state'), 10, spawn)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        self.assertFalse(manager.group_dir(self.root, 'dual_scale', self.seed).exists())
        for command, options in calls:
            self.assertIn('--num-epoch', command)
            self.assertIn('10', command)
            self.assertNotIn('--resume', command)
            self.assertTrue(options['start_new_session'])

    def test_preflight_rejects_candidate_recipe_drift_before_baseline(self):
        self.make_receipt('joint_state')
        def identity(root, mode, seed):
            result = self.identity(root, mode, seed)
            if mode == 'joint_state':
                result['recipe_fingerprint'] = 'changed-batch-or-precision'
            return result
        self.protocol.current_identity.side_effect = identity
        with self.assertRaisesRegex(RuntimeError, 'paired identity differs.*recipe_fingerprint'):
            manager.preflight(self.root, self.seed, ('baseline', 'joint_state'), 10, False)
        self.assertFalse(manager.group_dir(self.root, 'baseline', self.seed).exists())

    def test_source_changes_during_baseline_stop_later_candidate(self):
        self.make_receipt('joint_state')
        changed = False
        def identity(root, mode, seed):
            record = self.identity(root, mode, seed)
            if changed:
                record['source_sha256'] = 'changed-after-baseline'
            return record
        self.protocol.current_identity.side_effect = identity
        def spawn(command, **kwargs):
            nonlocal changed
            self.complete('baseline')
            changed = True
            return mock.Mock(pid=1234, wait=mock.Mock(return_value=0))
        spawn_mock = mock.Mock(side_effect=spawn)
        with mock.patch.object(manager.legacy, 'training_processes', return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'queue source/config/data: source_sha256 mismatch'):
                manager.run_modes(self.root, self.seed, ('baseline', 'joint_state'), 10, spawn_mock)
        self.assertEqual(spawn_mock.call_count, 1)
        self.assertFalse(manager.group_dir(self.root, 'joint_state', self.seed).exists())

    def test_failure_stops_before_next_group(self):
        self.make_receipt('joint_state')
        spawn = mock.Mock(return_value=mock.Mock(pid=1234, wait=mock.Mock(return_value=3)))
        with mock.patch.object(manager.legacy, 'training_processes', return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            code = manager.run_modes(self.root, self.seed, ('baseline', 'joint_state'), 10, spawn)
        self.assertEqual(code, 3)
        self.assertEqual(spawn.call_count, 1)
        self.assertFalse(manager.group_dir(self.root, 'joint_state', self.seed).exists())

    def test_extend_requires_latest_epoch_ten_and_rng(self):
        directory = self.complete('joint_state', increment=1)
        expected = self.identity(self.root, 'joint_state', self.seed)
        with mock.patch.object(manager, 'checkpoint_header', return_value=dict(epoch=8, has_model=True, has_optimizer=True, has_rng=True)):
            with self.assertRaisesRegex(RuntimeError, 'actual epoch-10'):
                manager.plan_mode(self.root, 'joint_state', self.seed, 20, expected)
        with mock.patch.object(manager, 'checkpoint_header', return_value=dict(epoch=9, has_model=True, has_optimizer=True, has_rng=False)):
            with self.assertRaisesRegex(RuntimeError, 'RNG state is missing'):
                manager.plan_mode(self.root, 'joint_state', self.seed, 20, expected)
        with mock.patch.object(manager, 'checkpoint_header', return_value=dict(epoch=9, has_model=True, has_optimizer=True, has_rng=True)):
            self.assertEqual(manager.plan_mode(self.root, 'joint_state', self.seed, 20, expected),
                             ('resume', directory / 'runs-latest.pt'))

    def test_only_single_eligible_winner_can_extend(self):
        self.complete('baseline')
        self.complete('joint_state', increment=1)
        self.complete('dual_scale', increment=.1)
        self.make_receipt('joint_state')
        self.make_receipt('dual_scale')
        self.assertEqual(manager.extension_modes(self.root, self.seed, 'joint_state'), ('baseline', 'joint_state'))
        with self.assertRaisesRegex(RuntimeError, 'single eligible winner'):
            manager.extension_modes(self.root, self.seed, 'dual_scale')
        queue = manager.queue_dir(self.root, self.seed)
        queue.mkdir(exist_ok=True)
        (queue / 'extension_selection.json').write_text(json.dumps({'mode': 'dual_scale'}))
        with self.assertRaisesRegex(RuntimeError, 'refusing a second candidate'):
            manager.extension_modes(self.root, self.seed, 'joint_state')

    def test_legacy_external_baseline_refused_and_valid20_can_be_frozen(self):
        external = self.root / 'external'
        external.mkdir()
        with self.assertRaises(FileNotFoundError):
            manager.reuse_baseline(self.root, self.seed, external)
        baseline = self.complete('baseline', epochs=20)
        # Move the fixture to simulate a previously completed qualified run.
        baseline.rename(external / 'run')
        with contextlib.redirect_stdout(io.StringIO()):
            manager.reuse_baseline(self.root, self.seed, external / 'run')
        self.assertEqual(manager.baseline_directory(self.root, self.seed), (external / 'run').resolve())

    def test_missing_explicit_modes_is_argument_error(self):
        with mock.patch.object(sys, 'argv', ['manager', 'start10']), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                manager.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
