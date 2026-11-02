"""Standard-library tests for the five-epoch context/retain stage rules."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import algorithm_protocol as protocol
from tools import manage_algorithm_screening as manager


def make_curve(root, scores, name=None, extra=''):
    directory = Path(root) / (name or f'curve{len(list(Path(root).iterdir()))}')
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for epoch, score in enumerate(scores, 1):
        lines.extend((f'Training epoch: {epoch}',
                      f'Training: {100 + epoch} seconds, 5 samples/second',
                      'Peak allocated CUDA memory: 1000 MiB.',
                      f'Eval epoch: {epoch}', f'Top1: {score:.4f}%', f'Top5: {score + 10:.4f}%'))
    lines.extend((f'Best accuracy: {max(scores) / 100}', f'Epoch number: {scores.index(max(scores)) + 1}',
                  extra))
    (directory / 'log.txt').write_text('\n'.join(lines), encoding='utf-8')
    return manager.parse_curve(directory)


class StageRuleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def comparison(self, candidate, baseline):
        return manager.compare(candidate, baseline, 5)

    def test_five_epoch_promotion_needs_mean_wins_and_final_three(self):
        baseline = make_curve(self.root, [50] * 5)
        strong = self.comparison(make_curve(self.root, [50.1, 50.1, 50.6, 50.7, 50.7]), baseline)
        self.assertEqual(manager.classify5(strong), '值得追加')
        # Same mean, but epoch three is behind: two of the final three only.
        uneven = self.comparison(make_curve(self.root, [50.9, 50.9, 49.9, 51.0, 51.0]), baseline)
        self.assertGreaterEqual(uneven['delta_last3'], 0.5)
        self.assertNotEqual(manager.classify5(uneven), '值得追加')
        # Only three of five epochs ahead.
        narrow = self.comparison(make_curve(self.root, [49.0, 49.0, 51.0, 51.0, 51.0]), baseline)
        self.assertEqual(manager.classify5(narrow), '观察')
        small = self.comparison(make_curve(self.root, [50.1] * 5), baseline)
        self.assertEqual(manager.classify5(small), '观察')
        behind = self.comparison(make_curve(self.root, [49.9] * 5), baseline)
        self.assertEqual(manager.classify5(behind), '暂缓')
        self.assertEqual(manager.classify5(strong, False), '先修模块')

    def test_incomplete_five_epoch_curve_never_promotes(self):
        baseline = make_curve(self.root, [50] * 5)
        partial = self.comparison(make_curve(self.root, [51] * 4), baseline)
        self.assertEqual(manager.classify5(partial), '未完成')
        self.assertIsNone(manager.choose_winner5({'context_write': partial}))

    def test_combination_must_beat_the_stronger_single_by_the_margin(self):
        baseline = make_curve(self.root, [50] * 5)
        single = self.comparison(make_curve(self.root, [50.1, 50.1, 50.6, 50.6, 50.6]), baseline)
        weak_combo = self.comparison(make_curve(self.root, [50.1, 50.1, 50.7, 50.7, 50.7]), baseline)
        self.assertGreaterEqual(weak_combo['delta_last3'] - single['delta_last3'], 0)
        self.assertLess(weak_combo['delta_last3'], single['delta_last3'] + manager.COMBINATION_MARGIN)
        self.assertEqual(manager.choose_winner5({'context_write': single,
                                                 'context_retain_write': weak_combo}), 'context_write')
        strong_combo = self.comparison(make_curve(self.root, [50.5, 50.5, 51.2, 51.2, 51.2]), baseline)
        self.assertEqual(manager.choose_winner5({'context_write': single,
                                                 'context_retain_write': strong_combo}),
                         'context_retain_write')

    def test_winner_ties_prefer_lower_cost(self):
        baseline = make_curve(self.root, [50] * 5)
        first = self.comparison(make_curve(self.root, [50.1] * 2 + [51.0] * 3), baseline)
        second = json.loads(json.dumps(first))
        second['training_hours'] = first['training_hours'] + 1
        self.assertEqual(manager.choose_winner5({'context_write': first,
                                                 'retain_write': second}), 'context_write')

    def test_legacy_reference_requires_architecture_proof(self):
        queue = manager.queue_dir(self.root, 4321)
        queue.mkdir(parents=True)
        external = self.root / 'external'
        external.mkdir()
        (queue / 'baseline_reference.json').write_text(
            json.dumps(dict(directory=str(external), legacy=True)), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'architecture proof'):
            manager.validate_reference(self.root, 4321, {'seed': 4321}, 5)


class StageLauncherTest(unittest.TestCase):
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
        self.protocol.validate_legacy_baseline.side_effect = self.validate_legacy
        self.protocol.validate_baseline_manifest.side_effect = self.validate_baseline
        for patcher in (mock.patch.object(manager, 'protocol_module', return_value=self.protocol),
                        mock.patch.object(manager, 'verify_baseline_architecture',
                                          return_value=dict(ok=True, parameters=1234))):
            patcher.start()
            self.addCleanup(patcher.stop)

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

    def validate_legacy(self, root, directory, seed, expected):
        (Path(directory) / 'config.yaml').write_text('{}', encoding='utf-8')
        return dict(legacy_ok=True)

    def validate_baseline(self, root, directory, seed, expected):
        self.validate_identity(self.protocol.load_manifest(Path(directory)),
                               self.identity(root, 'baseline', seed), kind='baseline')

    def make_receipt(self, mode, passed=True, steps=40):
        record = dict(self.identity(self.root, mode, self.seed), passed=passed, checked_steps=steps)
        path = manager.receipt_path(self.root, mode, self.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))

    def complete(self, mode, epochs, increment=0):
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

    def test_stage_five_preflight_requires_baseline_first_and_receipts(self):
        self.make_receipt('context_write')
        self.complete('baseline', 5)
        with self.assertRaisesRegex(ValueError, 'start5 accepts only'):
            manager.preflight(self.root, self.seed, ('joint_state',), 5, False)
        modes, _, plans = manager.preflight(self.root, self.seed,
                                            ('baseline', 'context_write'), 5, False)
        self.assertEqual(modes, ('baseline', 'context_write'))
        self.assertEqual(plans['baseline'], ('skip', None))
        self.assertEqual(plans['context_write'][0], 'new')
        with self.assertRaisesRegex(RuntimeError, 'Missing signal-check'):
            manager.preflight(self.root, self.seed, ('retain_write',), 5, False)

    def test_stage_five_preflight_rejects_candidate_recipe_drift(self):
        self.make_receipt('context_write')

        def identity(root, mode, seed):
            record = self.identity(root, mode, seed)
            if mode == 'context_write':
                record['recipe_fingerprint'] = 'changed-batch-or-precision'
            return record

        self.protocol.current_identity.side_effect = identity
        with self.assertRaisesRegex(RuntimeError, 'paired identity differs.*recipe_fingerprint'):
            manager.preflight(self.root, self.seed, ('baseline', 'context_write'), 5, False)

    def test_five_epoch_run_extends_only_its_epoch_five_latest(self):
        directory = self.complete('context_write', 5)
        expected = self.identity(self.root, 'context_write', self.seed)
        with mock.patch.object(manager, 'checkpoint_header',
                               return_value=dict(epoch=4, has_model=True,
                                                 has_optimizer=True, has_rng=True)):
            self.assertEqual(manager.plan_mode(self.root, 'context_write', self.seed, 10, expected),
                             ('resume', directory / 'runs-latest.pt'))
        with mock.patch.object(manager, 'checkpoint_header',
                               return_value=dict(epoch=2, has_model=True,
                                                 has_optimizer=True, has_rng=True)):
            with self.assertRaisesRegex(RuntimeError, 'actual epoch-5'):
                manager.plan_mode(self.root, 'context_write', self.seed, 10, expected)

    def test_extension_requires_the_five_epoch_winner(self):
        self.complete('baseline', 10)
        self.complete('context_write', 5, increment=1)
        self.complete('retain_write', 5, increment=.1)
        self.make_receipt('context_write')
        self.make_receipt('retain_write')
        self.assertEqual(manager.extension_modes_stage(self.root, self.seed, 'context_write'),
                         ('context_write',))
        with self.assertRaisesRegex(RuntimeError, 'single eligible five-epoch winner'):
            manager.extension_modes_stage(self.root, self.seed, 'retain_write')
        queue = manager.queue_dir(self.root, self.seed)
        queue.mkdir(exist_ok=True)
        (queue / 'stage_extension.json').write_text(json.dumps({'mode': 'retain_write'}))
        with self.assertRaisesRegex(RuntimeError, 'refusing a second candidate'):
            manager.extension_modes_stage(self.root, self.seed, 'context_write')

    def test_legacy_baseline_freeze_records_what_was_verified(self):
        external = self.root / 'external'
        (external / 'run').mkdir(parents=True)
        (external / 'run' / 'config.yaml').write_text('{}', encoding='utf-8')
        lines = []
        for epoch in range(1, 11):
            lines.extend((f'Training epoch: {epoch}', f'Eval epoch: {epoch}',
                          f'Top1: {50 + epoch}%', 'Top5: 95%'))
        lines.extend(('Best accuracy: .7', 'Epoch number: 10'))
        (external / 'run' / 'log.txt').write_text('\n'.join(lines))
        with contextlib.redirect_stdout(io.StringIO()):
            manager.reuse_legacy_baseline(self.root, self.seed, external / 'run')
        record = json.loads((manager.queue_dir(self.root, self.seed)
                             / 'baseline_reference.json').read_text())
        self.assertTrue(record['legacy'])
        self.assertTrue(record['architecture_verified'])
        self.assertEqual(record['available_epochs'], 10)
        self.assertEqual(record['architecture']['parameters'], 1234)
        # A second, different reference may not replace the frozen one.
        other = self.root / 'other'
        other.mkdir()
        (other / 'config.yaml').write_text('{}', encoding='utf-8')
        (other / 'log.txt').write_text((external / 'run' / 'log.txt').read_text())
        with self.assertRaisesRegex(RuntimeError, 'different baseline reference'):
            manager.reuse_legacy_baseline(self.root, self.seed, other)

    def test_read_only_stage_commands_create_nothing(self):
        for command in ('results5',):
            with mock.patch.object(manager, 'ROOT', self.root), \
                    mock.patch.object(sys, 'argv', ['manager', command]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(manager.main(), 0)
            self.assertEqual([path for path in self.root.iterdir() if path.name != '.cache'], [])

    def test_all_protocol_modes_have_a_configuration(self):
        for mode in protocol.MODES:
            path = ROOT / 'config/nturgbd-cross-subject' / f'joint_algorithm_{mode}.yaml'
            self.assertTrue(path.is_file(), path)
            config = json.loads(path.read_text(encoding='utf-8'))
            args = config['model_args']
            enabled = [name for name, value in args.items()
                       if (name in ('ctr_layers', 'fr_layers')
                           or name.startswith('ssm_') and name.endswith('_layers')) and value]
            if mode == 'baseline':
                self.assertEqual(enabled, [])
            elif mode == 'context_write':
                self.assertEqual(enabled, ['ssm_context_layers'])
                self.assertEqual(args['ssm_context_layers'], [4])
            elif mode == 'retain_write':
                self.assertEqual(enabled, ['ssm_retain_layers'])
            elif mode == 'context_retain_write':
                self.assertEqual(enabled, ['ssm_context_layers', 'ssm_retain_layers'])
            else:
                self.assertEqual(len(enabled), 1, mode)


if __name__ == '__main__':
    unittest.main()
