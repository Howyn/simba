"""CPU-only identity, cache, and baseline reuse checks (stdlib only)."""

import argparse
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('algorithm_protocol_subject',
                                             ROOT / 'tools/algorithm_protocol.py')
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


DEVICE = dict(name='NVIDIA GeForce RTX 4090', capability=[8, 9],
              torch='2.5.1+cu124', cuda='12.4', python='3.12.4', cudnn=90100)


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'ntu.npz'
        self.dataset.write_bytes(b'fake NPZ bytes for hashing only')
        self.base_config = dict(
            work_dir='./work_dir/baseline', seed=4321, device=[0], num_epoch=10,
            model='model.Hyperformer.Model',
            model_args=dict(num_class=60, num_point=25, num_person=2,
                            graph='graph.ntu_rgb_d.Graph', ctr_layers=[], fr_layers=[],
                            ssm_joint_state_layers=[], ssm_joint_state_weight=0.1,
                            ssm_dual_scale_layers=[], ssm_slow_scale=0.5,
                            ssm_scale_gate_init=0.1),
            feeder='feeders.feeder_ntu.Feeder',
            train_feeder_args=dict(data_path=str(self.dataset), split='train', window_size=64),
            test_feeder_args=dict(data_path=str(self.dataset), split='test', window_size=64),
            optimizer='SGD', step=[14, 18], base_lr=0.025,
            weight_decay=0.0004, warm_up_epoch=5,
            batch_size=16, test_batch_size=16, num_worker=8,
            tf32=False, precision='fp32', eval_interval=1,
        )
        for mode in protocol.MODES:
            config = copy.deepcopy(self.base_config)
            config['work_dir'] = f'./work_dir/{mode}'
            if mode == 'joint_state':
                config['model_args']['ssm_joint_state_layers'] = [4]
            elif mode == 'dual_scale':
                config['model_args']['ssm_dual_scale_layers'] = [4]
            self.write_config(mode, config)
        for relative in protocol.SOURCE_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# core source\nvalue = 1\n', encoding='utf-8')
        patcher = mock.patch.object(protocol, 'runtime_identity', return_value=copy.deepcopy(DEVICE))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_config(self, mode, config):
        path = self.root / 'config/nturgbd-cross-subject' / f'joint_algorithm_{mode}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config), encoding='utf-8')

    def identity(self, mode='baseline', seed=4321):
        return protocol.current_identity(self.root, mode, seed)

    def baseline_run(self, seed=4321):
        directory = self.root / 'saved_baseline'
        identity = self.identity(seed=seed)
        protocol.write_manifest(directory, identity)
        actual = copy.deepcopy(self.base_config)
        actual['seed'] = seed
        actual['work_dir'] = str(directory)
        actual['num_epoch'] = 20
        actual['save_interval'] = 10
        actual['model_saved_name'] = 'runs'
        actual['lambda_1'] = 0.0001
        (directory / 'config.yaml').write_text(json.dumps(actual), encoding='utf-8')
        return directory, identity, actual

    def test_modes_share_common_recipe_but_not_raw_config_hash(self):
        baseline, joint, dual = (self.identity(mode) for mode in protocol.MODES)
        for candidate in (joint, dual):
            self.assertEqual(candidate['recipe_fingerprint'], baseline['recipe_fingerprint'])
            self.assertEqual(candidate['source_sha256'], baseline['source_sha256'])
            self.assertNotEqual(candidate['config_sha256'], baseline['config_sha256'])
        changed_seed = self.identity('joint_state', seed=99)
        self.assertEqual(changed_seed['seed'], 99)
        self.assertNotEqual(changed_seed['recipe_fingerprint'], joint['recipe_fingerprint'])

    def test_incompatible_published_recipe_is_rejected_before_gpu_or_data_work(self):
        for key, value in (('precision', 'bf16'), ('tf32', True), ('tf32', None),
                           ('optimizer', 'AdamW'), ('ema', True),
                           ('persistent_workers', True)):
            changed = copy.deepcopy(self.base_config)
            changed[key] = value
            self.write_config('baseline', changed)
            with self.subTest(key=key, value=value):
                with mock.patch.object(protocol, 'runtime_identity') as runtime:
                    with mock.patch.object(protocol, 'dataset_fingerprint') as dataset:
                        with self.assertRaisesRegex(ValueError, 'published recipe'):
                            self.identity()
                        runtime.assert_not_called()
                        dataset.assert_not_called()

    def test_recipe_keeps_public_backbone_training_and_data(self):
        original = protocol.canonical_recipe(self.base_config, 4321)
        for section, key, value in ((None, 'base_lr', 0.1), (None, 'precision', 'bf16'),
                                    (None, 'batch_size', 32), (None, 'num_worker', 0),
                                    ('model_args', 'num_point', 20),
                                    ('train_feeder_args', 'window_size', 32)):
            changed = copy.deepcopy(self.base_config)
            (changed if section is None else changed[section])[key] = value
            self.assertNotEqual(protocol.canonical_recipe(changed, 4321), original, (section, key))
        changed = copy.deepcopy(self.base_config)
        changed.update(work_dir='elsewhere', num_epoch=20, start_epoch=10,
                       resume='latest.pt', weights=None, save_interval=1, save_latest=True)
        changed['model_args'].update(ssm_joint_state_layers=[4], ssm_joint_state_weight=0.5)
        self.assertEqual(protocol.canonical_recipe(changed, 4321), original)

    def test_dataset_cache_reuses_hash_and_invalidates_modified_file(self):
        with mock.patch.object(protocol, '_hash_binary', wraps=protocol._hash_binary) as hashing:
            first = self.identity()
            second = self.identity('joint_state')
            self.assertEqual(hashing.call_count, 1)
            self.assertEqual(first['dataset_sha256'], second['dataset_sha256'])
            stat = self.dataset.stat()
            self.dataset.write_bytes(b'X' * stat.st_size)
            os.utime(self.dataset, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            changed = self.identity()
            self.assertEqual(hashing.call_count, 2)
            self.assertNotEqual(changed['dataset_sha256'], first['dataset_sha256'])

    def test_source_hash_is_newline_normalized_and_excludes_docs(self):
        first = self.identity()['source_sha256']
        path = self.root / protocol.SOURCE_FILES[0]
        path.write_bytes(b'# core source\r\nvalue = 1\r\n')
        self.assertEqual(self.identity()['source_sha256'], first)
        (self.root / 'notes.md').write_text('new documentation', encoding='utf-8')
        self.assertEqual(self.identity()['source_sha256'], first)
        path.write_text('# core source\nvalue = 2\n', encoding='utf-8')
        self.assertNotEqual(self.identity()['source_sha256'], first)

    def test_missing_source_or_different_train_test_npz_is_rejected(self):
        (self.root / protocol.SOURCE_FILES[0]).unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing protocol source'):
            self.identity()
        config = copy.deepcopy(self.base_config)
        config['test_feeder_args']['data_path'] = 'other.npz'
        self.write_config('baseline', config)
        with self.assertRaisesRegex(ValueError, 'same NTU NPZ'):
            self.identity()

    def test_receipt_validation_rejects_each_identity_mismatch(self):
        expected = self.identity('joint_state')
        self.assertTrue(protocol.validate_identity(dict(identity=expected, passed=True), expected))
        for key in protocol.IDENTITY_KEYS:
            changed = copy.deepcopy(expected)
            changed[key] = 'different'
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                protocol.validate_identity(changed, expected)
        incomplete = dict(expected)
        del incomplete['source_sha256']
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            protocol.validate_identity(incomplete, expected)

    def test_manifest_roundtrip_does_not_overwrite_different_identity(self):
        directory = self.root / 'run'
        expected = self.identity('joint_state')
        path = protocol.write_manifest(directory, expected)
        self.assertEqual(path.name, 'run_manifest.json')
        original = path.read_bytes()
        self.assertTrue(protocol.validate_identity(protocol.load_manifest(directory), expected))
        self.assertEqual(protocol.write_manifest(directory, expected), path)
        self.assertEqual(path.read_bytes(), original)
        different = self.identity('dual_scale')
        with self.assertRaises(RuntimeError):
            protocol.write_manifest(directory, different)
        self.assertEqual(path.read_bytes(), original)

    def test_baseline_reuse_allows_candidate_mode_and_ten_vs_twenty_budget(self):
        directory, baseline, _ = self.baseline_run()
        # A malformed checkpoint is deliberately never read by identity checks.
        (directory / 'runs-latest.pt').write_bytes(b'not a torch checkpoint')
        result = protocol.validate_baseline_manifest(self.root, directory, 4321,
                                                      self.identity('joint_state'))
        self.assertEqual(result['mode'], 'baseline')
        self.assertEqual(result['source_sha256'], baseline['source_sha256'])

    def test_baseline_reuse_rejects_seed_environment_and_source_changes(self):
        directory, _, _ = self.baseline_run()
        for key, value in (('seed', 99), ('source_sha256', '0' * 64),
                           ('device', dict(DEVICE, name='NVIDIA RTX 3080 Ti'))):
            expected = self.identity('joint_state')
            expected[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                protocol.validate_baseline_manifest(self.root, directory, 4321, expected)

    def test_baseline_actual_config_catches_enabled_modules_and_cli_override(self):
        directory, _, actual = self.baseline_run()
        expected = self.identity('dual_scale')
        for key in ('ctr_layers', 'fr_layers', 'ssm_bidir_layers', 'ssm_motion_layers',
                    'ssm_bc_norm_layers', 'ssm_joint_state_layers', 'ssm_dual_scale_layers'):
            changed = copy.deepcopy(actual)
            changed['model_args'][key] = [4]
            (directory / 'config.yaml').write_text(json.dumps(changed), encoding='utf-8')
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'algorithm changes'):
                protocol.validate_baseline_manifest(self.root, directory, 4321, expected)
        changed = copy.deepcopy(actual)
        changed['base_lr'] = 0.1
        (directory / 'config.yaml').write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'actual training recipe'):
            protocol.validate_baseline_manifest(self.root, directory, 4321, expected)

    def test_old_baseline_without_manifest_is_refused_without_writes(self):
        directory = self.root / 'old_baseline'
        directory.mkdir()
        (directory / 'log.txt').write_text('old results', encoding='utf-8')
        before = set(directory.iterdir())
        with self.assertRaisesRegex(RuntimeError, 'cannot be verified'):
            protocol.validate_baseline_manifest(self.root, directory, 4321, self.identity())
        self.assertEqual(set(directory.iterdir()), before)


class ActualTrainingConfigTest(unittest.TestCase):
    def test_saved_argparse_configuration_matches_nominal_recipe(self):
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('str2bool', 'get_parser')]
        namespace = dict(argparse=argparse)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main.py', 'exec'), namespace)
        nominal = json.loads((ROOT / 'config/nturgbd-cross-subject/joint_algorithm_baseline.yaml')
                             .read_text(encoding='utf-8'))
        parser = namespace['get_parser']()
        self.assertFalse(set(nominal).difference(vars(parser.parse_args([]))))
        parser.set_defaults(**nominal)
        actual = vars(parser.parse_args(['--seed', '2026', '--num-epoch', '20',
                                        '--resume', '/example/runs-latest.pt']))
        expected = protocol.canonical_recipe(nominal, 2026)
        actual_recipe = protocol.canonical_recipe(actual, 2026)
        self.assertEqual({key: actual_recipe.get(key) for key in expected}, expected)


class RuntimeIdentityTest(unittest.TestCase):
    def tearDown(self):
        protocol._runtime_identity_cached.cache_clear()

    def test_subprocess_uses_active_python_and_parses_marker(self):
        protocol._runtime_identity_cached.cache_clear()
        output = 'an unrelated import notice\nSIMBA_RUNTIME_IDENTITY=' + json.dumps(DEVICE) + '\n'
        completed = subprocess.CompletedProcess([], 0, stdout=output, stderr='warning')
        with mock.patch.object(protocol.subprocess, 'run', return_value=completed) as run:
            first = protocol.runtime_identity()
            first['name'] = 'changed by caller'
            self.assertEqual(protocol.runtime_identity(), DEVICE)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertEqual(command[:2], [sys.executable, '-c'])
            self.assertIn('import json, platform, torch', command[2])

    def test_missing_cuda_subprocess_fails_clearly(self):
        protocol._runtime_identity_cached.cache_clear()
        completed = subprocess.CompletedProcess([], 1, stdout='', stderr='CUDA unavailable')
        with mock.patch.object(protocol.subprocess, 'run', return_value=completed):
            with self.assertRaisesRegex(RuntimeError, 'Cannot query CUDA/runtime'):
                protocol.runtime_identity()


if __name__ == '__main__':
    unittest.main()
