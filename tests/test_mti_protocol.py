import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import (
    check_resume_signature, stable_window, training_signature, validate_screening, write_run_results,
)


class MTIProtocolTests(unittest.TestCase):
    def configs(self):
        paths = [ROOT / f'config/nturgbd-cross-subject/joint_mti_{name}15.yaml'
                 for name in ('baseline', 'enabled')]
        return [yaml.safe_load(path.read_text(encoding='utf-8')) for path in paths]

    def test_paired_configs_only_differ_by_work_directory_and_switch(self):
        baseline, improved = self.configs()
        self.assertFalse(baseline['model_args']['use_mti'])
        self.assertTrue(improved['model_args']['use_mti'])
        improved['work_dir'] = baseline['work_dir']
        improved['model_args']['use_mti'] = False
        self.assertEqual(baseline, improved)
        args = argparse.Namespace(**baseline, weights=None, resume=None, start_epoch=0)
        validate_screening(args)
        args.num_epoch = 16
        with self.assertRaises(ValueError):
            validate_screening(args)

    def test_yaml_keys_match_cli_and_fixed_lr_never_changes(self):
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('get_parser', 'str2bool')]
        namespace = dict(argparse=argparse)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'main.py', 'exec'), namespace)
        parser = namespace['get_parser']()
        config = self.configs()[0]
        self.assertTrue(set(config).issubset(vars(parser.parse_args([]))))
        parser.set_defaults(**config)
        args = parser.parse_args([])
        validate_screening(args)
        processor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Processor')
        function = next(node for node in processor.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'adjust_learning_rate')
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'main.py', 'exec'), namespace)
        fake = argparse.Namespace(arg=args, optimizer=argparse.Namespace(param_groups=[{'lr': 99.0}]))
        for epoch in range(15):
            self.assertEqual(namespace['adjust_learning_rate'](fake, epoch), 0.005)
            self.assertEqual(fake.optimizer.param_groups[0]['lr'], 0.005)

    def test_stable_unstable_and_incomplete_windows(self):
        rows = [{'epoch': e, 'top1_percent': 85 + (-0.2 if e % 2 else 0.2)} for e in range(10, 16)]
        self.assertEqual(stable_window(rows)['status'], 'stable')
        self.assertAlmostEqual(stable_window(rows)['final_mean_top1_percent'], 85)
        rows[-1]['top1_percent'] = 86
        result = stable_window(rows)
        self.assertEqual(result['status'], 'unstable')
        self.assertIsNone(result['final_mean_top1_percent'])
        self.assertEqual(stable_window(rows[:-1])['status'], 'incomplete')

    def test_resume_rejects_learning_rate_seed_precision_and_variant_changes(self):
        config = training_signature(self.configs()[0])
        check_resume_signature(config, copy.deepcopy(config))
        for key, value in [('base_lr', 0.001), ('seed', 123), ('precision', 'fp16')]:
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                check_resume_signature(config, changed)
        changed = copy.deepcopy(config)
        changed['model_args']['use_mti'] = True
        with self.assertRaises(ValueError):
            check_resume_signature(config, changed)

    def test_checkpoint_roundtrip_preserves_amp_optimizer_rng_and_epoch_history(self):
        import torch
        import numpy as np
        from tools.training_rng import capture_rng_state, restore_rng_state
        # Extract the real checkpoint function without importing CUDA-only main.py dependencies.
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'epoch_saving')
        from collections import OrderedDict
        import os
        namespace = dict(torch=torch, OrderedDict=OrderedDict, os=os, tempfile=tempfile)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'main.py', 'exec'), namespace)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        scaler = torch.amp.GradScaler('cpu', init_scale=128)
        loss = model(torch.randn(2, 3)).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        config = training_signature(self.configs()[0])
        rows = [{'epoch': e, 'top1_percent': 85.0} for e in range(1, 16)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-latest.pt'
            namespace['epoch_saving'](14, model, optimizer, str(path), global_step=30,
                                      scaler=scaler, epoch_records=rows, run_config=config)
            expected_torch = torch.rand(4)
            expected_numpy = np.random.rand(4)
            checkpoint = torch.load(path, weights_only=True)
            restore_rng_state(checkpoint['rng_state'])
            torch.testing.assert_close(torch.rand(4), expected_torch)
            np.testing.assert_equal(np.random.rand(4), expected_numpy)
            restored_scaler = torch.amp.GradScaler('cpu')
            restored_scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.assertEqual(restored_scaler.state_dict(), scaler.state_dict())
            self.assertTrue(checkpoint['optimizer_state_dict']['state'])
            check_resume_signature(checkpoint['run_config'], config)
            summary = write_run_results(directory, checkpoint['epoch_records'], config, completed=True)
            self.assertEqual(summary['stability']['status'], 'stable')
            self.assertEqual(json.loads((Path(directory) / 'results.json').read_text())['seed'], 4321)
            self.assertEqual(len((Path(directory) / 'metrics.csv').read_text().splitlines()), 16)


if __name__ == '__main__':
    unittest.main()
