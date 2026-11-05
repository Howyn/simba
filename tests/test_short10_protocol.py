import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import (
    SIGNATURE_KEYS, check_resume_signature, planned_learning_rate,
    stable_window, training_signature, validate_screening, write_run_results,
)
from tools.run_bimamba_comparison import compare


class Short10ProtocolTests(unittest.TestCase):
    def args(self):
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('get_parser', 'str2bool')]
        ns = dict(argparse=argparse)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'main.py', 'exec'), ns)
        args = ns['get_parser']().parse_args([])
        values = dict(screening=True, screening_protocol='short10', fixed_lr=False,
                      base_lr=0.025, warm_up_epoch=2, step=[6], lr_decay_rate=0.1,
                      lr_type='multistep', num_epoch=10, optimizer='SGD', batch_size=24,
                      test_batch_size=64, precision='bf16', tf32=True, num_worker=8,
                      seed=4321, device=[0], ema=False, save_latest=True, strict_rng_resume=True, persistent_workers=False,
                      model_args=dict(use_mti=False, use_bimamba=False))
        for name, value in values.items():
            setattr(args, name, value)
        return args

    def test_actual_lr_function_and_resume_boundaries(self):
        args = self.args()
        validate_screening(args)
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Processor')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'adjust_learning_rate')
        ns = dict(planned_learning_rate=planned_learning_rate)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'main.py', 'exec'), ns)
        fake = argparse.Namespace(arg=args, optimizer=argparse.Namespace(param_groups=[{'lr': 99}]))
        expected = [0.0125] + [0.025] * 5 + [0.0025] * 4
        for epoch, value in enumerate(expected):
            self.assertAlmostEqual(ns['adjust_learning_rate'](fake, epoch), value)
            self.assertAlmostEqual(fake.optimizer.param_groups[0]['lr'], value)
        for epoch in (1, 5, 6, 9):  # Restored epoch drives the same schedule, not a new warmup.
            self.assertAlmostEqual(ns['adjust_learning_rate'](fake, epoch), expected[epoch])
        args.batch_size = 16
        with self.assertRaises(ValueError):
            validate_screening(args)

    def test_short_window_and_legacy_checkpoint_compatibility(self):
        args = self.args()
        signature = training_signature(args)
        self.assertEqual(signature['screening_protocol'], 'short10')
        changed = copy.deepcopy(signature)
        changed['screening_protocol'] = 'fixed15'
        with self.assertRaises(ValueError):
            check_resume_signature(signature, changed)
        legacy = {key: None for key in SIGNATURE_KEYS}
        check_resume_signature(legacy, training_signature(legacy))
        rows = [dict(epoch=e, top1_percent=80 + (e-9)*0.1) for e in (8, 9, 10)]
        self.assertEqual(stable_window(rows, 'short10')['status'], 'stable')
        self.assertEqual(stable_window(rows)['status'], 'incomplete')
        rows[-1]['top1_percent'] = 82
        self.assertIsNone(stable_window(rows, 'short10')['final_mean_top1_percent'])

    def test_paired_configs_and_spatial_forward_are_unchanged(self):
        a, b = [(ROOT / f'config/nturgbd-cross-subject/joint_bimamba_{name}10.yaml').read_text()
                for name in ('baseline', 'enabled')]
        self.assertEqual(a, b.replace('/bimamba_seed', '/baseline_seed').replace('use_bimamba: true', 'use_bimamba: false'))
        for filename, classes in [('model/Hyperformer.py', ('Model', 'TCN_ViT_unit')),
                                  ('mamba_blocks.py', ('MambaBlock',))]:
            previous = subprocess.check_output(['git', 'show', '3f0d117:' + filename], cwd=ROOT, text=True)
            current = (ROOT / filename).read_text(encoding='utf-8')
            def forwards(source):
                tree = ast.parse(source)
                return {cls.name: ast.dump(next(fn for fn in cls.body if isinstance(fn, ast.FunctionDef) and fn.name == 'forward'))
                        for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name in classes}
            self.assertEqual(forwards(previous), forwards(current))

    def test_comparison_does_not_invent_an_unstable_final_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in ('baseline', 'bimamba'):
                args = self.args()
                args.model_args['use_bimamba'] = variant == 'bimamba'
                rows = [dict(epoch=e, top1_percent=80 + (0.5 if variant == 'bimamba' else 0)) for e in range(1, 11)]
                if variant == 'bimamba':
                    rows[-1]['top1_percent'] = 82
                write_run_results(root / f'{variant}_seed4321', rows, training_signature(args), completed=True)
            compare(root, 4321)
            result = json.loads((root / 'comparison.json').read_text())
            self.assertEqual(result['verdict'], 'inconclusive_unstable')
            self.assertIsNone(result['delta_percentage_points'])


if __name__ == '__main__':
    unittest.main()
