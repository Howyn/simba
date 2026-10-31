"""Exercise checkpoint recovery without importing the CUDA training program."""

import ast
from collections import OrderedDict
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
definition = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'epoch_saving')
namespace = dict(torch=torch, OrderedDict=OrderedDict, os=os, tempfile=tempfile)
exec(compile(ast.Module(body=[definition], type_ignores=[]), 'main.py', 'exec'), namespace)
save_epoch = namespace['epoch_saving']


class TrainingCheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4321)
        self.model = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.025, momentum=0.9)
        self.inputs = torch.randn(3, 4)

    def step(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model(self.inputs).square().mean().backward()
        optimizer.step()

    def test_recovery_retains_optimizer_and_best_metrics(self):
        self.step(self.model, self.optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-latest.pt'
            save_epoch(4, self.model, self.optimizer, str(path), global_step=12600,
                       best_accuracy=0.68, best_epoch=4)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 4)
            self.assertEqual(checkpoint['global_step'], 12600)
            self.assertEqual(checkpoint['best_accuracy'], 0.68)
            self.assertEqual(checkpoint['best_epoch'], 4)
            restored = torch.nn.Linear(4, 2)
            restored.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.SGD(restored.parameters(), lr=1.0, momentum=0.9)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step(self.model, self.optimizer)
            self.step(restored, optimizer)
            for expected, actual in zip(self.model.parameters(), restored.parameters()):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_interrupted_save_retains_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-latest.pt'
            save_epoch(0, self.model, self.optimizer, str(path))
            original = path.read_bytes()
            with mock.patch.object(torch, 'save', side_effect=OSError('disk write interrupted')):
                with self.assertRaises(OSError):
                    save_epoch(1, self.model, self.optimizer, str(path))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_successful_save_replaces_rolling_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs-latest.pt'
            save_epoch(0, self.model, self.optimizer, str(path))
            self.step(self.model, self.optimizer)
            save_epoch(1, self.model, self.optimizer, str(path), best_accuracy=0.5, best_epoch=2)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['best_epoch'], 2)
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
