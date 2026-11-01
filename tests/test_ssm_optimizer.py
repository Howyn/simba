"""CPU checks for SSM-only optimizer protection; no CUDA/model imports."""

import copy
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from tools.ssm_optimizer import optimizer_parameter_groups


class TinyMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.bidir_logit = nn.Parameter(torch.full((8,), math.log(.1 / .9)))
        self.motion_down = nn.Linear(8, 3, bias=False)
        self.motion_up = nn.Linear(3, 2, bias=False)
        nn.init.zeros_(self.motion_up.weight)
        self.dt_proj = nn.Linear(2, 8)
        self.A_log = nn.Parameter(torch.ones(8, 2))
        self.D = nn.Parameter(torch.ones(8))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 2, bias=False)
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)
        self.l4 = nn.Module()
        self.l4.mamba = nn.Module()
        self.l4.mamba.mamba = TinyMamba()
        # Same leaf names elsewhere must retain the original decay.
        self.motion_down = nn.Linear(8, 3, bias=False)


class SSMOptimizerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4321)
        self.model = TinyModel()

    def optimizer(self, model, protect):
        return torch.optim.SGD(
            optimizer_parameter_groups(model, .0004, protect),
            lr=.025, weight_decay=.0004, momentum=.9, nesterov=True)

    def zero_task_gradients(self, model):
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)

    def test_default_matches_original_iterator_and_sgd_updates(self):
        self.assertEqual([id(p) for p in optimizer_parameter_groups(self.model, .0004)],
                         [id(p) for p in self.model.parameters()])
        original = copy.deepcopy(self.model)
        expected = torch.optim.SGD(original.parameters(), lr=.025, weight_decay=.0004,
                                   momentum=.9, nesterov=True)
        actual = self.optimizer(self.model, False)
        for _ in range(8):
            for number, (a, b) in enumerate(zip(self.model.parameters(), original.parameters())):
                if a.requires_grad:
                    a.grad = torch.full_like(a, .001 * (number + 1))
                    b.grad = a.grad.clone()
            actual.step()
            expected.step()
        for a, b in zip(self.model.parameters(), original.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_groups_are_complete_unique_and_leave_backbone_decay_unchanged(self):
        self.model.shared_alias = self.model.backbone.weight
        groups = optimizer_parameter_groups(self.model, .0004, True)
        by_name = dict(self.model.named_parameters())
        assigned = {id(parameter): group['weight_decay']
                    for group in groups for parameter in group['params']}
        parameters = [parameter for group in groups for parameter in group['params']]
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual(set(assigned), {id(parameter) for parameter in self.model.parameters()})
        for name, parameter in by_name.items():
            expected = 0.0 if name in ('l4.mamba.mamba.bidir_logit',
                                      'l4.mamba.mamba.motion_down.weight',
                                      'l4.mamba.mamba.motion_up.weight') else .0004
            self.assertEqual(assigned[id(parameter)], expected, name)
        self.assertEqual(assigned[id(self.model.frozen)], .0004)
        self.assertEqual([group['group_name'] for group in groups], ['backbone', 'ssm_addons'])

    def test_zero_task_gradient_cannot_decay_protected_gate_or_motion_weights(self):
        original = copy.deepcopy(self.model)
        initial = {name: value.clone() for name, value in self.model.named_parameters()}
        protected_optimizer = self.optimizer(self.model, True)
        original_optimizer = self.optimizer(original, False)
        for _ in range(128):
            self.zero_task_gradients(self.model)
            self.zero_task_gradients(original)
            protected_optimizer.step()
            original_optimizer.step()
        actual = dict(self.model.named_parameters())
        previous = dict(original.named_parameters())
        for name in actual:
            if name in ('l4.mamba.mamba.bidir_logit', 'l4.mamba.mamba.motion_down.weight',
                        'l4.mamba.mamba.motion_up.weight'):
                torch.testing.assert_close(actual[name], initial[name], atol=0, rtol=0)
            else:
                torch.testing.assert_close(actual[name], previous[name], atol=0, rtol=0)
        gate = self.model.l4.mamba.mamba.bidir_logit.sigmoid()
        torch.testing.assert_close(gate, torch.full_like(gate, .1))
        self.assertGreater(original.l4.mamba.mamba.bidir_logit.sigmoid().mean().item(), .1)
        self.assertFalse(torch.equal(previous['l4.mamba.mamba.motion_down.weight'],
                                     initial['l4.mamba.mamba.motion_down.weight']))

    def test_protected_addons_still_learn_from_nonzero_task_gradient(self):
        optimizer = self.optimizer(self.model, True)
        inner = self.model.l4.mamba.mamba
        before = [parameter.detach().clone() for parameter in
                  (inner.bidir_logit, inner.motion_down.weight, inner.motion_up.weight)]
        for parameter in (inner.bidir_logit, inner.motion_down.weight, inner.motion_up.weight):
            parameter.grad = torch.full_like(parameter, .1)
        optimizer.step()
        for parameter, initial in zip((inner.bidir_logit, inner.motion_down.weight,
                                       inner.motion_up.weight), before):
            self.assertFalse(torch.equal(parameter, initial))

    def test_protection_with_no_addon_parameters_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'no bidir_logit'):
            optimizer_parameter_groups(nn.Linear(8, 2), .0004, True)

    def test_bc_norm_only_parameters_are_protected(self):
        inner = self.model.l4.mamba.mamba
        for name in ('bidir_logit', 'motion_down', 'motion_up'):
            delattr(inner, name)
        names = ('bc_B_gain', 'bc_C_gain', 'bc_B_bias', 'bc_C_bias')
        for name in names:
            inner.register_parameter(name, nn.Parameter(torch.full((2,), .1)))
        # An unrelated similarly named parameter is still part of the backbone.
        self.model.bc_B_gain = nn.Parameter(torch.full((2,), .1))
        groups = optimizer_parameter_groups(self.model, .0004, True)
        no_decay = {id(parameter) for group in groups if group['weight_decay'] == 0
                    for parameter in group['params']}
        self.assertEqual(no_decay, {id(getattr(inner, name)) for name in names})
        self.assertNotIn(id(inner.A_log), no_decay)
        self.assertNotIn(id(inner.D), no_decay)
        self.assertNotIn(id(self.model.bc_B_gain), no_decay)
        initial = [getattr(inner, name).detach().clone() for name in names]
        optimizer = torch.optim.SGD(groups, lr=.025, momentum=.9, nesterov=True, weight_decay=.0004)
        for _ in range(16):
            self.zero_task_gradients(self.model)
            optimizer.step()
        for name, expected in zip(names, initial):
            torch.testing.assert_close(getattr(inner, name), expected, atol=0, rtol=0)

    def test_training_entry_point_uses_protected_groups_only_when_enabled(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text(encoding='utf-8'))
        processor = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == 'Processor')
        method = next(node for node in processor.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'load_optimizer')
        namespace = dict(torch=torch, optim=torch.optim)
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'main.py', 'exec'), namespace)
        for protect in (False, True):
            args = SimpleNamespace(optimizer='SGD', ssm_protect_addons=protect, base_lr=.025,
                                   momentum=.9, nesterov=True, weight_decay=.0004,
                                   warm_up_epoch=5, resume=None)
            subject = SimpleNamespace(arg=args, model=self.model, print_log=lambda message: None)
            namespace['load_optimizer'](subject)
            groups = subject.optimizer.param_groups
            self.assertEqual(len(groups), 2 if protect else 1)
            self.assertEqual(groups[-1]['weight_decay'], 0. if protect else .0004)
            self.assertEqual(groups[0]['weight_decay'], .0004)


if __name__ == '__main__':
    unittest.main()
