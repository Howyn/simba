"""CPU checks of actual Simba wrappers and model construction.

Only native scan/RMSNorm kernels are replaced with CPU equations. The real
Model, TCN_ViT_unit, Shift_gcn and Shift_tcn classes are compiled from their
source AST with explicit 'cuda' placements changed to 'cpu'. This verifies
the complete constructor and tensor wiring, not native CUDA execution.
"""

import ast
from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat


ROOT = Path(__file__).resolve().parents[1]


def reference_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False, return_last_state=False):
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        step = delta[:, :, t, None]
        state = (step * A[None]).exp() * state + step * B[:, None, :, t] * u[:, :, t, None]
        outputs.append((state * C[:, None, :, t]).sum(dim=-1))
    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + D[None, :, None] * u
    if z is not None:
        output = output * F.silu(z)
    return (output, state) if return_last_state else output


class CpuRMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)


def package(name):
    result = types.ModuleType(name)
    result.__path__ = []
    return result


def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CpuPlacement(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'cuda':
            return ast.copy_location(ast.Constant('cpu'), node)
        return node


def extract_definitions(relative, names, namespace):
    source = ast.parse((ROOT / relative).read_text(encoding='utf-8-sig'))
    wanted = set(names)
    definitions = [node for node in source.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise AssertionError('missing real source definitions: ' + relative)
    tree = ast.fix_missing_locations(CpuPlacement().visit(ast.Module(body=definitions, type_ignores=[])))
    exec(compile(tree, str(ROOT / relative), 'exec'), namespace)


@contextmanager
def isolated_sources():
    # einops lazily imports torch._dynamo and registers native libraries on
    # first use. Import these outside patch.dict so its rollback cannot evict
    # their Python modules while leaving native registrations alive.
    repeat(torch.zeros(1), 'd -> d n', n=1)
    scan = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    scan.selective_scan_fn = reference_scan
    scan.mamba_inner_fn = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('native fused path'))
    causal = types.ModuleType('causal_conv1d')
    causal.causal_conv1d_fn = causal.causal_conv1d_update = None
    update = types.ModuleType('mamba_ssm.ops.triton.selective_state_update')
    update.selective_state_update = None
    norms = types.ModuleType('mamba_ssm.ops.triton.layernorm')
    norms.RMSNorm = CpuRMSNorm
    norms.layer_norm_fn = norms.rms_norm_fn = None
    modules = {
        'mamba_ssm': package('mamba_ssm'),
        'mamba_ssm.modules': package('mamba_ssm.modules'),
        'mamba_ssm.ops': package('mamba_ssm.ops'),
        'mamba_ssm.ops.triton': package('mamba_ssm.ops.triton'),
        'mamba_ssm.ops.selective_scan_interface': scan,
        'mamba_ssm.ops.triton.selective_state_update': update,
        'mamba_ssm.ops.triton.layernorm': norms,
        'causal_conv1d': causal,
    }
    with mock.patch.dict(sys.modules, modules):
        simple = load_source('mamba_ssm.modules.mamba_simple', 'mamba_ssm/modules/mamba_simple.py')
        load_source('mamba_ssm.modules.mamba_joint_state', 'mamba_ssm/modules/mamba_joint_state.py')
        load_source('mamba_ssm.modules.mamba_dual_scale', 'mamba_ssm/modules/mamba_dual_scale.py')
        namespace = dict(__name__='algorithm_integration_subject', torch=torch, nn=nn,
                         F=F, np=np, math=math, Mamba=simple.Mamba, RMSNorm=CpuRMSNorm)
        extract_definitions('mamba_blocks.py', ['MambaBlock'], namespace)
        extract_definitions('layers.py', ['conv_init', 'bn_init', 'Shift_gcn'], namespace)
        extract_definitions('shiftgcn_model/shift_gcn.py', ['naive_temporal_shift', 'Shift_tcn'], namespace)
        extract_definitions('model/Hyperformer.py',
                            ['conv_init', 'bn_init', 'unit_tcn', 'TCN_ViT_unit', 'Model'], namespace)

        class Graph:
            def __init__(self, **kwargs):
                self.A = np.stack([np.eye(25, dtype=np.float32)] * 3)

        namespace['import_class'] = lambda name: Graph
        optimizer = load_source('algorithm_integration_optimizer', 'tools/ssm_optimizer.py')
        namespace['optimizer_groups'] = optimizer.algorithm_parameter_groups
        yield types.SimpleNamespace(**namespace)


class AlgorithmIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def configuration(self, mode):
        path = ROOT / 'config/nturgbd-cross-subject' / ('joint_algorithm_' + mode + '.yaml')
        return json.loads(path.read_text(encoding='utf-8-sig'))['model_args']

    def build(self, subject, mode):
        model = subject.Model(**self.configuration(mode))
        # Use the real unfused Mamba forward with the reference scan on CPU.
        for module in model.modules():
            if isinstance(module, subject.Mamba):
                module.use_fast_path = False
        return model

    def test_full_model_common_weights_and_rng_are_identical(self):
        with isolated_sources() as subject:
            torch.manual_seed(4321)
            baseline = self.build(subject, 'baseline')
            expected_rng = torch.get_rng_state().clone()
            expected = baseline.state_dict()
            for mode in ('joint_state', 'dual_scale'):
                with self.subTest(mode=mode):
                    torch.manual_seed(4321)
                    candidate = self.build(subject, mode)
                    self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                    actual = candidate.state_dict()
                    for name, value in expected.items():
                        self.assertIn(name, actual)
                        self.assertTrue(torch.equal(value, actual[name]), name)
                    self.assertEqual(candidate.l4.mamba.mamba.d_model, 625)
                    if mode == 'joint_state':
                        self.assertEqual(candidate.l4.mamba.joint_state.num_joints, 25)
                        self.assertEqual(candidate.l4.mamba.joint_state.d_model, 25)
                    del candidate

    def test_disabled_candidate_restores_full_baseline_logits(self):
        with isolated_sources() as subject:
            torch.manual_seed(4321)
            baseline = self.build(subject, 'baseline').eval()
            x = torch.randn(1, 3, 8, 25, 2)
            y = torch.tensor([7])
            with torch.no_grad():
                expected, passthrough = baseline(x, y)
            self.assertIs(passthrough, y)
            self.assertEqual(tuple(expected.shape), (1, 60))
            for mode in ('joint_state', 'dual_scale'):
                with self.subTest(mode=mode):
                    torch.manual_seed(4321)
                    candidate = self.build(subject, mode).eval()
                    if mode == 'joint_state':
                        candidate.l4.mamba.joint_state.state_disabled = True
                    else:
                        candidate.l4.mamba.mamba.dual_scale_enabled = False
                    with torch.no_grad():
                        actual, _ = candidate(x, y)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    del candidate

    def test_joint_wrapper_is_global_plus_fixed_state_branch_then_norm(self):
        with isolated_sources() as subject:
            block = subject.MambaBlock(625, joint_state=True, num_joints=25).eval()
            block.mamba.use_fast_path = False
            x = torch.randn(2, 5, 625)
            with torch.no_grad():
                expected = block.norm(block.mamba(x) + 0.1 * block.joint_state(x) + x)
                torch.testing.assert_close(block(x), expected, rtol=0, atol=0)
                block.joint_state_enabled = False
                disabled = block.norm(block.mamba(x) + x)
                torch.testing.assert_close(block(x), disabled, rtol=0, atol=0)
            self.assertEqual(tuple(expected.shape), tuple(x.shape))
            self.assertNotIn('joint_state.D', block.state_dict())

    def test_full_model_optimizer_protects_only_dual_gate(self):
        with isolated_sources() as subject:
            for mode in ('baseline', 'joint_state', 'dual_scale'):
                with self.subTest(mode=mode):
                    model = self.build(subject, mode)
                    groups = list(subject.optimizer_groups(model, 0.0005))
                    if mode != 'dual_scale':
                        self.assertEqual([id(p) for p in groups], [id(p) for p in model.parameters()])
                    else:
                        self.assertEqual(len(groups), 2)
                        self.assertEqual(groups[0]['weight_decay'], 0.0005)
                        self.assertEqual(groups[1]['weight_decay'], 0)
                        gates = [p for name, p in model.named_parameters()
                                 if name.endswith('.mamba.mamba.dual_scale_logit')]
                        self.assertEqual(len(gates), 1)
                        self.assertEqual([id(p) for p in groups[1]['params']], [id(p) for p in gates])
                        covered = [id(p) for group in groups for p in group['params']]
                        self.assertEqual(len(covered), len(set(covered)))
                        self.assertEqual(set(covered), {id(p) for p in model.parameters()})
                    del model

    def test_new_candidates_reject_mixing_with_old_modules(self):
        with isolated_sources() as subject:
            for kwargs in (dict(joint_state=True, dual_scale=True),
                           dict(joint_state=True, bc_norm=True),
                           dict(dual_scale=True, motion_delta=True)):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    subject.MambaBlock(625, **kwargs)
            args = self.configuration('joint_state')
            args['ctr_layers'] = [2]
            with self.assertRaises(ValueError):
                subject.Model(**args)


if __name__ == '__main__':
    unittest.main()
