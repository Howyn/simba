"""Numerical references on CPU; exact historical parity/full AMP backward on CUDA."""
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'requires torch')
class AdapterTests(unittest.TestCase):
    def test_temporal_interpolation_outputs_and_gradients(self):
        from model.combination1 import DeformableTemporalConv
        torch.manual_seed(51)
        module = DeformableTemporalConv(3, stride=2).eval()
        with torch.no_grad():
            module.tr.copy_(torch.tensor([-5.2, -0.7, 1.3, 5.8]))
        x = torch.randn(2, 3, 9, 4, requires_grad=True)
        output = module(x)
        samples = []
        for offset in module.tr:
            times = []
            for center in range(0, 9, 2):
                position = center + offset
                lo = int(position.detach().floor())
                ratio = position - lo
                times.append(x[:, :, max(0, min(8, lo)), :] * (1 - ratio)
                             + x[:, :, max(0, min(8, lo + 1)), :] * ratio)
            samples.append(torch.stack(times, dim=2))
        reference = module.conv_out(torch.stack(samples, dim=2)).squeeze(2)
        torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-6)
        probe = torch.randn_like(output)
        targets = (x, *module.parameters())
        a = torch.autograd.grad((output * probe).sum(), targets)
        b = torch.autograd.grad((reference * probe).sum(), targets)
        for actual, expected in zip(a, b):
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        self.assertGreater(a[1].abs().sum().item(), 0)

    def test_hypergraph_matches_upstream_equations_and_gradients(self):
        from model.combination1 import HyperGraphSpatial, virtual_graph
        torch.manual_seed(21)
        adjacency = torch.eye(5)[None].numpy()
        model = HyperGraphSpatial(32, adjacency, virtual_nodes=3, heads=4, neighbours=5).eval()
        prior = virtual_graph(adjacency, 3, 4)
        self.assertTrue(torch.all(prior[:, :5, 5:] == 1))
        self.assertEqual(prior[0, 5, 6].item(), 0)
        x = torch.randn(2, 32, 7, 5, requires_grad=True)
        actual = model(x)
        expanded = torch.cat((x, model.hyper_joint.T[None, :, None, :].repeat(2, 1, 7, 1)), dim=-1)
        points = model.to_V(expanded.mean(2)).reshape(2, 4, 2, 8).transpose(-1, -2).contiguous()
        distance = torch.cdist(points, points, compute_mode='donot_use_mm_for_euclid_dist')
        nearest, indices = distance.topk(5, largest=False)
        h = torch.zeros_like(distance).scatter(-1, indices, (-nearest).softmax(-1))
        w = torch.diag_embed(model.to_W(expanded.mean(2)))
        hw = h @ w
        dynamic = (hw / (hw.abs().sum(3, keepdim=True) + 1e-8)) @ (
            w / (h.abs().sum(2, keepdim=True) + 1e-8)) @ h.transpose(3, 2)
        fixed = model.PA * model.edge_importance
        fixed = fixed / (fixed.abs().sum(2, keepdim=True) + 1e-8)
        graph = fixed + model.alpha.relu() * dynamic
        projected = model.conv_d(expanded).view(2, 4, 8, 7, 8)
        reference = torch.einsum('nkuv,nkctv->nkctu', graph, projected).reshape(2, 32, 7, 8)[..., :5]
        reference = torch.relu(model.bn(reference) + x)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        probe = torch.randn_like(actual)
        targets = (x, *model.parameters())
        a = torch.autograd.grad((actual * probe).sum(), targets)
        b = torch.autograd.grad((reference * probe).sum(), targets)
        for value, expected in zip(a, b):
            self.assertTrue(torch.isfinite(value).all().item())
            torch.testing.assert_close(value, expected, rtol=3e-4, atol=3e-5)

    def test_actual_stage_lengths_and_bf16(self):
        from model.combination1 import MultiScaleDeformableTemporal, HyperGraphSpatial
        for length in (64, 32):
            module = MultiScaleDeformableTemporal(216, stride=2).eval()
            x = torch.randn(1, 216, length, 25)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                output = module(x)
            self.assertEqual(output.shape, (1, 216, length // 2, 25))
            self.assertTrue(torch.isfinite(output).all().item())
        spatial = HyperGraphSpatial(216, torch.eye(25)[None].numpy()).eval()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = spatial(torch.randn(1, 216, 16, 25))
        self.assertEqual(result.shape, (1, 216, 16, 25))
        self.assertTrue(torch.isfinite(result).all().item())

    def test_bf16_backward_through_both_adapters(self):
        from model.combination1 import MultiScaleDeformableTemporal, HyperGraphSpatial
        spatial = HyperGraphSpatial(216, torch.eye(25)[None].numpy())
        temporal = MultiScaleDeformableTemporal(216, stride=2)
        x = torch.randn(2, 216, 32, 25, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y = temporal(spatial(x))
            loss = (y.float() * torch.randn_like(y).float()).mean()
        loss.backward()
        for parameter in (spatial.hyper_joint, spatial.to_W[0].weight, temporal.branches[0][2].tr):
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires AutoDL CUDA/scan environment')
class FullModelTests(unittest.TestCase):
    def test_historical_baseline_parity_common_weights_and_combined_backward(self):
        import yaml
        from model.Hyperformer import Model
        from tools.run_combination1 import BASELINE_REVISION
        if not torch.cuda.is_bf16_supported():
            self.fail('BF16 is required by the approved protocol')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        config = yaml.safe_load((ROOT / 'config/nturgbd-cross-subject/joint_bimamba_baseline10.yaml').read_text())
        # Load the exact previous model definition, not a reimplementation of it.
        source = subprocess.check_output(['git', 'show', BASELINE_REVISION + ':model/Hyperformer.py'], cwd=ROOT, text=True)
        old = types.ModuleType('historical_simba')
        exec(compile(source, 'historical_simba.py', 'exec'), old.__dict__)
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        historical = old.Model(**config['model_args']).cuda().eval()
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        baseline = Model(**config['model_args']).cuda().eval()
        for key, value in historical.state_dict().items():
            torch.testing.assert_close(baseline.state_dict()[key], value, rtol=0, atol=0)
        baseline_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        x = torch.randn(2, 3, 64, 25, 2, device='cuda')
        y = torch.nn.functional.one_hot(torch.tensor([0, 1], device='cuda'), 60)
        with torch.no_grad():
            torch.testing.assert_close(historical(x, y)[0], baseline(x, y)[0], rtol=0, atol=0)
        del historical
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        combined = Model(**config['model_args'], use_hypergcn=True, use_detgc=True).cuda()
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
        replaced = ('l4.shift_gcn.', 'l7.shift_gcn.', 'l10.shift_gcn.', 'l5.tcn1.', 'l8.tcn1.')
        for key, value in baseline.state_dict().items():
            if not key.startswith(replaced):
                torch.testing.assert_close(combined.state_dict()[key], value, rtol=0, atol=0)
        del baseline
        torch.cuda.empty_cache()
        optimizer = torch.optim.SGD(combined.parameters(), lr=0.0125, momentum=0.9, nesterov=True)
        scaler = torch.amp.GradScaler('cuda', init_scale=1)
        x = torch.randn(24, 3, 64, 25, 2, device='cuda')
        labels = torch.arange(24, device='cuda') % 60
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, _ = combined(x, torch.nn.functional.one_hot(labels, 60))
            loss = torch.nn.functional.cross_entropy(output, labels)
        self.assertTrue(torch.isfinite(loss).item())
        scaler.scale(loss).backward()
        for parameter in (combined.l4.shift_gcn.to_V.weight, combined.l10.shift_gcn.hyper_joint,
                          combined.l5.tcn1.branches[0][2].tr, combined.l8.tcn1.branches[1][2].tr):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        scaler.step(optimizer)
        scaler.update()
        print(f'COMBINATION1 PREFLIGHT OK: BF16 batch24 backward; peak={torch.cuda.max_memory_allocated()/1024**2:.0f} MiB')


if __name__ == '__main__':
    unittest.main()
