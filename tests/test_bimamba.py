"""Actual CUDA checks, also run automatically before the paired AutoDL job."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'requires the AutoDL CUDA environment')
class BiMambaTests(unittest.TestCase):
    def test_a_common_weights_and_rng_are_preserved(self):
        from mamba_blocks import MambaBlock
        torch.manual_seed(4321)
        baseline = MambaBlock(625)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(4321)
        improved = MambaBlock(625, use_bimamba=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(improved.state_dict()[key], value, rtol=0, atol=0)
        extra = sum(p.numel() for p in improved.parameters()) - sum(p.numel() for p in baseline.parameters())
        self.assertEqual(extra, 168750)
        with self.assertRaises(ValueError):
            MambaBlock(625, use_mti=True, use_bimamba=True)

    def test_b_fused_outputs_gradients_and_time_alignment(self):
        from mamba_ssm.modules.bimamba import BiMambaV2
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(17)
        model = BiMambaV2(32, d_state=4, d_conv=4).cuda()
        x = torch.randn(2, 9, 32, device='cuda', requires_grad=True)
        probe = torch.randn_like(x)
        fused = model(x)
        reference = model.forward_reference(x)
        torch.testing.assert_close(fused, reference, rtol=2e-3, atol=2e-4)
        targets = (x, *model.parameters())
        fused_grad = torch.autograd.grad((fused * probe).sum(), targets)
        reference_grad = torch.autograd.grad((reference * probe).sum(), targets)
        for a, b in zip(fused_grad, reference_grad):
            torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-4)
        # Equal directional weights must give reversal equivariance. This catches
        # flipping channels instead of time, or forgetting to realign the output.
        with torch.no_grad():
            for forward, backward in [(model.conv1d, model.conv1d_b),
                                       (model.x_proj, model.x_proj_b), (model.dt_proj, model.dt_proj_b)]:
                backward.load_state_dict(forward.state_dict())
            model.A_b_log.copy_(model.A_log)
            model.D_b.copy_(model.D)
            torch.testing.assert_close(model(x), model(x.flip(1)).flip(1), rtol=2e-3, atol=2e-4)

    def test_c_full_model_bf16_batch24_backward(self):
        import yaml
        from model.Hyperformer import Model
        if not torch.cuda.is_bf16_supported():
            self.fail('the agreed protocol requires BF16 support')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        config = yaml.safe_load((ROOT / 'config/nturgbd-cross-subject/joint_bimamba_baseline10.yaml').read_text())
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        baseline = Model(**config['model_args']).cuda()
        torch.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        improved = Model(**dict(config['model_args'], use_bimamba=True)).cuda()
        state = improved.state_dict()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        extra = sum(p.numel() for p in improved.parameters()) - sum(p.numel() for p in baseline.parameters())
        self.assertEqual(extra, 1687500)
        del baseline, state
        torch.cuda.empty_cache()
        optimizer = torch.optim.SGD(improved.parameters(), lr=0.0125, momentum=0.9, nesterov=True)
        scaler = torch.amp.GradScaler('cuda', init_scale=1)
        x = torch.randn(24, 3, 64, 25, 2, device='cuda')
        labels = torch.arange(24, device='cuda') % 60
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output, _ = improved(x, torch.nn.functional.one_hot(labels, 60))
            loss = torch.nn.functional.cross_entropy(output, labels)
        self.assertEqual(output.shape, (24, 60))
        self.assertTrue(torch.isfinite(loss).item())
        scaler.scale(loss).backward()
        for layer in (improved.l1, improved.l5, improved.l10):
            core = layer.mamba.mamba
            for parameter in (core.A_b_log, core.conv1d_b.weight, core.dt_proj_b.weight):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        scaler.step(optimizer)
        scaler.update()
        print('BIMAMBA PREFLIGHT: batch=24 BF16+GradScaler+TF32 backward OK; '
              f'peak={torch.cuda.max_memory_allocated()/1024**2:.0f} MiB; extra_params={extra}', flush=True)


if __name__ == '__main__':
    unittest.main()
