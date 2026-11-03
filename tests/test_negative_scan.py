"""Run on AutoDL after building negative_selective_scan_cuda; failures block training."""
import unittest
import torch
from mamba_ssm.ops.negative_scan_interface import selective_scan_fn
from mamba_ssm.ops.negative_scan_reference import negative_scan_ref


class NegativeScanTests(unittest.TestCase):
    def test_forward_and_gradients(self):
        if not torch.cuda.is_available():
            self.fail('CUDA is required for this verification')
        for length in (1, 17, 64, 2049):
            for with_gate in (False, True):
                with self.subTest(length=length, with_gate=with_gate):
                    torch.manual_seed(23)
                    device = 'cuda'
                    # delta after softplus spans negative and positive transition regimes.
                    shapes = [(2, 4, length), (2, 4, length), (4, 3),
                              (2, 3, length), (2, 3, length), (4,),
                              (2, 4, length), (4,)]
                    values = [torch.randn(*s, device=device) * 0.2 for s in shapes]
                    values[1] = torch.linspace(-0.5, 1, length, device=device).view(1, 1, -1).expand(2, 4, -1).contiguous()
                    values[2] = -torch.tensor([[0.2, 1., 4.]] * 4, device=device)
                    values[7].fill_(-0.2)
                    originals = [x.detach().clone().requires_grad_() for x in values]
                    refs = [x.detach().clone().requires_grad_() for x in values]
                    def apply(fn, v):
                        return fn(*v[:5], D=v[5], z=v[6] if with_gate else None,
                                  delta_bias=v[7], delta_softplus=True)
                    actual = apply(selective_scan_fn, originals)
                    expected = apply(negative_scan_ref, refs)
                    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
                    grad = torch.randn_like(actual)
                    (actual * grad).sum().backward()
                    (expected * grad).sum().backward()
                    for index in range(len(values)):
                        if index == 6 and not with_gate:
                            continue
                        torch.testing.assert_close(originals[index].grad, refs[index].grad,
                                                   rtol=1e-2, atol=4e-3,
                                                   msg=lambda msg: f'input {index}: {msg}')
                    delta = torch.nn.functional.softplus(values[1] + values[7][None, :, None])
                    transition = 2 * torch.exp(delta[..., None] * values[2][None, :, None, :]) - 1
                    self.assertTrue(torch.any(transition < 0).item())
                    self.assertTrue(torch.any(transition > 0).item())

    def test_fused_mamba_matches_unfused(self):
        from mamba_ssm.modules.mamba_simple import Mamba
        torch.manual_seed(7)
        fast = Mamba(d_model=16, d_state=4, scan_mode='negative').cuda()
        slow = Mamba(d_model=16, d_state=4, scan_mode='negative', use_fast_path=False).cuda()
        slow.load_state_dict(fast.state_dict())
        x = torch.randn(2, 19, 16, device='cuda', requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = fast(x), slow(y)
        torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-3)
        grad = torch.randn_like(a)
        (a * grad).sum().backward(); (b * grad).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-2, atol=4e-3)
        for (na, pa), (nb, pb) in zip(fast.named_parameters(), slow.named_parameters()):
            self.assertEqual(na, nb)
            torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-2, atol=4e-3,
                                       msg=lambda msg: f'{na}: {msg}')

if __name__ == '__main__':
    unittest.main()
