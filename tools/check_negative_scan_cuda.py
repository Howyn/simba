"""Verify the negative-eigenvalue selective scan against a plain PyTorch reference on CUDA.

Run this before any training. It uses synthetic tensors only: no NTU data, no training
files, no checkpoints.

    python tools/check_negative_scan_cuda.py
    python tools/check_negative_scan_cuda.py --precision bf16

Checks, in order:
  1. the loaded extension advertises the negative branch (a stale build stops here);
  2. scan level, both modes: fused kernel vs reference, forward and every gradient,
     for lengths covering padding, a single chunk and several chunks;
  3. reference gradients against finite differences in float64;
  4. Mamba module level, both modes: fused vs reference output and parameter gradients,
     identical initialisation, and that "negative" really changes the output;
  5. the full Simba model with scan_mode='negative' runs forward/backward on a random
     skeleton batch, so the l1-l10 wiring (64/32/16 frames) is exercised.
"""

import argparse
import contextlib
import sys
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Relative tolerances are deliberately loose enough for fp32 accumulation order but far
# tighter than any structural error (a missing 2x factor is ~100% off).
TOLERANCES = {
    'fp32': dict(output=(1e-4, 2e-2), grad=(2e-3, 5e-2)),
    'bf16': dict(output=(2e-2, 5e-2), grad=(5e-2, 2e-1)),
}


def relative_error(actual, expected):
    """Max absolute difference and that difference relative to the largest reference value.

    An element-wise ratio would blow up on reference entries that are legitimately near zero,
    which says nothing about the operator.
    """
    actual = actual.detach().float()
    expected = expected.detach().float()
    diff = (actual - expected).abs()
    scale = expected.abs().max().clamp(min=1e-6)
    return float(diff.max()), float(diff.max() / scale)


def assert_close(name, actual, expected, atol, rtol):
    abs_max, rel_max = relative_error(actual, expected)
    ok = abs_max <= atol or rel_max <= rtol
    print(f'    {name}: max|diff|={abs_max:.3e} max rel={rel_max:.3e} '
          f'{"ok" if ok else "FAIL"}', flush=True)
    if not ok:
        raise AssertionError(f'{name} outside tolerance (atol={atol}, rtol={rtol})')


def scan_case(length, dim, batch, dstate, negative, device, tolerance, dtype=torch.float32):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    generator = torch.Generator(device=device).manual_seed(4321 + length)
    u = torch.randn(batch, dim, length, device=device, generator=generator).to(dtype)
    delta = (torch.rand(batch, dim, length, device=device, generator=generator) * 0.1).to(dtype)
    A = -torch.exp(torch.randn(dim, dstate, device=device, generator=generator) * 0.5)
    Bvar = torch.randn(batch, dstate, length, device=device, generator=generator).to(dtype)
    Cvar = torch.randn(batch, dstate, length, device=device, generator=generator).to(dtype)
    D = torch.randn(dim, device=device, generator=generator)
    z = torch.randn(batch, dim, length, device=device, generator=generator).to(dtype)
    delta_bias = torch.randn(dim, device=device, generator=generator) * 0.1

    def run(scan):
        leaves = [t.clone().requires_grad_(True) for t in (u, delta, A, Bvar, Cvar, D, z, delta_bias)]
        out = scan(*leaves, delta_softplus=True, negative=negative)
        weight = torch.linspace(-1, 1, out.numel(), device=device).reshape(out.shape)
        (out * weight).sum().backward()
        return out, [leaf.grad for leaf in leaves]

    native_out, native_grads = run(selective_scan_fn)
    ref_out, ref_grads = run(selective_scan_ref)

    if not torch.isfinite(native_out).all():
        raise AssertionError(f'non-finite forward output for length={length} negative={negative}')
    assert_close(f'out (L={length})', native_out, ref_out, *tolerance['output'])
    for name, native, ref in zip(('du', 'ddelta', 'dA', 'dB', 'dC', 'dD', 'dz', 'ddelta_bias'),
                                 native_grads, ref_grads):
        if not torch.isfinite(native).all():
            raise AssertionError(f'non-finite {name} for length={length} negative={negative}')
        assert_close(f'{name} (L={length})', native, ref, *tolerance['grad'])


def check_reference_gradcheck():
    from mamba_ssm.ops.scan_reference import selective_scan_reference

    torch.manual_seed(0)
    batch, dim, dstate, length = 2, 3, 2, 5
    u = torch.randn(batch, dim, length, dtype=torch.float64) * 0.5
    delta = torch.rand(batch, dim, length, dtype=torch.float64) * 0.2
    A = -torch.rand(dim, dstate, dtype=torch.float64)
    Bvar = torch.randn(batch, dstate, length, dtype=torch.float64)
    Cvar = torch.randn(batch, dstate, length, dtype=torch.float64)
    delta_bias = torch.randn(dim, dtype=torch.float64) * 0.1
    for negative in (False, True):
        inputs = tuple(t.clone().requires_grad_(True) for t in (u, delta, A, Bvar, Cvar, delta_bias))
        assert torch.autograd.gradcheck(
            lambda *_args: selective_scan_reference(*_args, delta_softplus=True, negative=negative),
            inputs, eps=1e-6, atol=1e-5, rtol=1e-4,
        ), f'gradcheck failed for negative={negative}'
        print(f'    gradcheck negative={negative}: ok', flush=True)


def check_mamba_module(d_model, device, tolerance):
    import mamba_ssm.modules.mamba_simple as module
    from mamba_ssm.modules.mamba_simple import Mamba

    def build(negative):
        torch.manual_seed(4321)
        m = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2,
                  scan_mode='negative' if negative else 'baseline').to(device)
        return m

    baseline = build(False)
    negative = build(True)
    for (name, left), (other, right) in zip(baseline.state_dict().items(),
                                            negative.state_dict().items()):
        if other != name or not torch.equal(left, right):
            raise AssertionError(f'initialisation differs between the groups at {name}')
    print('    identical initialisation for both modes: ok', flush=True)

    length = 64
    x = torch.randn(2, length, d_model, device=device, generator=torch.Generator(device=device).manual_seed(7))

    def run(model, reference):
        model.zero_grad(set_to_none=True)
        data = x.clone().requires_grad_(True)
        patch = (mock.patch.object(module, 'mamba_inner_fn', _reference_inner) if reference
                 else contextlib.nullcontext())
        with patch:
            out = model(data)
            (out.square().mean()).backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        return out.detach(), grads

    outputs = {}
    for negative_mode in (False, True):
        tol = tolerance
        native_out, native_grads = run(build(negative_mode), reference=False)
        ref_out, ref_grads = run(build(negative_mode), reference=True)
        assert_close(f'{"negative" if negative_mode else "baseline"} mamba out',
                     native_out, ref_out, *tol['output'])
        for name in native_grads:
            assert_close(f'{"negative" if negative_mode else "baseline"} grad {name}',
                         native_grads[name], ref_grads[name], *tol['grad'])
        if not all(torch.isfinite(g).all() for g in native_grads.values()):
            raise AssertionError('non-finite parameter gradient')
        if float(native_grads['A_log'].abs().max()) == 0.0:
            raise AssertionError('A_log gradient is identically zero')
        outputs[negative_mode] = native_out

    difference = float((outputs[True] - outputs[False]).abs().max())
    if difference < 1e-3:
        raise AssertionError('negative mode produced the same output as baseline; '
                             'the branch is not engaged')
    print(f'    negative vs baseline output difference: {difference:.3e} (mechanism engaged)',
          flush=True)


def _reference_inner(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                     out_proj_weight, out_proj_bias, A, B=None, C=None, D=None,
                     delta_bias=None, B_proj_bias=None, C_proj_bias=None,
                     delta_softplus=True, negative=False):
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_ref
    # The patched callable receives the same arguments; forward the flag explicitly.
    return mamba_inner_ref(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                           out_proj_weight, out_proj_bias, A, B, C, D, delta_bias,
                           B_proj_bias, C_proj_bias, delta_softplus, negative=negative)


def check_full_model(device, precision):
    from model.Hyperformer import Model

    model_args = dict(
        num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph',
        graph_args=dict(labeling_mode='spatial'),
        joint_label=[0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1],
        scan_mode='negative',
    )
    torch.manual_seed(4321)
    model = Model(**model_args).to(device)
    modes = {block.mamba.scan_mode for block in _mamba_blocks(model)}
    if modes != {'negative'}:
        raise AssertionError(f'expected every block to be negative, found {modes}')
    print(f'    {len(_mamba_blocks(model))} Mamba blocks, all scan_mode=negative: ok', flush=True)

    data = torch.randn(2, 3, 64, 25, 2, device=device)
    label = torch.randint(0, 60, (2,), device=device)
    autocast = (torch.cuda.amp.autocast(dtype=torch.bfloat16) if precision == 'bf16'
                else contextlib.nullcontext())
    with autocast:
        output, _ = model(data, torch.nn.functional.one_hot(label, num_classes=60))
    if output.shape != (2, 60) or not torch.isfinite(output).all():
        raise AssertionError('full model produced an invalid output')
    output.float().square().mean().backward()
    missing = sorted(name for name, parameter in model.named_parameters()
                     if parameter.grad is None)
    # TCN_ViT_unit still builds pe_proj for the disabled ViT branch; it is never reached and
    # is therefore not expected to receive a gradient. Anything else missing is a real error.
    unexpected = [name for name in missing if not name.endswith('pe_proj.weight')]
    if unexpected:
        raise AssertionError(f'parameters without a gradient: {unexpected}')
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise AssertionError(f'{name} has a non-finite gradient')
    received = sum(1 for _, parameter in model.named_parameters() if parameter.grad is not None)
    print(f'    full-model forward/backward with 64/32/16 frames: ok '
          f'({received} tensors receive gradients, {len(missing)} unused pe_proj)', flush=True)


def _mamba_blocks(model):
    return [module for name, module in model.named_modules() if name.endswith('.mamba')
            and hasattr(module, 'scan_mode') and module.__class__.__name__ == 'Mamba']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--skip-model', action='store_true',
                        help='skip the full-model smoke test')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; run this on the training machine.')
    torch.cuda.set_device(args.device)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('This device does not support BF16.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', args.device)
    tolerance = TOLERANCES[args.precision]

    import selective_scan_cuda
    if not getattr(selective_scan_cuda, 'negative_scan_supported', False):
        raise SystemExit('the loaded selective_scan_cuda has no negative branch; rebuild it with '
                         'python setup.py build_ext --inplace')
    print('extension advertises the negative branch: ok', flush=True)

    print('scan level: fused vs reference', flush=True)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    for length in (1, 17, 64, 130, 300):
        for negative in (False, True):
            print(f'  L={length} negative={negative}', flush=True)
            scan_case(length, dim=16, batch=3, dstate=16, negative=negative,
                      device=device, tolerance=tolerance, dtype=dtype)
    print('  realistic d_inner=1250, L=64', flush=True)
    scan_case(64, dim=1250, batch=2, dstate=16, negative=True, device=device,
              tolerance=tolerance, dtype=dtype)

    print('reference gradients (float64 gradcheck)', flush=True)
    check_reference_gradcheck()

    print('Mamba module level, d_model=64', flush=True)
    check_mamba_module(d_model=64, device=device, tolerance=tolerance)

    if not args.skip_model:
        print('full Simba model', flush=True)
        check_full_model(device, args.precision)

    torch.cuda.synchronize()
    print(f'\nAll negative-eigenvalue checks passed on {torch.cuda.get_device_name(device)} '
          f'({args.precision}). No NTU data or training files were touched.')


if __name__ == '__main__':
    main()
