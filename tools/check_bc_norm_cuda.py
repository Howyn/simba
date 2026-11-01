"""Check B/C normalization's real CUDA scan output and gradients on AutoDL.

Synthetic data only; no NTU data, experiment logs, or checkpoint writes.
    python tools/check_bc_norm_cuda.py
"""
import argparse
import contextlib
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BC_NAMES = ('bc_B_gain', 'bc_C_gain', 'bc_B_bias', 'bc_C_bias')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--d-model', type=int, default=625)
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if min(args.d_model, args.frames, args.batch_size) < 1:
        parser.error('d-model, frames and batch-size must be positive')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; run this check in the existing AutoDL environment.')
    torch.cuda.set_device(args.device)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('This CUDA device does not support BF16.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(ROOT))
    from mamba_ssm.modules.mamba_simple import Mamba
    import mamba_ssm.modules.mamba_skeleton as skeleton
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    device = torch.device('cuda', args.device)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
    parameters = dict(d_model=args.d_model, d_state=16, d_conv=4, expand=2)
    torch.manual_seed(4321)
    baseline = Mamba(**parameters).to(device)
    torch.manual_seed(4321)
    model = skeleton.SkeletonMamba(**parameters, bc_norm=True,
                                   bc_norm_gain=.1, bc_norm_bias=.1,
                                   bc_norm_eps=1e-6).to(device)
    for name, tensor in baseline.state_dict().items():
        if not torch.equal(tensor, model.state_dict()[name]):
            raise RuntimeError(f'Shared initialization changed: {name}')
    if sum(getattr(model, name).numel() for name in BC_NAMES) != 64:
        raise RuntimeError('B/C normalization must add exactly 64 parameters at state size 16')
    print('Shared initialization bitwise equal; four B/C vectors add 64 parameters.', flush=True)
    data = torch.randn(args.batch_size, args.frames, args.d_model, device=device)

    def autocast():
        return (torch.autocast(device_type='cuda', dtype=dtype) if dtype != torch.float32
                else contextlib.nullcontext())

    output_tol = dict(atol=3e-3, rtol=5e-2) if dtype == torch.bfloat16 else dict(atol=3e-5, rtol=5e-4)
    gradient_tol = dict(atol=3e-4, rtol=1e-1) if dtype == torch.bfloat16 else dict(atol=3e-6, rtol=3e-3)
    native_scan = skeleton.selective_scan_fn
    calls = []

    def record_scan(*scan_args, **scan_kwargs):
        calls.append((scan_args, scan_kwargs))
        return native_scan(*scan_args, **scan_kwargs)

    def reference_scan(*scan_args, **scan_kwargs):
        # The CUDA kernel accumulates the recurrence and SiLU gate in FP32.
        # Keep the reference's einsums from being downcast by outer autocast.
        if scan_kwargs.get('z') is not None:
            scan_kwargs['z'] = scan_kwargs['z'].float()
        with torch.autocast(device_type='cuda', enabled=False):
            return selective_scan_ref(*scan_args, **scan_kwargs)

    def evaluate(reference):
        model.zero_grad(set_to_none=True)
        inputs = data.detach().clone().requires_grad_()
        scan = reference_scan if reference else record_scan
        with mock.patch.object(skeleton, 'selective_scan_fn', side_effect=scan), autocast():
            output = model(inputs)
            loss = output.float().square().mean()
        if output.shape != inputs.shape or not torch.isfinite(loss):
            raise RuntimeError('Invalid B/C-normalized model output or loss')
        loss.backward()
        grads = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f'Missing/non-finite gradient: {name}')
            grads[name] = parameter.grad.detach().clone()
        if not torch.isfinite(inputs.grad).all():
            raise RuntimeError('Non-finite input gradient')
        return output.detach().float(), inputs.grad.detach().float(), grads

    native_output, native_input_gradient, native_grads = evaluate(False)
    if len(calls) != 1:
        raise RuntimeError(f'Expected one scan for the B/C-only path, got {len(calls)}')
    scan_args, scan_kwargs = calls.pop()
    if scan_args[5] is None or scan_kwargs.get('z') is None:
        raise RuntimeError('Single-direction B/C path must retain native D skip and z gate')
    reference_output, reference_input_gradient, reference_grads = evaluate(True)
    torch.testing.assert_close(native_output, reference_output, **output_tol)
    torch.testing.assert_close(native_input_gradient, reference_input_gradient, **gradient_tol)
    for name, gradient in native_grads.items():
        try:
            torch.testing.assert_close(gradient.float(), reference_grads[name].float(), **gradient_tol)
        except AssertionError as error:
            raise AssertionError(f'Native/reference gradient mismatch: {name}') from error
    for name in BC_NAMES:
        if native_grads[name].abs().sum() == 0:
            raise RuntimeError(f'B/C parameter has zero gradient: {name}')
        # Prevent a loose absolute tolerance from masking errors in small new gradients.
        relative = ((native_grads[name].float() - reference_grads[name].float()).norm()
                    / reference_grads[name].float().norm().clamp_min(1e-12)).item()
        if relative > (.2 if dtype == torch.bfloat16 else .02):
            raise RuntimeError(f'B/C gradient relative error is too large: {name} ({relative:.3g})')
    print('Single native scan + reference output/input/parameter gradients: passed.', flush=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    before = {name: getattr(model, name).detach().clone() for name in BC_NAMES}
    for name, parameter in model.named_parameters():
        parameter.grad = native_grads[name]
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError('Non-finite parameter after optimizer update')
    if not any(not torch.equal(before[name], getattr(model, name)) for name in BC_NAMES):
        raise RuntimeError('B/C parameters did not update')
    model.eval()
    with torch.no_grad(), autocast():
        output = model(data)
    if not torch.isfinite(output).all():
        raise RuntimeError('Non-finite evaluation output after optimizer update')
    torch.cuda.synchronize()
    print(f'B/C normalization {args.precision}: optimizer update/eval passed on '
          f'{torch.cuda.get_device_name(device)}; no NTU data or output files were used.')


if __name__ == '__main__':
    main()
