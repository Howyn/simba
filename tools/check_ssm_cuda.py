"""Check the new SSM mechanisms against the reference scan on AutoDL CUDA.

Uses synthetic sequences only; does not read NTU or write training results.
    python tools/check_ssm_cuda.py
    python tools/check_ssm_cuda.py --precision bf16
"""

import argparse
import contextlib
from pathlib import Path
import sys
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_shared_weights(model, baseline):
    result = model.load_state_dict(baseline.state_dict(), strict=False)
    if result.unexpected_keys or any(
            not (name == 'bidir_logit' or name.startswith('motion_'))
            for name in result.missing_keys):
        raise RuntimeError(f'Unexpected shared-weight mismatch: {result}')


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
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; run this check in the existing AutoDL environment.')
    torch.cuda.set_device(args.device)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('This CUDA device does not support BF16.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from mamba_ssm.modules.mamba_simple import Mamba
    import mamba_ssm.modules.mamba_skeleton as skeleton
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    device = torch.device('cuda', args.device)
    dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32

    def autocast():
        return (torch.cuda.amp.autocast(dtype=dtype) if dtype != torch.float32
                else contextlib.nullcontext())

    output_tol = dict(atol=2e-3, rtol=3e-2) if dtype == torch.bfloat16 else dict(atol=3e-5, rtol=5e-4)
    gradient_tol = dict(atol=3e-4, rtol=8e-2) if dtype == torch.bfloat16 else dict(atol=3e-6, rtol=3e-3)
    model_args = dict(d_model=args.d_model, d_state=16, d_conv=4, expand=2)
    inputs = torch.randn(args.batch_size, args.frames, args.d_model, device=device)
    torch.manual_seed(4321)
    baseline = Mamba(**model_args).to(device).eval()
    baseline_state = {name: tensor.detach().clone()
                      for name, tensor in baseline.state_dict().items()}
    torch.manual_seed(4321)
    combined = skeleton.SkeletonMamba(**model_args, bidirectional=True,
                                      motion_delta=True).to(device).eval()
    for name, tensor in baseline_state.items():
        if not torch.equal(tensor, combined.state_dict()[name]):
            raise RuntimeError(f'Shared initialization changed: {name}')
    print('Shared base initialization: bitwise equal', flush=True)

    motion_zero = skeleton.SkeletonMamba(**model_args, motion_delta=True).to(device).eval()
    load_shared_weights(motion_zero, baseline)
    with torch.no_grad(), autocast():
        expected, actual = baseline(inputs), motion_zero(inputs)
    torch.testing.assert_close(actual.float(), expected.float(), **output_tol)
    print('Zero-initialized motion branch vs original fused Mamba: passed', flush=True)

    for name, flags in (
            ('ssm_bidir', dict(bidirectional=True)),
            ('ssm_motion', dict(motion_delta=True)),
            ('ssm_bidir_motion', dict(bidirectional=True, motion_delta=True))):
        model = skeleton.SkeletonMamba(**model_args, **flags).to(device).train()
        load_shared_weights(model, baseline)
        # Exercise a learned, nonzero motion correction in reference comparison.
        if flags.get('motion_delta'):
            torch.nn.init.normal_(model.motion_up.weight, std=0.02)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def evaluate(reference):
            model.zero_grad(set_to_none=True)
            data = inputs.detach().clone().requires_grad_()
            scan_context = (mock.patch.object(skeleton, 'selective_scan_fn', selective_scan_ref)
                            if reference else contextlib.nullcontext())
            with scan_context, autocast():
                output = model(data)
                loss = output.float().square().mean()
            if output.shape != data.shape or not torch.isfinite(loss):
                raise RuntimeError(f'{name}: invalid shape/output/loss')
            loss.backward()
            gradients = {}
            for key, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'{name}: missing/non-finite gradient for {key}')
                gradients[key] = parameter.grad.detach().clone()
            if not torch.isfinite(data.grad).all():
                raise RuntimeError(f'{name}: non-finite input gradient')
            return output.detach().float(), data.grad.detach().float(), gradients

        native_output, native_input_grad, native_grads = evaluate(False)
        reference_output, reference_input_grad, reference_grads = evaluate(True)
        torch.testing.assert_close(native_output, reference_output, **output_tol)
        torch.testing.assert_close(native_input_grad, reference_input_grad, **gradient_tol)
        for key in native_grads:
            try:
                torch.testing.assert_close(native_grads[key].float(),
                                           reference_grads[key].float(), **gradient_tol)
            except AssertionError as error:
                raise AssertionError(f'{name}: native/reference gradient mismatch for {key}') from error
        if flags.get('bidirectional') and native_grads['bidir_logit'].abs().sum() == 0:
            raise RuntimeError(f'{name}: direction gate has zero gradient')
        if flags.get('motion_delta') and native_grads['motion_up.weight'].abs().sum() == 0:
            raise RuntimeError(f'{name}: motion branch has zero gradient')
        # Take an optimizer step with the native gradients, then check eval output.
        for key, parameter in model.named_parameters():
            parameter.grad = native_grads[key]
        optimizer.step()
        model.eval()
        with torch.no_grad(), autocast():
            after_step = model(inputs)
        if not torch.isfinite(after_step).all() or any(
                not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f'{name}: non-finite values after optimizer step')
        print(f'{name}: native/reference output + gradients + optimizer/eval passed; '
              f'parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)

    torch.cuda.synchronize()
    print(f'All checks passed on {torch.cuda.get_device_name(device)} '
          f'with {args.precision}; no NTU data or training files were used.')


if __name__ == '__main__':
    main()
