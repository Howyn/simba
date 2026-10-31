"""Run one synthetic forward/backward/evaluation for an experiment on AutoDL.

The SSM modes also verify module placement and gradients in the new branches.
No dataset, checkpoint, or experiment log is read or written.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.Hyperformer import Model  # noqa: E402


def check_ssm_modules(model, config):
    """Check that the requested SSM modules, and no CTR/FR modules, are active."""
    if config['model_args'].get('ctr_layers') or config['model_args'].get('fr_layers'):
        raise RuntimeError('SSM experiments must not enable CTR-GC or FR-Head')
    if model.fr_head is not None:
        raise RuntimeError('FR-Head unexpectedly enabled for an SSM experiment')
    bidir_layers = set(config['model_args'].get('ssm_bidir_layers', ()))
    motion_layers = set(config['model_args'].get('ssm_motion_layers', ()))
    branches = []
    for layer in range(1, 11):
        block = getattr(model, f'l{layer}')
        if type(block.shift_gcn).__name__ == 'CTRGCUnit':
            raise RuntimeError(f'CTR-GC unexpectedly enabled at l{layer}')
        inner = block.mamba.mamba
        if bool(getattr(inner, 'bidirectional', False)) != (layer in bidir_layers):
            raise RuntimeError(f'Incorrect bidirectional SSM placement at l{layer}')
        motion_enabled = bool(getattr(inner, 'motion_delta', False))
        if motion_enabled != (layer in motion_layers):
            raise RuntimeError(f'Incorrect motion-delta placement at l{layer}')
        if layer in bidir_layers:
            parameters = [parameter for name, parameter in inner.named_parameters(recurse=False)
                          if 'bidir' in name]
            if not parameters:
                raise RuntimeError(f'No trainable direction gate found at l{layer}')
            branches.append((f'l{layer} bidirectional gate', parameters))
        if motion_enabled:
            modules = [getattr(inner, name, None)
                       for name in ('motion_down', 'motion_up', 'motion_norm')]
            if not all(isinstance(module, torch.nn.Module) for module in modules):
                raise RuntimeError(f'Incomplete motion-delta branch at l{layer}')
            parameters = [parameter for module in modules for parameter in module.parameters()]
            branches.append((f'l{layer} motion delta', parameters))
    return branches


def check_branch_gradients(branches):
    for name, parameters in branches:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        # The motion branch can initialize its final projection to zero; its
        # earlier projections need not have non-zero gradients on the first step.
        if not gradients or not any(torch.count_nonzero(gradient).item() for gradient in gradients):
            raise RuntimeError(f'{name} did not receive non-zero gradients')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'ctr', 'fr', 'ctr_fr',
                                        'ssm_bidir', 'ssm_motion', 'ssm_bidir_motion'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--tf32', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required by the Simba backbone for this test')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('The current CUDA device does not support BF16')
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    path = ROOT / 'config' / 'nturgbd-cross-subject' / f'joint_{args.mode}_10ep.yaml'
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    torch.manual_seed(config['seed'])
    model = Model(**config['model_args']).cuda()
    branches = check_ssm_modules(model, config) if args.mode.startswith('ssm_') else []
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['base_lr'],
                                momentum=0.9, weight_decay=config['weight_decay'])
    num_classes = config['model_args']['num_class']
    frames = config['train_feeder_args']['window_size']
    points = config['model_args']['num_point']
    persons = config['model_args']['num_person']
    inputs = torch.randn(args.batch_size, 3, frames, points, persons, device='cuda')
    labels = torch.arange(args.batch_size, device='cuda') % num_classes
    with torch.cuda.amp.autocast(enabled=args.precision == 'bf16', dtype=torch.bfloat16):
        logits, second = model(inputs, F.one_hot(labels, num_classes))
        auxiliary = second if model.fr_head is not None else logits.new_zeros(())
        loss = F.cross_entropy(logits, labels) + auxiliary
    if logits.shape != (args.batch_size, num_classes) or not torch.isfinite(loss):
        raise RuntimeError('Invalid model output or loss')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Non-finite gradients')
    if model.fr_head is not None and not any(p.grad is not None for p in model.fr_head.parameters()):
        raise RuntimeError('FR-Head did not receive gradients')
    check_branch_gradients(branches)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('Non-finite parameter after optimizer step')
    memories = {k: v.clone() for k, v in model.state_dict().items() if 'prototypes' in k}
    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(inputs, F.one_hot(labels, num_classes))
    if eval_logits.shape != logits.shape or not torch.isfinite(eval_logits).all():
        raise RuntimeError('Invalid evaluation output')
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in memories.items()):
        raise RuntimeError('FR prototype memory changed during evaluation')
    torch.cuda.synchronize()
    print(f'{args.mode} {args.precision} (TF32={args.tf32}): forward/backward/step/eval passed; loss={loss.item():.4f}, '
          f'weighted FR={auxiliary.item():.4f}, peak GPU memory='
          f'{torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB')
    if branches:
        print('SSM placement and non-zero branch gradients passed: '
              + ', '.join(name for name, _ in branches))


if __name__ == '__main__':
    main()
