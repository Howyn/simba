"""Measure the discrete state-retention coefficients of every Mamba layer on real data.

This is the mechanism check from the experiment plan: it answers "is the negative branch
actually active, and where?" without touching training.

For every Mamba block it recomputes the pre-scan quantities from the block's own input and
weights (same arithmetic as the fused kernel, just in Python) and reports, per layer:

    min / max / mean of the transition, the fraction of negative transitions,
    the fraction of transitions with |a| < 1e-3, and the gradient norms of A_log and
    dt_proj after a short training-style backward pass.

Nothing here changes the model or consumes random numbers from the training stream.

    python tools/scan_statistics.py --config config/nturgbd-cross-subject/joint_autodl_negative.yaml
    python tools/scan_statistics.py --batches 4            # uses random skeletons if no NPZ

Add --checkpoint /path/to/runs-best.pt to measure a trained model instead of a fresh one.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # The scan has no meaning outside a compiled mamba build.
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover - fallback for an unusual environment
    causal_conv1d_fn = None


class LayerStats:
    """Accumulates transition statistics for one Mamba layer."""

    def __init__(self):
        self.count = 0
        self.negative = 0
        self.near_zero = 0
        self.total = 0.0
        self.minimum = float('inf')
        self.maximum = float('-inf')

    def update(self, transitions):
        flat = transitions.detach().float()
        self.count += flat.numel()
        self.negative += int((flat < 0).sum())
        self.near_zero += int((flat.abs() < 1e-3).sum())
        self.total += float(flat.sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))

    def row(self):
        if self.count == 0:
            return dict(min=float('nan'), max=float('nan'), mean=float('nan'),
                        negative=float('nan'), near_zero=float('nan'))
        return dict(min=self.minimum, max=self.maximum, mean=self.total / self.count,
                    negative=self.negative / self.count,
                    near_zero=self.near_zero / self.count)


def transition_statistics(module, inputs, chunk_size):
    """Recompute 2 * exp(softplus(dt) * A) - 1 (or exp(...)) for one Mamba module."""
    x = inputs[0]
    states = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            piece = x[start:start + chunk_size]
            xz = torch.nn.functional.linear(piece, module.in_proj.weight, module.in_proj.bias)
            xs, _ = xz.chunk(2, dim=-1)
            xs = xs.transpose(1, 2)
            if causal_conv1d_fn is not None:
                xs = causal_conv1d_fn(xs, module.conv1d.weight[:, 0, :], module.conv1d.bias,
                                      activation='silu')
            else:
                xs = torch.nn.functional.silu(module.conv1d(xs)[..., :piece.shape[1]])
            x_dbl = module.x_proj(xs.transpose(1, 2))
            delta = x_dbl[..., :module.dt_rank]
            delta = torch.nn.functional.linear(delta, module.dt_proj.weight)
            delta = torch.nn.functional.softplus(delta + module.dt_proj.bias)
            decay = -torch.exp(module.A_log.float())
            # (batch, d_inner, d_state, length)
            transitions = torch.exp(delta.transpose(1, 2)[:, :, None, :] * decay[None, :, :, None])
            if module.negative:
                transitions = 2.0 * transitions - 1.0
            states.append(transitions)
    return states


def build_model(config_path, checkpoint, device):
    config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    from model.Hyperformer import Model

    model_args = dict(config['model_args'])
    model = Model(**model_args).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
        print(f'loaded checkpoint {checkpoint}')
    return model, config


def load_batches(config, batches, device, batch_size):
    data_path = config['test_feeder_args'].get('data_path')
    if data_path and os.path.isfile(data_path):
        from feeders.feeder_ntu import Feeder
        args = dict(config['test_feeder_args'])
        args.pop('data_path', None)
        dataset = Feeder(data_path=data_path, **args)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             drop_last=True, num_workers=0)
        print(f'using {data_path} ({len(dataset)} samples, test split)')
        produced = []
        for index, (data, _, _) in enumerate(loader):
            if index >= batches:
                break
            produced.append(data.to(device, dtype=torch.float32))
        return produced, 'real'
    print('no NPZ found; falling back to random skeleton batches')
    generator = torch.Generator(device=device).manual_seed(0)
    return [torch.randn(batch_size, 3, 64, 25, 2, device=device, generator=generator)
            for _ in range(batches)], 'random'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/nturgbd-cross-subject/joint_autodl_negative.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=4,
                        help='samples per recomputation chunk; lower it if memory is tight')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; run this on the training machine.')
    device = torch.device('cuda', args.device)
    model, config = build_model(args.config, args.checkpoint, device)
    mode = config['model_args'].get('scan_mode', 'baseline')
    blocks = [(name, module) for name, module in model.named_modules()
              if module.__class__.__name__ == 'Mamba']
    for name, module in blocks:
        if module.scan_mode != mode:
            raise SystemExit(f'{name} is {module.scan_mode} but the config says {mode}')

    stats = {index: LayerStats() for index in range(len(blocks))}
    hooks = []
    for index, (_, module) in enumerate(blocks):
        def hook(mod, inputs, output, index=index):
            for transitions in transition_statistics(mod, inputs, args.chunk_size):
                stats[index].update(transitions)
        hooks.append(module.register_forward_hook(hook))

    batches, source = load_batches(config, args.batches, device, args.batch_size)
    model.eval()
    with torch.no_grad():
        for data in batches:
            label = torch.zeros(data.shape[0], dtype=torch.long, device=device)
            model(data, torch.nn.functional.one_hot(label, num_classes=60))
    for handle in hooks:
        handle.remove()

    # Short training-style pass for gradient norms.
    model.train()
    grad_norms = {index: [] for index in range(len(blocks))}
    for data in batches[:2]:
        label = torch.randint(0, 60, (data.shape[0],), device=device)
        model.zero_grad(set_to_none=True)
        output, _ = model(data, torch.nn.functional.one_hot(label, num_classes=60))
        torch.nn.functional.cross_entropy(output, label).backward()
        for index, (_, module) in enumerate(blocks):
            grad_norms[index].append((
                float(module.A_log.grad.norm()) if module.A_log.grad is not None else float('nan'),
                float(module.dt_proj.weight.grad.norm()) if module.dt_proj.weight.grad is not None else float('nan'),
            ))
    model.zero_grad(set_to_none=True)

    print(f'\nscan_mode={mode}  source={source}  batches={len(batches)}  '
          f'device={torch.cuda.get_device_name(device)}')
    header = (f'{"layer":<6}{"min":>10}{"max":>10}{"mean":>10}{"frac<0":>10}'
              f'{"frac|a|<1e-3":>14}{"|grad A_log|":>14}{"|grad dt_proj|":>16}')
    print(header)
    for index, (name, _) in enumerate(blocks):
        row = stats[index].row()
        gradients = grad_norms[index][-1] if grad_norms[index] else (float('nan'), float('nan'))
        print(f'{name:<6}{row["min"]:>10.4f}{row["max"]:>10.4f}{row["mean"]:>10.4f}'
              f'{row["negative"]:>10.3f}{row["near_zero"]:>14.4f}'
              f'{gradients[0]:>14.3e}{gradients[1]:>16.3e}')

    if mode == 'negative':
        share = sum(stats[i].negative for i in stats) / max(1, sum(stats[i].count for i in stats))
        print(f'\noverall negative share: {share:.3f}')
        if share == 0:
            raise SystemExit('no negative transitions were produced; the branch is not engaged')
    print('\nGradient norms are from one backward pass over the last measured batch.')


if __name__ == '__main__':
    main()
