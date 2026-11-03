"""Inspect real transition coefficients and gradients on one NTU60 training batch."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch.utils.data import DataLoader
from feeders.feeder_ntu import Feeder
from model.Hyperformer import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/nturgbd-cross-subject/joint_negative_negative.yaml')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--weights', help='Optional checkpoint; default is initialization')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    model = Model(**cfg['model_args']).cuda().train()
    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    observed = {}
    hooks = []
    for n in range(1, 11):
        name = f'l{n}'
        m = getattr(model, name).mamba.mamba
        def inspect(module, inputs, name=name):
            with torch.no_grad():
                h = inputs[0]
                xz = module.in_proj(h).transpose(1, 2)
                x = xz.chunk(2, dim=1)[0]
                x = F.silu(module.conv1d(x)[..., :h.shape[1]])
                features = module.x_proj(rearrange(x, 'b d l -> (b l) d'))
                dt = F.linear(features[:, :module.dt_rank], module.dt_proj.weight, module.dt_proj.bias)
                dt = F.softplus(dt).reshape(h.shape[0], h.shape[1], module.d_inner)
                # Chunk time dimension so diagnostics need not materialize all transitions.
                total = neg = near = 0; minimum = float('inf'); maximum = -float('inf'); total_sum = 0.
                A = -torch.exp(module.A_log.float())
                for part in dt.split(8, dim=1):
                    v = 2 * torch.exp(part.float()[..., None] * A[None, None]) - 1
                    total += v.numel(); neg += (v < 0).sum().item(); near += (v.abs() < 0.01).sum().item()
                    total_sum += v.sum().item(); minimum = min(minimum, v.min().item()); maximum = max(maximum, v.max().item())
                observed[name] = (minimum, maximum, total_sum / total, neg / total, near / total)
        hooks.append(m.register_forward_pre_hook(inspect))
    feeder_args = dict(cfg['train_feeder_args']); feeder_args['random_rot'] = False
    data, labels, *_ = next(iter(DataLoader(Feeder(**feeder_args), batch_size=args.batch_size, num_workers=0)))
    output, _ = model(data.float().cuda(), labels.long().cuda())
    loss = F.cross_entropy(output, labels.long().cuda()); loss.backward()
    if len(observed) != 10:
        raise RuntimeError(f'Expected 10 Mamba calls, observed {len(observed)}')
    print('Note: diagnostics include all 64 resampled positions, including padding; no padding mask is available.')
    print(f'loss={loss.item():.6f}')
    for n in range(1, 11):
        name = f'l{n}'; m = getattr(model, name).mamba.mamba
        stats = observed[name]
        print(f'{name}: mode={m.scan_mode} min={stats[0]:.5g} max={stats[1]:.5g} mean={stats[2]:.5g} '
              f'negative={stats[3]:.3%} near_zero={stats[4]:.3%} '
              f'A_grad={m.A_log.grad.norm().item():.5g} dt_grad={m.dt_proj.weight.grad.norm().item():.5g}')
        if not all(torch.isfinite(p.grad).all() for p in (m.A_log, m.dt_proj.weight)):
            raise RuntimeError(f'Non-finite gradient in {name}')
    for hook in hooks:
        hook.remove()

if __name__ == '__main__':
    main()
