"""Read-only small-batch CUDA probe of a saved SSM model.

Runs forward/backward only: no optimizer, no weight updates, no output files.
BatchNorm buffers can change in this discarded in-memory model copy.
"""
import argparse
from itertools import islice
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]


def delta_change_metrics(rank, correction, weight, bias, scale):
    """Measure the actual positive delta change after dt_proj and softplus."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        rank = rank.detach().float()
        correction = correction.detach().float().reshape_as(rank)
        weight, bias = weight.detach().float(), bias.detach().float()
        # Use the same W @ rank.T layout as SkeletonMamba. At tiny correction
        # scales, changing GEMM layout/cast order can obscure rounding effects.
        original = F.softplus((weight @ rank.t()).t() + bias)
        changed = F.softplus((weight @ (rank + scale * correction.tanh()).t()).t() + bias)
        difference = changed - original
        if not torch.isfinite(changed).all() or not torch.isfinite(original).all():
            raise RuntimeError('Non-finite delta in the signal probe')
        return dict(base_square=original.double().square().sum().item(),
                    difference_square=difference.double().square().sum().item(),
                    max_relative=(difference.abs() / original.clamp_min(1e-12)).max().item())


def gradient_metrics(parameter, weight_decay):
    """Current CE gradient versus the old coupled decay term, before momentum."""
    import torch
    rms = parameter.detach().double().square().mean().sqrt().item()
    decay = weight_decay * rms
    if parameter.grad is None:
        return dict(parameter_rms=rms, task_grad_rms=None, decay_grad_rms=decay, ratio=None)
    if not torch.isfinite(parameter.grad).all():
        raise RuntimeError('Non-finite task gradient in the signal probe')
    gradient = parameter.grad.detach().double().square().mean().sqrt().item()
    ratio = gradient / decay if decay > 0 else (math.inf if gradient else 0.)
    return dict(parameter_rms=rms, task_grad_rms=gradient, decay_grad_rms=decay, ratio=ratio)


class MotionDeltaProbe:
    """Observe a real module's low-rank values without changing its forward."""
    def __init__(self, module):
        self.module, self.rank, self.measurements = module, None, []
        self.handles = [module.x_proj.register_forward_hook(self.capture_rank),
                        module.motion_up.register_forward_hook(self.measure_correction)]

    def capture_rank(self, module, inputs, output):
        self.rank = output[:, :self.module.dt_rank].detach()

    def measure_correction(self, module, inputs, output):
        if self.rank is None:
            raise RuntimeError('Motion correction appeared before its delta projection')
        self.measurements.append(delta_change_metrics(
            self.rank, output, self.module.dt_proj.weight,
            self.module.dt_proj.bias, self.module.motion_scale))
        self.rank = None

    def summary(self):
        if not self.measurements:
            raise RuntimeError('Motion branch was not executed')
        base = sum(item['base_square'] for item in self.measurements)
        change = sum(item['difference_square'] for item in self.measurements)
        return math.sqrt(change / max(base, 1e-300)), max(
            item['max_relative'] for item in self.measurements)

    def close(self):
        for handle in self.handles:
            handle.remove()


def assert_parameters_unchanged(model, before):
    import torch
    for name, parameter in model.named_parameters():
        if name not in before or not torch.equal(parameter.detach().cpu(), before[name]):
            raise RuntimeError(f'Parameter unexpectedly changed during probe: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/nturgbd-cross-subject/joint_ssm_bidir_motion_20ep.yaml')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'work_dir/ntu60/xsub/full_ssm_bidir_motion_20ep/runs-best.pt')
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if min(args.batches, args.batch_size) < 1:
        parser.error('batches and batch-size must be positive')
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        import yaml
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; run this small-batch probe on AutoDL')
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        sys.path.insert(0, str(ROOT))
        from tools.diagnose_ssm_results import load_checkpoint_cpu
        from model.Hyperformer import Model
        from feeders.feeder_ntu import Feeder
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if config['model_args'].get('ctr_layers') or config['model_args'].get('fr_layers'):
            raise ValueError('This probe expects the SSM-only experiment without CTR/FR')
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = Model(**config['model_args']).cuda(args.device).float()
        checkpoint = load_checkpoint_cpu(args.checkpoint)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
        del checkpoint, state
        model.train()
        before = {name: parameter.detach().to('cpu', copy=True)
                  for name, parameter in model.named_parameters()}
        probes, gradients = {}, {}
        for layer in range(1, 11):
            inner = getattr(model, f'l{layer}').mamba.mamba
            if getattr(inner, 'motion_delta', False):
                if not getattr(inner, 'bidirectional', False):
                    raise ValueError('This probe expects the combined checkpoint with both mechanisms')
                probes[layer] = MotionDeltaProbe(inner)
                for name in ('bidir_logit', 'motion_down.weight', 'motion_up.weight'):
                    gradients[(layer, name)] = []
        if not probes:
            raise ValueError('No combined SSM modules found; synchronize the model code')
        print('Loading the existing training dataset for a small-batch diagnostic; '
              'no optimizer updates or checkpoint writes.', flush=True)
        loader = torch.utils.data.DataLoader(Feeder(**config['train_feeder_args']),
                                             batch_size=args.batch_size, shuffle=True, num_workers=0)
        batches, samples = 0, 0
        print(f'Read-only probe: up to {args.batches} minibatches, batch={args.batch_size}, '
              'FP32, training augmentations, no optimizer steps.', flush=True)
        try:
            for batch, (inputs, labels, _) in enumerate(islice(loader, args.batches), 1):
                model.zero_grad(set_to_none=True)
                inputs = inputs.cuda(args.device).float()
                labels = labels.cuda(args.device).long()
                logits, _ = model(inputs, F.one_hot(labels, config['model_args']['num_class']))
                loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite CE loss in the signal probe')
                loss.backward()
                batches, samples = batch, samples + labels.numel()
                for (layer, name), rows in gradients.items():
                    inner = getattr(model, f'l{layer}').mamba.mamba
                    parameter = dict(inner.named_parameters())[name]
                    rows.append(gradient_metrics(parameter, config['weight_decay']))
                print(f'  minibatch {batch}: samples={labels.numel()}, CE={loss.item():.6f}', flush=True)
            if not batches:
                raise RuntimeError('No training samples were available')
            assert_parameters_unchanged(model, before)
            for layer, probe in probes.items():
                relative, maximum = probe.summary()
                print(f'l{layer} effective delta: relative_L2={relative:.6e}, max_relative={maximum:.6e}')
                for name in ('bidir_logit', 'motion_down.weight', 'motion_up.weight'):
                    rows = gradients[(layer, name)]
                    task = [row['task_grad_rms'] for row in rows if row['task_grad_rms'] is not None]
                    decay = rows[0]['decay_grad_rms']
                    mean = sum(task) / len(task) if task else None
                    ratio = (mean / decay if decay else math.inf if mean else 0.) if mean is not None else None
                    value = 'missing' if mean is None else f'{mean:.6e}'
                    ratio_text = 'missing' if ratio is None else f'{ratio:.6e}'
                    print(f'  {name}: mean_batch_task_grad_rms={value}, '
                          f'old_decay_grad_rms={decay:.6e}, task/decay={ratio_text}')
        finally:
            for probe in probes.values():
                probe.close()
        print(f'Completed {batches} minibatches / {samples} samples. Parameters unchanged; no files written.')
        print('Gradients use these small batches and the original coupled decay term before momentum. '
              'They do not establish accuracy gains or multi-seed stability.')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
