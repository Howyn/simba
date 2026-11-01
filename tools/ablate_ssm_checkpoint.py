"""Inference-only functional ablations of one saved SSM checkpoint.

Default: first 16 test batches only. Use --max-batches 0 for the full test set.
No training, optimizer, output files, or checkpoint updates are performed.
"""
import argparse
from contextlib import contextmanager, ExitStack
from itertools import islice
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'motion_off', 'bidir_off', 'addons_off',
            'state_readout_off', 'mamba_output_off')
_MISSING = object()


def _restore_attribute(module, name, previous):
    if previous is _MISSING:
        delattr(module, name)
    else:
        setattr(module, name, previous)


def _restore_rows(weight, previous):
    import torch
    with torch.no_grad():
        weight[-previous.shape[0]:].copy_(previous)


@contextmanager
def ablation(modules, variant):
    """Apply one temporary intervention, restoring even when forward fails."""
    import torch
    if variant not in VARIANTS:
        raise ValueError(f'Unknown ablation: {variant}')
    with ExitStack() as changes:
        for module in modules:
            if variant == 'addons_off' and hasattr(module, 'motion_delta') and hasattr(module, 'use_fast_path'):
                # Disabling both flags delegates to vanilla Mamba. Keep its
                # explicit path to reduce fused-vs-unfused rounding differences.
                previous = module.use_fast_path
                changes.callback(setattr, module, 'use_fast_path', previous)
                module.use_fast_path = False
            for flag, disabled in (('motion_delta', ('motion_off', 'addons_off')),
                                   ('bidirectional', ('bidir_off', 'addons_off'))):
                if variant in disabled and hasattr(module, flag):
                    previous = module.__dict__.get(flag, _MISSING)
                    changes.callback(_restore_attribute, module, flag, previous)
                    setattr(module, flag, False)
            if variant == 'state_readout_off':
                weight = module.x_proj.weight
                if weight.shape[0] != module.dt_rank + 2 * module.d_state:
                    raise ValueError('Unexpected x_proj layout; cannot isolate C readout')
                previous = weight[-module.d_state:].detach().clone()
                changes.callback(_restore_rows, weight, previous)
                with torch.no_grad():
                    weight[-module.d_state:].zero_()
            elif variant == 'mamba_output_off':
                previous = module.__dict__.get('forward', _MISSING)
                changes.callback(_restore_attribute, module, 'forward', previous)
                module.forward = lambda hidden_states, *args, **kwargs: torch.zeros_like(hidden_states)
        yield


def record(metrics, logits, labels, reference):
    import torch
    if logits.shape != reference.shape or logits.shape[0] != labels.numel():
        raise ValueError('Unexpected prediction shape during functional ablation')
    if not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite logits during functional ablation')
    predicted = logits.argmax(dim=1)
    metrics['samples'] += labels.numel()
    metrics['correct'] += (predicted == labels).sum().item()
    metrics['changed'] += (predicted != reference.argmax(dim=1)).sum().item()
    metrics['max_logit_delta'] = max(metrics['max_logit_delta'],
                                     (logits - reference).abs().max().item())


def assert_state_unchanged(model, before):
    import torch
    after = model.state_dict()
    if set(after) != set(before):
        raise RuntimeError('Model state keys changed during ablation')
    for name, tensor in after.items():
        if not torch.equal(tensor.detach().cpu(), before[name]):
            raise RuntimeError(f'Model state unexpectedly changed: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/nturgbd-cross-subject/joint_ssm_bidir_motion_20ep.yaml')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'work_dir/ntu60/xsub/full_ssm_bidir_motion_20ep/runs-best.pt')
    parser.add_argument('--max-batches', type=int, default=16, help='0 evaluates the entire test set')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.max_batches < 0 or args.batch_size < 1:
        parser.error('max-batches must be non-negative and batch-size positive')
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        import yaml
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; run inference on the existing AutoDL environment')
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
            raise ValueError('This tool expects the SSM-only experiment without CTR/FR')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        model = Model(**config['model_args']).cuda(args.device).float().eval()
        checkpoint = load_checkpoint_cpu(args.checkpoint)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
        del checkpoint, state
        modules = [getattr(model, f'l{number}').mamba.mamba for number in range(1, 11)]
        if not any(getattr(module, 'motion_delta', False) and getattr(module, 'bidirectional', False)
                   for module in modules):
            raise ValueError('No combined SSM modules found; check config and synchronized source')
        before = {key: value.detach().to('cpu', copy=True) for key, value in model.state_dict().items()}
        feeder = dict(config['test_feeder_args'], split='test', debug=False)
        feeder.update(random_choose=False, random_shift=False, random_move=False, random_rot=False)
        print('Loading test data; the same samples are reused for all six interventions.', flush=True)
        dataset = Feeder(**feeder)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batches = loader if args.max_batches == 0 else islice(loader, args.max_batches)
        totals = {name: dict(samples=0, correct=0, changed=0, max_logit_delta=0.) for name in VARIANTS}
        print(f'Inference only: max-batches={args.max_batches}, batch-size={args.batch_size}; no weight saves.', flush=True)
        with torch.no_grad():
            for batch, (inputs, labels, _) in enumerate(batches, 1):
                inputs, labels = inputs.cuda(args.device).float(), labels.cuda(args.device).long()
                one_hot = F.one_hot(labels, config['model_args']['num_class'])
                reference = None
                for variant in VARIANTS:
                    with ablation(modules, variant):
                        logits, _ = model(inputs, one_hot)
                    if reference is None:
                        reference = logits
                    record(totals[variant], logits, labels, reference)
                if batch == 1 or batch % 16 == 0:
                    print(f'  processed {totals["full"]["samples"]} test samples', flush=True)
        assert_state_unchanged(model, before)
        size = totals['full']['samples']
        if not size:
            raise RuntimeError('No test samples were available')
        scope = 'FULL TEST SET' if size == len(dataset) else 'SUBSET PREFIX (not benchmark accuracy)'
        print(f'Scope: {scope}; samples={size}/{len(dataset)}')
        print('variant\tsamples\ttop1_%\tdelta_vs_full_pp\tchanged_predictions\tmax_abs_logit_delta')
        base_accuracy = 100 * totals['full']['correct'] / size
        for variant, row in totals.items():
            accuracy = 100 * row['correct'] / row['samples']
            print(f'{variant}\t{row["samples"]}\t{accuracy:.4f}\t{accuracy-base_accuracy:+.4f}\t'
                  f'{row["changed"]}\t{row["max_logit_delta"]:.6e}')
        print('All state restored exactly; no files written. This tests one trained checkpoint, '
              'not separately trained A/B models or multi-seed stability. Do not compare variant speeds.')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
