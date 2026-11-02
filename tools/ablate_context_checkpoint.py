"""Inference-only functional ablation of one trained context/retain checkpoint.

Answers the pre-training question the design asks for: does the classification
actually depend on the modified state path, or only on the D skip and the
pointwise projections? Default: first 16 test batches. Use --max-batches 0 for
the full test set. No training, optimizer, output files or checkpoint updates.

    python tools/ablate_context_checkpoint.py \
        --config config/nturgbd-cross-subject/joint_algorithm_context_retain_write.yaml \
        --checkpoint work_dir/ntu60/xsub/algorithm_context_retain_write_seed4321/runs-best.pt
"""

import argparse
from contextlib import contextmanager, ExitStack
from itertools import islice
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'context_off', 'retain_off', 'both_off', 'state_readout_off')
# (variant, attribute, value while the variant is active)
TOGGLES = {
    'context_off': ('context_enabled', False),
    'retain_off': ('retain_write_enabled', False),
    'both_off': ('correction_enabled', False),
    'state_readout_off': ('state_enabled', False),
}
_MISSING = object()


def _restore_attribute(module, name, previous):
    if previous is _MISSING:
        delattr(module, name)
    else:
        setattr(module, name, previous)


@contextmanager
def ablation(modules, variant):
    """Apply one temporary intervention, restoring even when forward fails."""
    import torch
    if variant not in VARIANTS:
        raise ValueError(f'Unknown ablation: {variant}')
    with ExitStack() as changes:
        for module in modules:
            if variant == 'full':
                continue
            flag, value = TOGGLES[variant]
            previous = module.__dict__.get(flag, _MISSING)
            changes.callback(_restore_attribute, module, flag, previous)
            setattr(module, flag, value)
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
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'config/nturgbd-cross-subject/joint_algorithm_context_retain_write.yaml')
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'work_dir/ntu60/xsub/algorithm_context_retain_write_seed4321/runs-best.pt')
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
        model_args = config['model_args']
        if model_args.get('ctr_layers') or model_args.get('fr_layers'):
            raise ValueError('This tool expects the context/retain experiment without CTR/FR')
        if not (model_args.get('ssm_context_layers') or model_args.get('ssm_retain_layers')):
            raise ValueError('This config enables neither ssm_context_layers nor ssm_retain_layers')
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        model = Model(**model_args).cuda(args.device).float().eval()
        checkpoint = load_checkpoint_cpu(args.checkpoint)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()},
                              strict=True)
        del checkpoint, state
        layers = sorted(set(model_args.get('ssm_context_layers') or ())
                        | set(model_args.get('ssm_retain_layers') or ()))
        modules = [getattr(model, f'l{number}').mamba.mamba for number in layers]
        if not all(hasattr(module, 'probe_components') for module in modules):
            raise ValueError('No context/retain modules found; check config and synchronized source')
        before = {key: value.detach().to('cpu', copy=True) for key, value in model.state_dict().items()}
        feeder = dict(config['test_feeder_args'], split='test', debug=False)
        feeder.update(random_choose=False, random_shift=False, random_move=False, random_rot=False)
        print('Loading test data; the same samples are reused for all interventions.', flush=True)
        dataset = Feeder(**feeder)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0)
        batches = loader if args.max_batches == 0 else islice(loader, args.max_batches)
        totals = {name: dict(samples=0, correct=0, changed=0, max_logit_delta=0.)
                  for name in VARIANTS}
        print(f'Inference only: max-batches={args.max_batches}, batch-size={args.batch_size}; '
              'no weight saves.', flush=True)
        with torch.no_grad():
            for batch, (inputs, labels, _) in enumerate(batches, 1):
                inputs, labels = inputs.cuda(args.device).float(), labels.cuda(args.device).long()
                one_hot = F.one_hot(labels, model_args['num_class'])
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
        print(f'Scope: {scope}; samples={size}/{len(dataset)}; modified layers={layers}')
        print('variant\tsamples\ttop1_%\tdelta_vs_full_pp\tchanged_predictions\tmax_abs_logit_delta')
        base_accuracy = 100 * totals['full']['correct'] / size
        for variant, row in totals.items():
            accuracy = 100 * row['correct'] / row['samples']
            print(f'{variant}\t{row["samples"]}\t{accuracy:.4f}\t{accuracy-base_accuracy:+.4f}\t'
                  f'{row["changed"]}\t{row["max_logit_delta"]:.6e}')
        print('All state restored exactly; no files written. "both_off" keeps the explicit scan '
              'wiring but removes the correction, so its row also measures fused-versus-explicit '
              'rounding. Ablating one trained checkpoint is not the same as separately trained '
              'models and does not establish a gain.')
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
