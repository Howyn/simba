"""Inspect saved best-model predictions and SSM weights on CPU; never trains.

The training program's runs-best_right.txt contains ALL (prediction, label)
rows, despite its name. Matching label sequences assume the same test order.
"""
import argparse
from collections import defaultdict
import csv
import importlib
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_predictions(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for number, fields in enumerate(csv.reader(stream), 1):
            try:
                if len(fields) != 2:
                    raise ValueError()
                prediction, label = map(int, fields)
                if not (0 <= prediction < 60 and 0 <= label < 60):
                    raise ValueError()
            except ValueError:
                raise ValueError(f'{path}:{number}: expected prediction,label integers in 0..59') from None
            rows.append((prediction, label))
    if not rows:
        raise ValueError(f'{path}: no predictions')
    return rows


def compare_predictions(baseline, combined):
    if len(baseline) != len(combined):
        raise ValueError(f'Prediction counts differ: baseline={len(baseline)}, combined={len(combined)}')
    if not baseline:
        raise ValueError('No paired predictions')
    counts = dict(both_correct=0, both_wrong=0, gained=0, lost=0)
    classes = defaultdict(lambda: dict(samples=0, baseline_correct=0, combined_correct=0))
    for number, ((base, label), (combo, other_label)) in enumerate(zip(baseline, combined), 1):
        if label != other_label:
            raise ValueError(f'Test labels/order differ at row {number}: {label} vs {other_label}')
        a, b = base == label, combo == label
        counts['both_correct' if a and b else 'both_wrong' if not a and not b else 'gained' if b else 'lost'] += 1
        classes[label]['samples'] += 1
        classes[label]['baseline_correct'] += int(a)
        classes[label]['combined_correct'] += int(b)
    size = len(baseline)
    return dict(counts, samples=size, net_correct=counts['gained'] - counts['lost'],
                baseline_top1=100 * (counts['both_correct'] + counts['lost']) / size,
                combined_top1=100 * (counts['both_correct'] + counts['gained']) / size,
                delta_pp=100 * (counts['gained'] - counts['lost']) / size,
                classes=dict(classes))


def load_checkpoint_cpu(path):
    """Read tensor checkpoints, including legacy NumPy scalar metadata, safely."""
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        raise ValueError('This PyTorch version does not support safe checkpoint loading; '
                         'use --skip-checkpoint for prediction-only analysis.') from None
    except pickle.UnpicklingError as error:
        if 'numpy' not in str(error).lower():
            raise ValueError(f'{path}: safe checkpoint loading failed: {error}') from None
    try:
        import numpy as np
    except ImportError:
        raise ValueError('NumPy is needed to read this older checkpoint metadata; '
                         'use --skip-checkpoint for prediction-only analysis.') from None
    if not hasattr(torch.serialization, 'safe_globals'):
        raise ValueError('This PyTorch version cannot safely read NumPy checkpoint metadata; '
                         'use --skip-checkpoint for prediction-only analysis.')
    scalar_module = 'numpy._core.multiarray' if int(np.__version__.split('.')[0]) >= 2 else 'numpy.core.multiarray'
    scalar = importlib.import_module(scalar_module).scalar

    def scalar_alias(module):
        # NumPy 2 renamed core to _core. Local wrappers recognize either pickle
        # path on PyTorch 2.5 as well as newer versions without patching NumPy.
        def reconstruct(*args):
            return scalar(*args)
        reconstruct.__module__ = module
        reconstruct.__name__ = reconstruct.__qualname__ = 'scalar'
        return reconstruct

    allowed = [scalar_alias('numpy.core.multiarray'),
               scalar_alias('numpy._core.multiarray'), np.dtype]
    allowed.extend(type(np.dtype(name)) for name in ('float64', 'float32', 'int64', 'int32'))
    existing = torch.serialization.get_safe_globals()
    allowed = [item for item in allowed if item not in existing]
    try:
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(f'{path}: safe loading still failed after allowing numeric NumPy metadata: '
                         f'{error}') from None


def checkpoint_stats(path):
    """Lazy torch import: prediction comparison itself needs only stdlib."""
    import torch
    checkpoint = load_checkpoint_cpu(path)
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f'{path}: no model_state_dict')
    state = {name.removeprefix('module.'): value for name, value in state.items()}
    results = []
    for layer in (4, 7, 10):
        prefix = f'l{layer}.mamba.mamba.'
        values = {}
        for name in ('bidir_logit', 'motion_up.weight', 'motion_down.weight'):
            key = prefix + name
            if key not in state:
                raise ValueError(f'{path}: missing {key}; check the actual trained module placement')
            tensor = state[key].detach().cpu().float()
            if not tensor.numel() or not torch.isfinite(tensor).all():
                raise ValueError(f'{path}: empty/non-finite parameter {key}')
            values[name] = tensor
        gate = values['bidir_logit'].sigmoid()
        item = dict(layer=layer, gate_mean=gate.mean().item(),
                    gate_quantiles=torch.quantile(gate, torch.tensor([0., .25, .5, .75, 1.])).tolist())
        for name in ('motion_up.weight', 'motion_down.weight'):
            tensor = values[name]
            item[name] = dict(rms=tensor.double().square().mean().sqrt().item(),
                              nonzero=int(torch.count_nonzero(tensor)), elements=tensor.numel())
        results.append(item)
    return results


def print_comparison(result, top_classes):
    print(f'Paired test samples: {result["samples"]}')
    print(f'Baseline saved-best Top1: {result["baseline_top1"]:.4f}%')
    print(f'Combined saved-best Top1: {result["combined_top1"]:.4f}%')
    print(f'Difference: {result["delta_pp"]:+.4f} percentage points')
    print(f'Both correct: {result["both_correct"]}; both wrong: {result["both_wrong"]}')
    print(f'Baseline wrong -> combined correct: {result["gained"]}')
    print(f'Baseline correct -> combined wrong: {result["lost"]}')
    print(f'Net additional correct: {result["net_correct"]:+d}')
    ranked = []
    for label, row in result['classes'].items():
        delta = 100 * (row['combined_correct'] - row['baseline_correct']) / row['samples']
        ranked.append((label, delta, row))
    for title, sign in (('Improved classes', 1), ('Worsened classes', -1)):
        chosen = sorted((x for x in ranked if x[1] * sign > 0), key=lambda x: -sign * x[1])[:top_classes]
        if chosen:
            print(f'{title} (zero-based class IDs):')
            for label, delta, row in chosen:
                print(f'  class {label}: {delta:+.2f} pp; '
                      f'{row["baseline_correct"]}->{row["combined_correct"]} correct / {row["samples"]}')
    print('This is one paired test-set comparison, not evidence of multi-seed training stability.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parent = ROOT / 'work_dir/ntu60/xsub'
    parser.add_argument('--baseline-dir', type=Path, default=parent / 'full_baseline_20ep')
    parser.add_argument('--combined-dir', type=Path, default=parent / 'full_ssm_bidir_motion_20ep')
    parser.add_argument('--skip-checkpoint', action='store_true')
    parser.add_argument('--top-classes', type=int, default=3)
    args = parser.parse_args()
    if args.top_classes < 0:
        parser.error('--top-classes must be non-negative')
    try:
        baseline_path = args.baseline_dir / 'runs-best_right.txt'
        combined_path = args.combined_dir / 'runs-best_right.txt'
        print(f'Baseline: {baseline_path}\nCombined: {combined_path}')
        result = compare_predictions(load_predictions(baseline_path), load_predictions(combined_path))
        print_comparison(result, args.top_classes)
        print('Pairing assumes the same test dataset and sample order; labels alone cannot prove sample identity.')
        if not args.skip_checkpoint:
            path = args.combined_dir / 'runs-best.pt'
            if not path.is_file():
                print(f'Checkpoint inspection skipped: missing {path}')
                return 0
            try:
                rows = checkpoint_stats(path)
            except ImportError:
                print('Checkpoint inspection skipped: PyTorch is unavailable; prediction comparison is complete.')
                return 0
            print(f'CPU checkpoint inspection: {path}')
            for row in rows:
                quantiles = ','.join(f'{value:.5f}' for value in row['gate_quantiles'])
                print(f'l{row["layer"]} reverse gate: mean={row["gate_mean"]:.5f}, '
                      f'q0/25/50/75/100=[{quantiles}]; initial gate=0.10000')
                for name in ('motion_up.weight', 'motion_down.weight'):
                    data = row[name]
                    print(f'  {name}: rms={data["rms"]:.6e}, nonzero={data["nonzero"]}/{data["elements"]}')
            print('Gate changes may include weight-decay effects. Parameter magnitude/change alone does not '
                  'prove contribution or the size of the actual delta correction.')
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
