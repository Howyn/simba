"""Paired screening protocols and atomic result files (top-1 in percent)."""

import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile


SIGNATURE_KEYS = (
    'seed', 'model', 'model_args', 'feeder', 'train_feeder_args',
    'test_feeder_args', 'optimizer', 'base_lr', 'fixed_lr', 'lr_type',
    'momentum', 'nesterov', 'weight_decay', 'batch_size', 'test_batch_size',
    'precision', 'tf32', 'num_worker', 'persistent_workers', 'prefetch_factor', 'lr_decay_rate',
    'warm_up_epoch', 'step', 'device', 'num_epoch', 'ema', 'screening',
)


def training_signature(args):
    values = vars(args) if not isinstance(args, dict) else args
    # Make an independent snapshot using only checkpoint-safe built-in types.
    result = {key: values[key] for key in SIGNATURE_KEYS}
    # Old fixed15 checkpoints had no protocol field; retain their exact signature.
    if values.get('screening_protocol', 'fixed15') != 'fixed15':
        result['screening_protocol'] = values['screening_protocol']
    return json.loads(json.dumps(result))


def planned_learning_rate(epoch, base_lr, warmup_epochs, milestones, decay):
    if epoch < warmup_epochs:
        return float(base_lr * (epoch + 1) / warmup_epochs)
    return float(base_lr * decay ** sum(epoch >= milestone for milestone in milestones))


def validate_screening(args):
    budget = getattr(args, 'stop_after_epoch', None)
    if budget is not None and (not args.screening or not 1 <= budget <= args.num_epoch):
        raise ValueError('stop_after_epoch requires screening and must be between 1 and num_epoch')
    if not args.screening:
        return
    protocol = getattr(args, 'screening_protocol', 'fixed15')
    if protocol == 'short10':
        expected = dict(fixed_lr=False, warm_up_epoch=2, step=[6], base_lr=0.025,
                        lr_decay_rate=0.1, lr_type='multistep', num_epoch=10,
                        optimizer='SGD', batch_size=24, test_batch_size=64,
                        precision='bf16', tf32=True, num_worker=8)
        changed = [key for key, value in expected.items() if getattr(args, key, None) != value]
        if changed:
            raise ValueError('short10 protocol differs: ' + ', '.join(changed))
        if args.model_args.get('use_mti', False):
            raise ValueError('short10 BiMamba experiment must not enable MTI')
    elif protocol == 'fixed15':
        if not args.fixed_lr or args.warm_up_epoch != 0 or args.step:
            raise ValueError('screening requires fixed_lr=true, warm_up_epoch=0, step=[]')
    else:
        raise ValueError('unknown screening protocol: ' + str(protocol))
    if not math.isfinite(args.base_lr) or args.base_lr <= 0:
        raise ValueError('base_lr must be positive and finite')
    if not 1 <= args.num_epoch <= 15:
        raise ValueError('screening is capped at 15 epochs')
    if args.precision not in ('fp16', 'bf16'):
        raise ValueError('screening requires AMP (fp16 or bf16) with matching precision in both runs')
    if len(args.device) != 1 or args.ema:
        raise ValueError('screening requires one GPU and ema=false')
    if args.weights or (args.start_epoch != 0 and not args.resume):
        raise ValueError('start from epoch 0 without weights, or resume this exact run')
    if not args.save_latest or not args.strict_rng_resume or args.persistent_workers:
        raise ValueError('screening requires latest/RNG checkpoints and nonpersistent workers')


def check_resume_signature(saved, current):
    if saved is None:
        raise ValueError('screening cannot resume a legacy checkpoint without run_config')
    changed = [key for key in SIGNATURE_KEYS if saved.get(key) != current.get(key)]
    if saved.get('screening_protocol', 'fixed15') != current.get('screening_protocol', 'fixed15'):
        changed.append('screening_protocol')
    if changed:
        raise ValueError('resume configuration differs: ' + ', '.join(changed))


def stable_window(records, protocol='fixed15'):
    epochs = [8, 9, 10] if protocol == 'short10' else list(range(10, 16))
    selected = {int(row['epoch']): row for row in records if int(row['epoch']) in epochs}
    result = {
        'epochs': epochs,
        'tolerance_percentage_points': 0.2,
        'rule': 'max(abs(top1_percent - window_mean)) <= 0.2 percentage points',
        'top1_percent': [selected[e]['top1_percent'] for e in epochs if e in selected],
        'status': 'incomplete',
        'final_mean_top1_percent': None,
    }
    if set(selected) != set(epochs):
        return result
    values = [float(selected[e]['top1_percent']) for e in epochs]
    if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
        raise ValueError('top1_percent must contain finite values in [0, 100]')
    mean = sum(values) / len(values)
    deviation = max(abs(value - mean) for value in values)
    stable = deviation <= 0.2 + 1e-12
    result.update(
        status='stable' if stable else 'unstable',
        min_top1_percent=min(values), max_top1_percent=max(values),
        range_percentage_points=max(values) - min(values),
        max_deviation_percentage_points=deviation,
        final_mean_top1_percent=mean if stable else None,
    )
    return result


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_run_results(work_dir, records, run_config, *, completed=False, best_accuracy=None):
    if len({row['epoch'] for row in records}) != len(records):
        raise ValueError('duplicate epoch records')
    records = sorted(records, key=lambda row: row['epoch'])
    summary = {
        'completed': completed,
        'variant': ('bimamba' if run_config['model_args'].get('use_bimamba', False)
                    else 'mti' if run_config['model_args'].get('use_mti', False) else 'baseline'),
        'seed': run_config['seed'],
        'evaluation_split': 'official NTU60 X-Sub test split',
        'run_config': run_config,
        'epochs': records,
        'stability': stable_window(records, run_config.get('screening_protocol', 'fixed15')),
        'best_top1_percent': None if best_accuracy is None else 100 * float(best_accuracy),
        'final_checkpoint': str(Path(work_dir) / 'runs-final.pt') if completed else None,
    }
    atomic_text(Path(work_dir) / 'results.json', json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    if records:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
        atomic_text(Path(work_dir) / 'metrics.csv', stream.getvalue())
    return summary
