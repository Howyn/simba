"""Benchmark Simba on the real training feeder without writing experiment files.

Run from the repository root, for example:
  python tools/benchmark_4090.py ctr_fr --precision fp32 --tf32 false
  python tools/benchmark_4090.py ctr_fr --precision bf16 --tf32 false

Each invocation builds a fresh model and trains on a bounded number of
batches.  No checkpoint, metric log, score file, or dataset is written.
"""

import argparse
import importlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def import_class(path):
    module, name = path.rsplit('.', 1)
    return getattr(importlib.import_module(module), name)


def seed_worker(worker_id):
    # Match main.py's worker_init_fn, so augmentation uses the same seeding rule.
    seed = worker_id
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def positive_int(text):
    result = int(text)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def nonnegative_int(text):
    result = int(text)
    if result < 0:
        raise argparse.ArgumentTypeError('must be a non-negative integer')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('baseline', 'ctr', 'fr', 'ctr_fr',
                                        'ssm_bidir', 'ssm_motion', 'ssm_bidir_motion'))
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='fp32')
    parser.add_argument('--tf32', choices=('true', 'false'), default='false')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', type=Path, default=None,
                        help='optional path to an experiment YAML')
    parser.add_argument('--batch-size', type=positive_int, default=None,
                        help='default: batch_size from the experiment YAML')
    parser.add_argument('--num-workers', type=nonnegative_int, default=None,
                        help='default: num_worker from the experiment YAML')
    parser.add_argument('--prefetch-factor', type=positive_int, default=2,
                        help='batches per worker; ignored when num-workers is 0')
    parser.add_argument('--warmup-steps', type=nonnegative_int, default=5)
    parser.add_argument('--steps', type=positive_int, default=20,
                        help='number of measured batches after warmup')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires an available CUDA GPU.')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise SystemExit('--device must specify one CUDA device, e.g. cuda:0')
    # Existing Simba layers use the current default CUDA device in __init__.
    torch.cuda.set_device(device)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise SystemExit('This GPU/PyTorch setup does not support CUDA BF16.')

    config_path = args.config or ROOT / 'config' / 'nturgbd-cross-subject' / f'joint_{args.mode}_10ep.yaml'
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    seed = int(config.get('seed', 4321))
    batch_size = args.batch_size or int(config['batch_size'])
    num_workers = args.num_workers if args.num_workers is not None else int(config['num_worker'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    tf32 = args.tf32 == 'true'
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    feeder = import_class(config['feeder'])(**config['train_feeder_args'])
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(dataset=feeder, batch_size=batch_size, shuffle=True,
                       drop_last=True, pin_memory=True, num_workers=num_workers,
                       generator=generator, worker_init_fn=seed_worker)
    if num_workers > 0:
        loader_args['prefetch_factor'] = args.prefetch_factor
    loader = DataLoader(**loader_args)
    required = args.warmup_steps + args.steps
    if len(loader) < required:
        raise SystemExit(f'Training loader has {len(loader)} complete batches; '
                         f'{required} are required. Reduce --steps or --batch-size.')

    model = import_class(config['model'])(**config['model_args']).to(device)
    model.train()
    base_lr = float(config.get('base_lr', 0.025))
    warmup_epochs = int(config.get('warm_up_epoch', 0))
    # Match the training loop's epoch-one warmup learning rate.
    learning_rate = base_lr / warmup_epochs if warmup_epochs > 0 else base_lr
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=float(config.get('momentum', 0.9)),
                                nesterov=bool(config.get('nesterov', False)),
                                weight_decay=float(config.get('weight_decay', 0.0004)))
    num_classes = int(config['model_args']['num_class'])
    has_fr = getattr(model, 'fr_head', None) is not None
    iterator = iter(loader)

    def train_batch():
        data, labels, _ = next(iterator)
        data = data.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.precision == 'bf16', dtype=torch.bfloat16):
            logits, auxiliary = model(data, F.one_hot(labels, num_classes))
            loss = F.cross_entropy(logits, labels)
            if has_fr:
                loss = loss + auxiliary
        loss.backward()
        optimizer.step()
        return int(labels.shape[0]), loss.detach()

    print(f'GPU: {torch.cuda.get_device_name(device)} | PyTorch: {torch.__version__}', flush=True)
    print(f'mode={args.mode} precision={args.precision} tf32={args.tf32} '
          f'seed={seed} batch_size={batch_size} num_workers={num_workers} '
          f'prefetch_factor={args.prefetch_factor if num_workers else "disabled"} '
          f'warmup={args.warmup_steps} steps={args.steps}', flush=True)
    print(f'config={config_path}\ndata={config["train_feeder_args"]["data_path"]}', flush=True)
    print(f'optimizer=SGD learning_rate={learning_rate:g} '
          f'momentum={optimizer.param_groups[0]["momentum"]:g} '
          f'weight_decay={optimizer.param_groups[0]["weight_decay"]:g} '
          f'nesterov={optimizer.param_groups[0]["nesterov"]} '
          f'trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}', flush=True)
    for _ in range(args.warmup_steps):
        train_batch()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    samples = 0
    total_loss = torch.zeros((), device=device)
    for _ in range(args.steps):
        count, loss = train_batch()
        samples += count
        total_loss += loss.float()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    mean_loss = (total_loss / args.steps).item()
    if not np.isfinite(mean_loss):
        raise RuntimeError('Measured training loss is non-finite; do not use this precision setting.')
    print(f'measured_batches={args.steps} samples={samples} elapsed={seconds:.3f} s samples/s={samples / seconds:.2f} '
          f'mean_loss={mean_loss:.6f}')
    print(f'peak allocated={torch.cuda.max_memory_allocated(device) / 1024 ** 2:.0f} MiB '
          f'peak reserved={torch.cuda.max_memory_reserved(device) / 1024 ** 2:.0f} MiB')
    print('Timing includes data loading, transfer, forward, backward and optimizer step; '
          'warmup, initialization and evaluation are excluded.')


if __name__ == '__main__':
    main()
