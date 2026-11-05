"""Short real-data CUDA throughput benchmark; never launches an epoch or saves weights.

Each case runs in a separate process. The actual Processor.train/eval methods
are used, so finite-loss checks, GradScaler, FP32 evaluation and data transforms
match main.py. Warm-up batches are excluded from timing. Accuracy is meaningless
here: models are freshly initialized and their temporary updates are discarded.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import atomic_text


def make_cases(include_batch32=False, include_tf32=True):
    cases = [
        dict(name='reference_b16_eval16', batch_size=16, test_batch_size=16, tf32=False),
        dict(name='b16_eval64', batch_size=16, test_batch_size=64, tf32=False),
        dict(name='b24_eval64', batch_size=24, test_batch_size=64, tf32=False),
    ]
    if include_tf32:
        cases.append(dict(name='b24_eval64_tf32', batch_size=24, test_batch_size=64, tf32=True))
    if include_batch32:
        cases.append(dict(name='b32_eval64', batch_size=32, test_batch_size=64, tf32=False))
    return cases


def summarize(rows):
    valid = [row for row in rows if row.get('status') == 'ok']
    reference = next((row for row in valid if row['name'] == 'reference_b16_eval16'), None)
    best = min(valid, key=lambda row: row['estimated_epoch_seconds']) if valid else None
    for row in valid:
        row['estimated_time_saved_percent'] = (
            round(100 * (1 - row['estimated_epoch_seconds'] / reference['estimated_epoch_seconds']), 6)
            if reference else None
        )
    return dict(
        cases=rows, reference=reference['name'] if reference else None,
        fastest=best['name'] if best else None,
        target_20_percent_met=bool(reference and best and best['estimated_time_saved_percent'] >= 20),
        estimate_note='Steady-state real-data throughput projection; excludes epoch startup, checkpoint I/O and initialization. Confirm with one full epoch before adopting.',
        accuracy_note='No accuracy comparison: temporary benchmark models are discarded.',
    )


def write_reports(directory, rows):
    summary = summarize(rows)
    atomic_text(directory / 'results.json', json.dumps(summary, indent=2) + '\n')
    fields = ['name', 'status', 'batch_size', 'test_batch_size', 'tf32',
              'train_samples_per_second', 'eval_samples_per_second', 'train_peak_mib',
              'eval_peak_mib', 'estimated_epoch_seconds', 'estimated_time_saved_percent']
    import io
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(directory / 'results.csv', stream.getvalue())
    lines = ['# AutoDL speed benchmark', '', summary['estimate_note'], '', summary['accuracy_note'], '',
             '| Case | Status | Train samples/s | Eval samples/s | Train MiB | Estimated min/epoch | Time saved |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['status'] != 'ok':
            lines.append(f"| {row['name']} | {row['status']} | | | | | |")
            continue
        saved = row['estimated_time_saved_percent']
        saved_text = f'{saved:+.1f}%' if saved is not None else 'n/a'
        lines.append(f"| {row['name']} | ok | {row['train_samples_per_second']:.1f} | "
                     f"{row['eval_samples_per_second']:.1f} | {row['train_peak_mib']:.0f} | "
                     f"{row['estimated_epoch_seconds']/60:.2f} | {saved_text} |")
    lines += ['', 'Fastest measured candidate: ' + str(summary['fastest']), '',
              'Changing training batch size changes optimizer-update count and BatchNorm statistics. '
              'Use the same new settings for both variants, from scratch; do not compare against old runs. '
              'TF32 changes FP32 arithmetic. This file is not a training configuration.']
    atomic_text(directory / 'summary.md', '\n'.join(lines) + '\n')
    if summary['fastest']:
        best = next(row for row in rows if row['name'] == summary['fastest'])
        atomic_text(directory / 'recommended_cli.txt',
                    f"--batch-size {best['batch_size']} --test-batch-size {best['test_batch_size']} "
                    f"--precision bf16 --tf32 {str(best['tf32']).lower()}\n")
    return summary


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


class LabelWindow:
    """The real evaluation labels matching a contiguous window of loader batches."""
    def __init__(self, dataset, start, count):
        self.label = dataset.label[start:start + count]
        self.sample_name = dataset.sample_name[start:start + count]

    def top_k(self, scores, k):
        if len(scores) != len(self.label):
            raise ValueError('benchmark scores and labels are misaligned')
        ranks = scores.argsort()
        return sum(label in ranks[i, -k:] for i, label in enumerate(self.label)) / len(self.label)


class LoaderWindow:
    """Consume the existing iterator so worker startup and warm-up are not re-timed."""
    def __init__(self, iterator, steps, dataset):
        self.iterator, self.steps, self.dataset = iterator, steps, dataset

    def __len__(self):
        return self.steps

    def __iter__(self):
        return itertools.islice(self.iterator, self.steps)


def run_worker(plan, case, directory):
    import torch
    import main as training

    config = dict(plan['config'])
    parser = training.get_parser()
    parser.set_defaults(**config)
    args = parser.parse_args([])
    # No Processor.__init__: benchmark runs must not touch experiment folders,
    # resume weights, create TensorBoard writers, or invoke training start().
    args.work_dir = str(directory)
    args.model_saved_name = str(directory / 'unused')
    args.weights = args.resume = None
    args.start_epoch = 0
    args.screening = False
    args.stop_after_epoch = None
    args.ema = False
    args.precision = 'bf16'
    args.fixed_lr = True
    args.batch_size = case['batch_size']
    args.test_batch_size = case['test_batch_size']
    args.tf32 = case['tf32']
    args.persistent_workers = False
    args.device = [0]
    args.num_worker = plan['workers']
    args.log_interval = 100
    training.arg = args  # Legacy eval() reads the module-level argument too.
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required; use the working AutoDL .venv')
    training.init_seed(args.seed)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    processor = object.__new__(training.Processor)
    processor.arg = args
    processor.output_device = 0
    processor.print_log = lambda *a, **kw: None
    processor.train_writer = processor.val_writer = NullWriter()
    processor.global_step = 0
    processor.lr = args.base_lr
    processor.epoch_records = []
    processor.run_config = {}
    processor.best_acc, processor.best_acc_epoch = -1, 0
    processor.best_acc_ema = 0
    model_class = training.import_class(args.model)
    processor.model = model_class(**args.model_args).cuda(0)
    processor.loss = torch.nn.CrossEntropyLoss().cuda(0)
    processor.load_optimizer()
    processor.scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1.0)
    processor.load_data()
    train_loader = processor.data_loader['train']
    test_loader = processor.data_loader['test']
    if plan['warmup'] + plan['steps'] > len(train_loader):
        raise ValueError('training dataset has too few batches for this benchmark')
    if plan['eval_warmup'] + plan['eval_steps'] > len(test_loader):
        raise ValueError('test dataset has too few batches for this benchmark')

    print('TRAIN_WARMUP', flush=True)
    train_iterator = iter(train_loader)
    processor.data_loader['train'] = LoaderWindow(train_iterator, plan['warmup'], train_loader.dataset)
    processor.train(0, save_model=False)
    print('TRAIN_MEASURE', flush=True)
    processor.data_loader['train'] = LoaderWindow(train_iterator, plan['steps'], train_loader.dataset)
    processor.train(0, save_model=False)
    train = processor.last_train_metrics.copy()
    if not math.isfinite(train['train_loss']):
        raise RuntimeError('non-finite measured training loss')
    del train_iterator
    processor.data_loader['train'] = train_loader
    processor.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    print('EVAL_WARMUP', flush=True)
    test_iterator = iter(test_loader)
    warm_samples = plan['eval_warmup'] * args.test_batch_size
    measured_samples = plan['eval_steps'] * args.test_batch_size
    processor.data_loader['test'] = LoaderWindow(
        test_iterator, plan['eval_warmup'], LabelWindow(test_loader.dataset, 0, warm_samples))
    processor.eval(0, save_score=False)
    print('EVAL_MEASURE', flush=True)
    processor.data_loader['test'] = LoaderWindow(
        test_iterator, plan['eval_steps'], LabelWindow(test_loader.dataset, warm_samples, measured_samples))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    processor.eval(0, save_score=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    eval_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    if not math.isfinite(processor.last_eval_metrics['test_loss']):
        raise RuntimeError('non-finite measured evaluation loss')
    train_count = len(train_loader) * args.batch_size  # Actual drop_last=True behavior.
    eval_count = len(test_loader.dataset)
    eval_rate = measured_samples / elapsed
    result = dict(
        **case, status='ok', train_samples_per_second=train['samples_per_second'],
        eval_samples_per_second=eval_rate, train_peak_mib=train['peak_cuda_mib'], eval_peak_mib=eval_peak,
        measured_train_seconds=train['train_seconds'], measured_eval_seconds=elapsed,
        train_samples_per_epoch=train_count, eval_samples_per_epoch=eval_count,
        estimated_epoch_seconds=train_count / train['samples_per_second'] + eval_count / eval_rate,
        forward_dtype=train['forward_dtype'], grad_scaler_enabled=processor.scaler.is_enabled(),
        finite_loss=True, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
    )
    if train['forward_dtype'] != 'torch.bfloat16':
        raise RuntimeError('expected BF16 forward output, found ' + train['forward_dtype'])
    atomic_text(directory / 'result.json', json.dumps(result, indent=2) + '\n')


def worker_main(plan_path, case_name):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    case = next(item for item in plan['cases'] if item['name'] == case_name)
    directory = plan_path.parent / case_name
    directory.mkdir(exist_ok=True)
    try:
        run_worker(plan, case, directory)
        return 0
    except Exception as error:
        # CUDA binding TypeErrors can include whole tensors; never dump those.
        message = str(error).split('Invoked with:')[0][:2500]
        is_oom = type(error).__name__ == 'OutOfMemoryError' or 'out of memory' in message.lower()
        row = dict(**case, status='oom' if is_oom else 'failed', error_type=type(error).__name__, error=message)
        atomic_text(directory / 'result.json', json.dumps(row, indent=2) + '\n')
        print(''.join(traceback.format_list(traceback.extract_tb(error.__traceback__))), flush=True)
        print(f"BENCHMARK_CASE_FAILED {type(error).__name__}: {message}", flush=True)
        return 2 if is_oom else 1


def check_idle_gpu(allow_busy):
    if allow_busy:
        return
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return  # Runtime preflight will still check CUDA.
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    if result.returncode == 0 and pids:
        raise RuntimeError('GPU compute processes are already running (PIDs: ' + ', '.join(pids)
                           + '). Finish those runs before benchmarking, or explicitly use --allow-busy.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/nturgbd-cross-subject/joint_mti_baseline15.yaml')
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--steps', type=int, default=80, help='measured training batches per case')
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--eval-steps', type=int, default=20)
    parser.add_argument('--eval-warmup', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--include-batch32', action='store_true', help='optional higher-memory case; OOM is recorded and skipped')
    parser.add_argument('--no-tf32-case', action='store_true')
    parser.add_argument('--allow-busy', action='store_true')
    parser.add_argument('--resume', action='store_true', help='skip completed cases in an existing benchmark directory')
    parser.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-case', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan:
        return worker_main(args.worker_plan, args.worker_case)
    if min(args.steps, args.warmup, args.eval_steps, args.eval_warmup) < 1 or args.workers < 0:
        parser.error('batch counts must be positive; workers must be nonnegative')
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.data_path:
        for split in ('train', 'test'):
            config[f'{split}_feeder_args']['data_path'] = str(args.data_path.resolve())
    for split in ('train', 'test'):
        path = Path(config[f'{split}_feeder_args']['data_path']).resolve()
        if not path.is_file():
            parser.error(f'NTU NPZ not found: {path}; set --data-path')
        config[f'{split}_feeder_args']['data_path'] = str(path)
    if config['model_args'].get('use_mti', False):
        parser.error('benchmark the baseline first; use_mti must be false')
    if config['model'] != 'model.Hyperformer.Model' or config.get('optimizer', 'SGD') != 'SGD':
        parser.error('this benchmark targets the current Simba baseline with SGD')
    directory = (args.output_dir or ROOT / 'work_dir' / ('speed_' + time.strftime('%Y%m%d_%H%M%S'))).resolve()
    plan = dict(config=config, steps=args.steps, warmup=args.warmup, eval_steps=args.eval_steps,
                eval_warmup=args.eval_warmup, workers=args.workers,
                cases=make_cases(args.include_batch32, not args.no_tf32_case))
    plan_path = directory / 'plan.json'
    if args.resume:
        if not plan_path.is_file() or json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            parser.error('--resume requires the same benchmark plan and output directory')
    elif directory.exists() and any(directory.iterdir()):
        parser.error('output directory is not empty; use --resume or a new directory')
    check_idle_gpu(args.allow_busy)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_text(plan_path, json.dumps(plan, indent=2) + '\n')
    from tools.run_mti_experiment import ENVIRONMENT_CHECK
    subprocess.run([sys.executable, '-c', ENVIRONMENT_CHECK, str(ROOT)], cwd=ROOT, check=True)
    print(f'BENCHMARK START: {len(plan["cases"])} cases; results: {directory}', flush=True)
    print('Real train/eval loops, BF16 training + FP32 evaluation; no checkpoints or accuracy experiment.', flush=True)
    rows = []
    for case in plan['cases']:
        result_path = directory / case['name'] / 'result.json'
        if args.resume and result_path.is_file():
            previous = json.loads(result_path.read_text(encoding='utf-8'))
            if previous['status'] in ('ok', 'oom'):
                rows.append(previous)
                write_reports(directory, rows)
                print('SKIP completed: ' + case['name'], flush=True)
                continue
        print('CASE START: ' + case['name'], flush=True)
        (directory / case['name']).mkdir(exist_ok=True)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-plan', str(plan_path),
                   '--worker-case', case['name']]
        with (directory / case['name'] / 'worker.log').open('w', encoding='utf-8') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result_path.is_file():
            row = json.loads(result_path.read_text(encoding='utf-8'))
            if process.returncode and row.get('status') == 'ok':
                row.update(status='failed', error=f'worker exited {process.returncode} after writing its result')
        else:
            row = dict(**case, status='failed', error=f'worker exited {process.returncode}; see worker.log (possible host RAM exhaustion)')
        rows.append(row)
        summary = write_reports(directory, rows)
        if row['status'] == 'ok':
            saved = row['estimated_time_saved_percent']
            saved_text = f'{saved:+.1f}%' if saved is not None else 'n/a'
            print(f"CASE RESULT: {case['name']} train={row['train_samples_per_second']:.1f} samples/s "
                  f"peak={row['train_peak_mib']:.0f} MiB estimated_epoch={row['estimated_epoch_seconds']/60:.2f} min "
                  f"time_saved={saved_text}", flush=True)
        else:
            print(f"CASE RESULT: {case['name']} {row['status']}: {row.get('error', '')[:600]}", flush=True)
            if row['status'] != 'oom' or case['name'] == 'reference_b16_eval16':
                print('BENCHMARK STOPPED: fix the runtime error, then rerun with --resume.', flush=True)
                return 1
    summary = write_reports(directory, rows)
    print('BENCHMARK COMPLETE: fastest=' + str(summary['fastest'])
          + '; estimated_20_percent_target=' + str(summary['target_20_percent_met']), flush=True)
    print('SUMMARY: ' + str(directory / 'summary.md'), flush=True)
    print('NEXT: confirm the candidate with one full epoch; do not treat a short projection as measured epoch time.', flush=True)
    return 0 if summary['reference'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
