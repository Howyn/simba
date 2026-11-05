"""Run only the requested short10 variant; reuse the verified previous baseline.

Default: combined Hyper-GCN + DeTGC. No baseline or ablation training is chained.
The opt-in AutoDL shutdown waits until this entire job exits, including failure.
"""
import argparse
import errno
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import atomic_text, check_resume_signature, stable_window, training_signature

BASELINE_REVISION = '282f095766ff9d60a86c9199fe19bf28e32122ab'
VARIANTS = {'combined': (True, True), 'spatial': (True, False), 'temporal': (False, True)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_records(result):
    rows = result.get('epochs', [])
    if not result.get('completed') or [r.get('epoch') for r in rows] != list(range(1, 11)):
        raise ValueError('a completed short10 run must have exactly epochs 1..10')
    signature = result.get('run_config', {})
    for row in rows:
        value = row.get('top1_percent', float('nan'))
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError('invalid top1 in saved results')
        epoch = row['epoch']
        expected = 0.0125 if epoch == 1 else 0.025 if epoch <= 6 else 0.0025
        if not math.isclose(row.get('learning_rate', -1), expected, rel_tol=1e-8):
            raise ValueError('saved learning-rate history differs from short10')
    return signature, rows


def validate_baseline(directory, expected_config):
    """Read-only reuse; never relabel old numbers or mix training recipes."""
    directory = directory.resolve()
    result = read_json(directory / 'results.json')
    signature, _ = checked_records(result)
    check_resume_signature(signature, training_signature(expected_config))
    revision = directory.parent / 'source_revision.txt'
    if not revision.is_file() or revision.read_text().strip() != BASELINE_REVISION:
        raise ValueError('baseline must be the verified 282f095 short10 run; '
                         'missing/different parent source_revision.txt')
    return result


def compare(root, variant, seed, baseline, improved):
    checked_records(baseline)
    checked_records(improved)
    a = stable_window(baseline['epochs'], 'short10')
    b = stable_window(improved['epochs'], 'short10')
    stable = a['status'] == b['status'] == 'stable'
    delta = b['final_mean_top1_percent'] - a['final_mean_top1_percent'] if stable else None
    verdict = ('initial_signal' if delta > 0 else 'no_signal') if stable else 'inconclusive_unstable'
    summary = dict(variant=variant, seed=seed, verdict=verdict, delta_percentage_points=delta,
                   baseline=a, improved=b, window=[8, 9, 10],
                   note='Single-seed screening on the official NTU test split; not final convergence evidence.')
    rows = {k: {r['epoch']: r for r in value['epochs']}
            for k, value in [('baseline', baseline), ('improved', improved)]}
    lines = ['# Combination1 short10 comparison', '',
             f'Variant: {variant}; seed: {seed}; baseline reused read-only.', '',
             '| Epoch | Baseline Top1 (%) | Improved Top1 (%) | Delta (pp) |', '|---|---:|---:|---:|']
    for epoch in range(1, 11):
        x, y = rows['baseline'][epoch]['top1_percent'], rows['improved'][epoch]['top1_percent']
        lines.append(f'| {epoch} | {x:.4f} | {y:.4f} | {y-x:+.4f} |')
    lines += ['', f'Baseline stability: {a["status"]}; improved stability: {b["status"]}.',
              f'Verdict: {verdict}.']
    if stable:
        lines.append(f'Stable means: {a["final_mean_top1_percent"]:.4f}% / '
                     f'{b["final_mean_top1_percent"]:.4f}%; delta={delta:+.4f} pp.')
    else:
        lines.append('No stable final mean: at least one run fails the +/-0.2 pp rule.')
    name = f'comparison_{variant}_seed{seed}'
    atomic_text(root / (name + '.json'), json.dumps(summary, indent=2) + '\n')
    atomic_text(root / (name + '.md'), '\n'.join(lines) + '\n')
    print(f'COMPARISON RESULT: {verdict}; {root / (name + ".md")}', flush=True)


def call_autodl_shutdown():
    program = Path('/usr/bin/shutdown')
    command_path = program.as_posix()
    try:
        return subprocess.run([command_path], timeout=30)
    except OSError as error:
        if error.errno != errno.ENOEXEC:
            raise
        # Some AutoDL images provide a shell wrapper without a shebang. Direct
        # execve fails although the documented shell invocation works. Do not
        # reinterpret an incompatible binary as shell source.
        prefix = program.read_bytes()[:4096]
        if not prefix or b'\x00' in prefix or prefix.startswith(b'\x7fELF'):
            raise RuntimeError('shutdown has an unsupported executable format') from error
        prefix.decode('utf-8')
        syntax = subprocess.run(['/bin/bash', '-n', command_path], timeout=10)
        if syntax.returncode:
            raise RuntimeError('shutdown is not a valid shell wrapper') from error
        print('AUTODL SHUTDOWN: ENOEXEC; invoking the text wrapper with /bin/bash', flush=True)
        return subprocess.run(['/bin/bash', command_path], timeout=30)


def wait_and_shutdown(root, delay, *, sleep=time.sleep, monotonic=time.monotonic):
    """AutoDL official /usr/bin/shutdown, invoked ONLY after the countdown.

    A persistent KEEP_RUNNING marker cancels both a pending and a future shutdown.
    Refuse shutdown if nvidia-smi fails or another GPU compute process appears.
    """
    marker = root / 'KEEP_RUNNING'
    deadline = monotonic() + delay
    atomic_text(root / 'shutdown_status.json', json.dumps(dict(
        status='waiting', delay_seconds=delay, scheduled_unix=time.time() + delay,
        cancel_file=str(marker)), indent=2) + '\n')
    print(f'AUTODL SHUTDOWN: in {delay} seconds (success or failure). Cancel: touch {marker}', flush=True)
    while monotonic() < deadline:
        if marker.exists():
            atomic_text(root / 'shutdown_status.json', '{"status":"cancelled_by_marker"}\n')
            print('AUTODL SHUTDOWN CANCELLED', flush=True)
            return
        sleep(min(1, max(0, deadline - monotonic())))
    if marker.exists():
        atomic_text(root / 'shutdown_status.json', '{"status":"cancelled_by_marker"}\n')
        return
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=15)
        if gpu.returncode or any(line.strip().isdigit() for line in gpu.stdout.splitlines()):
            raise RuntimeError('GPU idle check failed or another compute process is running')
        atomic_text(root / 'shutdown_status.json', '{"status":"calling_shutdown"}\n')
        print('AUTODL SHUTDOWN NOW', flush=True)
        if hasattr(os, 'sync'):
            os.sync()
        result = call_autodl_shutdown()
        if result.returncode:
            raise RuntimeError('AutoDL shutdown returned exit=' + str(result.returncode))
    except (OSError, RuntimeError, UnicodeError, subprocess.TimeoutExpired) as error:
        atomic_text(root / 'shutdown_status.json', json.dumps(dict(status='shutdown_not_executed_or_failed',
                                                                 error=str(error))) + '\n')
        print('AUTODL SHUTDOWN FAILED/SKIPPED: ' + str(error), flush=True)


def run(args):
    import yaml
    from tools.run_mti_experiment import ENVIRONMENT_CHECK, run_lock
    from tools.benchmark_autodl import check_idle_gpu
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.shutdown_on_exit and (sys.platform != 'linux' or not Path('/usr/bin/shutdown').is_file()):
        raise ValueError('--shutdown-on-exit requires AutoDL Linux /usr/bin/shutdown')
    with run_lock(root):
        check_idle_gpu(False)
        # Do not arm shutdown until this process has acquired its lock and idle GPU.
        try:
            config = yaml.safe_load((ROOT / 'config/nturgbd-cross-subject/joint_bimamba_baseline10.yaml').read_text())
            config['seed'] = args.seed
            if args.data_path:
                for split in ('train', 'test'):
                    config[f'{split}_feeder_args']['data_path'] = str(args.data_path.resolve())
            for split in ('train', 'test'):
                if not Path(config[f'{split}_feeder_args']['data_path']).is_file():
                    raise ValueError('NTU NPZ not found; use --data-path')
            baseline = validate_baseline(args.baseline_dir, config)
            print('BASELINE REUSED: ' + str(args.baseline_dir.resolve()), flush=True)
            spatial, temporal = VARIANTS[args.variant]
            config['model_args'].update(use_hypergcn=spatial, use_detgc=temporal)
            work = root / f'{args.variant}_seed{args.seed}'
            config['work_dir'] = str(work)
            checkpoint = work / 'runs-latest.pt'
            launch = work / 'launch_config.yaml'
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            saved_revision = root / 'source_revision.txt'
            if saved_revision.is_file() and saved_revision.read_text().strip() != revision:
                raise ValueError('source revision changed; restore the recorded commit before resuming')
            atomic_text(saved_revision, revision + '\n')
            if launch.is_file() and yaml.safe_load(launch.read_text()) != config:
                raise ValueError('saved launch config differs; use a new output root')
            if work.exists() and any(work.iterdir()) and not launch.is_file():
                raise ValueError('existing work directory has no verified launch config')
            result_path = work / 'results.json'
            if result_path.is_file():
                old = read_json(result_path)
                check_resume_signature(old['run_config'], training_signature(config))
                if old['completed']:
                    compare(root, args.variant, args.seed, baseline, old)
                    print('SKIP ALREADY COMPLETE', flush=True)
                    return 0
            if (work / 'runs').exists() and not checkpoint.is_file():
                raise ValueError('first epoch failed without checkpoint; use a new output root to retry')
            work.mkdir(parents=True, exist_ok=True)
            atomic_text(launch, yaml.safe_dump(config, sort_keys=False))
            atomic_text(root / 'baseline_reference.json', json.dumps(dict(
                path=str(args.baseline_dir.resolve()), revision=BASELINE_REVISION,
                run_config=baseline['run_config'], epochs=baseline['epochs']), indent=2) + '\n')
            subprocess.run([sys.executable, '-c', ENVIRONMENT_CHECK, str(ROOT)], cwd=ROOT, check=True)
            if not args.skip_preflight:
                print('PREFLIGHT: adapter reference/gradient checks and full combined BF16 backward.', flush=True)
                subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                '-p', 'test_combination1_modules.py', '-v'], cwd=ROOT, check=True)
            command = [sys.executable, '-u', str(ROOT / 'main.py'), '--config', str(launch)]
            if checkpoint.is_file():
                command += ['--resume', str(checkpoint)]
            status = dict(status='running', variant=args.variant, seed=args.seed, revision=revision,
                          work_dir=str(work), started_unix=time.time(), launcher_pid=os.getpid())
            atomic_text(work / 'launcher_status.json', json.dumps(status, indent=2) + '\n')
            print(f'START {args.variant}: short10, unchanged recipe; resume={checkpoint.is_file()}', flush=True)
            result = subprocess.run(command, cwd=ROOT)
            status.update(status='completed' if result.returncode == 0 else 'failed',
                          exit_code=result.returncode, finished_unix=time.time())
            atomic_text(work / 'launcher_status.json', json.dumps(status, indent=2) + '\n')
            if result.returncode:
                raise RuntimeError('training failed, exit=' + str(result.returncode))
            improved = read_json(result_path)
            compare(root, args.variant, args.seed, baseline, improved)
            print('COMBINATION1 COMPLETE; no further training is queued.', flush=True)
            return 0
        except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
            atomic_text(root / 'job_error.json', json.dumps(dict(error=str(error), unix=time.time())) + '\n')
            print('EXPERIMENT FAILED: ' + str(error)[:1200], flush=True)
            return 1
        finally:
            if args.shutdown_on_exit:
                wait_and_shutdown(root, args.shutdown_delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=tuple(VARIANTS), default='combined')
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'work_dir/ntu60/xsub/combination1_short10_v1')
    parser.add_argument('--baseline-dir', type=Path,
                        default=ROOT / 'work_dir/ntu60/xsub/bimamba_short10_v1/baseline_seed4321')
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--skip-preflight', action='store_true', help='only after this GPU environment passed preflight')
    parser.add_argument('--shutdown-on-exit', action='store_true')
    parser.add_argument('--shutdown-delay', type=int, default=300)
    args = parser.parse_args()
    if args.shutdown_delay < 0:
        parser.error('shutdown delay must be nonnegative')
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError) as error:
        print('LAUNCH REFUSED: ' + str(error), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
