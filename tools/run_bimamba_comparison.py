"""Locked short10 protocol: baseline first, then only Vim v2 BiMamba; auto-resume."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import atomic_text, check_resume_signature, training_signature


def compare(root, seed):
    results = {}
    for variant in ('baseline', 'bimamba'):
        path = root / f'{variant}_seed{seed}' / 'results.json'
        if not path.is_file():
            return
        results[variant] = json.loads(path.read_text(encoding='utf-8'))
    if not all(item['completed'] for item in results.values()):
        return
    baseline, improved = results['baseline'], results['bimamba']
    a, b = baseline['stability'], improved['stability']
    stable = a['status'] == b['status'] == 'stable'
    delta = b['final_mean_top1_percent'] - a['final_mean_top1_percent'] if stable else None
    verdict = ('initial_signal' if delta > 0 else 'no_signal') if stable else 'inconclusive_unstable'
    summary = dict(seed=seed, protocol='short10', window=[8, 9, 10], verdict=verdict,
                   delta_percentage_points=delta, baseline=a, bimamba=b,
                   note='Single-seed short-budget screening; not proof of a converged improvement.')
    atomic_text(root / 'comparison.json', json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    atomic_text(root / f'comparison_seed{seed}.json', json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    lines = ['# 10 epoch paired comparison', '', '| Epoch | Baseline Top-1 (%) | BiMamba Top-1 (%) | Delta (pp) |',
             '|---|---:|---:|---:|']
    rows = {name: {r['epoch']: r for r in result['epochs']} for name, result in results.items()}
    for epoch in (8, 9, 10):
        x, y = rows['baseline'][epoch]['top1_percent'], rows['bimamba'][epoch]['top1_percent']
        lines.append(f'| {epoch} | {x:.4f} | {y:.4f} | {y-x:+.4f} |')
    lines += ['', f'Baseline stability: {a["status"]}; BiMamba stability: {b["status"]}.',
              f'Verdict: {verdict}.']
    if stable:
        lines += [f'Stable-window means: baseline={a["final_mean_top1_percent"]:.4f}%, '
                  f'BiMamba={b["final_mean_top1_percent"]:.4f}%, delta={delta:+.4f} pp.']
    else:
        lines += ['At least one run fails the +/-0.2 pp rule; no stable final mean is reported.',
                  f'Baseline range: {a["min_top1_percent"]:.4f}–{a["max_top1_percent"]:.4f}%; '
                  f'BiMamba range: {b["min_top1_percent"]:.4f}–{b["max_top1_percent"]:.4f}%.']
    atomic_text(root / 'comparison.md', '\n'.join(lines) + '\n')
    atomic_text(root / f'comparison_seed{seed}.md', '\n'.join(lines) + '\n')
    print('COMPARISON RESULT: ' + verdict + '; ' + str(root / 'comparison.md'), flush=True)


def run_variant(variant, root, seed, data_path):
    import yaml
    filename = 'baseline' if variant == 'baseline' else 'enabled'
    config = yaml.safe_load((ROOT / f'config/nturgbd-cross-subject/joint_bimamba_{filename}10.yaml').read_text())
    config['seed'] = seed
    work = root / f'{variant}_seed{seed}'
    config['work_dir'] = str(work)
    if data_path:
        for split in ('train', 'test'):
            config[f'{split}_feeder_args']['data_path'] = str(data_path.resolve())
    for split in ('train', 'test'):
        if not Path(config[f'{split}_feeder_args']['data_path']).is_file():
            raise ValueError('NTU NPZ not found; set --data-path')
    launch = work / 'launch_config.yaml'
    checkpoint = work / 'runs-latest.pt'
    if launch.is_file():
        if yaml.safe_load(launch.read_text()) != config:
            raise ValueError('saved launch configuration differs; use a new output root')
    elif work.exists() and any(work.iterdir()):
        raise ValueError('existing directory has no verified launch configuration: ' + str(work))
    result_path = work / 'results.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        check_resume_signature(result['run_config'], training_signature(config))
        if result['completed']:
            if [r['epoch'] for r in result['epochs']] != list(range(1, 11)):
                raise ValueError('completed run has incomplete epoch records')
            print('SKIP already complete: ' + variant, flush=True)
            return
    if work.exists() and (work / 'runs').exists() and not checkpoint.is_file():
        raise ValueError('no completed checkpoint in ' + str(work)
                         + '; choose a new output root to retry the failed first epoch')
    work.mkdir(parents=True, exist_ok=True)
    atomic_text(launch, yaml.safe_dump(config, sort_keys=False))
    status = dict(variant=variant, seed=seed, status='running', started_unix=time.time())
    atomic_text(work / 'launcher_status.json', json.dumps(status, indent=2) + '\n')
    command = [sys.executable, '-u', str(ROOT / 'main.py'), '--config', str(launch)]
    if checkpoint.is_file():
        command += ['--resume', str(checkpoint)]
    print(f'START {variant}: 10 epochs; batch=24/64; BF16+TF32; '
          f'LR=0.0125,0.025 x5,0.0025 x4; resume={checkpoint.is_file()}', flush=True)
    result = subprocess.run(command, cwd=ROOT)
    status.update(status='completed' if result.returncode == 0 else 'failed',
                  exit_code=result.returncode, finished_unix=time.time())
    atomic_text(work / 'launcher_status.json', json.dumps(status, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f'{variant} failed (exit={result.returncode}); the next variant was not started')
    if not result_path.is_file() or not json.loads(result_path.read_text(encoding='utf-8'))['completed']:
        raise RuntimeError(variant + ' exited without a completed result summary; inspect its log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'work_dir/ntu60/xsub/bimamba_short10_v1')
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--variant', choices=('both', 'baseline', 'bimamba'), default='both')
    parser.add_argument('--skip-preflight', action='store_true', help='only for resuming an environment already checked')
    args = parser.parse_args()
    from tools.run_mti_experiment import ENVIRONMENT_CHECK, run_lock
    from tools.benchmark_autodl import check_idle_gpu
    root = args.output_root.resolve()
    with run_lock(root):
        check_idle_gpu(False)
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            metadata = root / 'source_revision.txt'
            if metadata.is_file() and metadata.read_text().strip() != revision:
                raise ValueError('code revision differs from the original run; restore it or use a new output root')
            atomic_text(metadata, revision + '\n')
            subprocess.run([sys.executable, '-c', ENVIRONMENT_CHECK, str(ROOT)], cwd=ROOT, check=True)
            if not args.skip_preflight:
                print('PREFLIGHT: fused/reference gradients, baseline parity, full batch24 BF16 backward.', flush=True)
                subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                                '-p', 'test_bimamba.py', '-v'], cwd=ROOT, check=True)
            variants = ('baseline', 'bimamba') if args.variant == 'both' else (args.variant,)
            for variant in variants:
                run_variant(variant, root, args.seed, args.data_path)
            compare(root, args.seed)
            print('BOTH 10 EPOCH RUNS COMPLETE' if args.variant == 'both' else 'VARIANT COMPLETE', flush=True)
            return 0
        except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            print('EXPERIMENT FAILED: ' + str(error)[:1200], flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
