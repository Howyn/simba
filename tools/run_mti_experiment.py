"""Run ONE explicitly selected MTI experiment; never chain the two stages."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import atomic_text


ENVIRONMENT_CHECK = """
from pathlib import Path
import re
import sys
import torch
import selective_scan_cuda as scan
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this Python environment')
if not hasattr(torch.amp, 'GradScaler'):
    raise SystemExit('This protocol requires PyTorch with torch.amp.GradScaler (2.3+)')
if Path(scan.__file__).resolve().parent != Path(sys.argv[1]).resolve():
    raise SystemExit('Build the repository baseline extension: python setup_baseline_scan.py build_ext --inplace')
arguments = [int(value) for value in re.findall(r'arg(\\d+):', scan.fwd.__doc__ or '')]
if arguments and max(arguments) != 8:
    raise SystemExit('selective_scan_cuda API mismatch: rebuild with setup_baseline_scan.py in this Python environment')
print('Runtime: torch=' + torch.__version__ + '; scan=' + scan.__file__, flush=True)
print('GPU: ' + torch.cuda.get_device_name(0), flush=True)
"""


@contextmanager
def run_lock(root):
    # AutoDL is Linux. Keep this descriptor in the launcher for the entire run.
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.launcher.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another experiment launcher is already running in this output root') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['baseline', 'mti'], required=True)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'work_dir/ntu60/xsub/mti_fixed15')
    parser.add_argument('--resume', action='store_true', help='resume this variant/seed from runs-latest.pt')
    args = parser.parse_args()
    name = 'baseline' if args.variant == 'baseline' else 'enabled'
    config_path = ROOT / f'config/nturgbd-cross-subject/joint_mti_{name}15.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['seed'] = args.seed
    output_root = args.output_root.resolve()
    work_dir = output_root / f'{args.variant}_seed{args.seed}'
    config['work_dir'] = str(work_dir)
    if args.data_path:
        for split in ('train', 'test'):
            config[f'{split}_feeder_args']['data_path'] = str(args.data_path.resolve())
    for split in ('train', 'test'):
        if not Path(config[f'{split}_feeder_args']['data_path']).is_file():
            parser.error('NTU60 NPZ not found; set --data-path to the same file for both variants')

    with run_lock(output_root):
        launch_config = work_dir / 'launch_config.yaml'
        checkpoint = work_dir / 'runs-latest.pt'
        if args.resume:
            if not checkpoint.is_file() or not launch_config.is_file():
                parser.error('--resume requires runs-latest.pt and launch_config.yaml for this run')
            saved_config = yaml.safe_load(launch_config.read_text(encoding='utf-8'))
            if config != saved_config:
                parser.error('resume settings differ from launch_config.yaml; restore the original settings')
        else:
            if work_dir.exists() and any(work_dir.iterdir()):
                parser.error(f'run directory already has files: {work_dir}; use --resume or a new output root')
            atomic_text(launch_config, yaml.safe_dump(config, sort_keys=False))

        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = 'unknown'
        command = [sys.executable, '-u', str(ROOT / 'main.py'), '--config', str(launch_config)]
        if args.resume:
            command.extend(['--resume', str(checkpoint)])
        status = {
            'variant': args.variant, 'seed': args.seed, 'revision': revision,
            'python': sys.executable, 'work_dir': str(work_dir),
            'started_unix': time.time(), 'status': 'running', 'launcher_pid': os.getpid(),
        }
        atomic_text(work_dir / 'launcher_status.json', json.dumps(status, indent=2) + '\n')
        print(f"Launching ONLY {args.variant}, seed={args.seed}; fixed LR=0.005; {config['precision']} AMP; 15 epochs.", flush=True)
        print(f'Python: {sys.executable}\nWork directory: {work_dir}', flush=True)
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        result = subprocess.run([sys.executable, '-c', ENVIRONMENT_CHECK, str(ROOT)],
                                cwd=ROOT, env=environment)
        if result.returncode == 0:
            result = subprocess.run(command, cwd=ROOT, env=environment)
        status.update(status='completed' if result.returncode == 0 else 'failed',
                      exit_code=result.returncode, finished_unix=time.time())
        atomic_text(work_dir / 'launcher_status.json', json.dumps(status, indent=2) + '\n')
        if result.returncode:
            print(f'EXPERIMENT FAILED, exit={result.returncode}; inspect the log before resuming.', flush=True)
        else:
            print('This variant finished. No other variant will be started automatically.', flush=True)
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
