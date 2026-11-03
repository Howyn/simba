"""Run a paired, five-epoch fine-tuning screen from completed epoch 20."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['both', 'baseline', 'negative'], default='both')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'work_dir/ntu60/xsub')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'work_dir/ntu60/xsub/short_convergence')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading checkpoints or training')
    args = parser.parse_args()
    modes = ['baseline', 'negative'] if args.mode == 'both' else [args.mode]
    jobs = []
    for mode in modes:
        checkpoint = args.source_root.resolve() / f'negative_eigen_{mode}/runs-latest.pt'
        output = args.output_root.resolve() / mode
        if output.exists():
            raise SystemExit(f'Refusing to overwrite existing output: {output}')
        command = [sys.executable, str(ROOT / 'main.py'),
                   '--config', str(ROOT / f'config/nturgbd-cross-subject/joint_negative_{mode}.yaml'),
                   '--work-dir', str(output), '--resume', str(checkpoint),
                   '--strict-rng-resume', 'True', '--num-epoch', '25',
                   '--step', '20', '--base-lr', '0.025', '--lr-decay-rate', '0.1',
                   '--precision', 'fp32', '--batch-size', '16', '--test-batch-size', '16',
                   '--persistent-workers', 'False', '--save-latest', 'True']
        jobs.append((checkpoint, command))
    if not args.dry_run:
        import torch
        # Validate every requested source before starting either training job.
        for checkpoint, _ in jobs:
            if not checkpoint.is_file():
                raise SystemExit(f'Missing checkpoint: {checkpoint}')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved.get('epoch') != 19:
                raise SystemExit(f'Expected completed epoch 20 (epoch=19): {checkpoint}')
            if any(key not in saved for key in ('model_state_dict', 'optimizer_state_dict', 'rng_state')):
                raise SystemExit(f'Checkpoint lacks model, optimizer or RNG state: {checkpoint}')
            del saved
        if 'negative' in modes:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                            '-p', 'test_negative_scan.py', '-v'], cwd=ROOT, check=True)
    for _, command in jobs:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
