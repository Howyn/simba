"""Continue a running baseline to a budget, then run MTI to the same budget.

For the existing nohup queue, watch atomic results and stop ONLY its verified
training child after a completed epoch/checkpoint. Stop its verified old shell
queue first so it cannot separately start an unbudgeted MTI run.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.experiment_results import atomic_text


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def process_args(pid):
    try:
        return [value.decode(errors='replace') for value in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if value]
    except (FileNotFoundError, ProcessLookupError):
        return []


def parent_pid(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('PPid:'):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def verified_launcher(work_dir):
    status = read_json(work_dir / 'launcher_status.json')
    pid = status.get('launcher_pid')
    if not isinstance(pid, int) or pid <= 1:
        return None
    argv = process_args(pid)
    if (str(ROOT / 'tools/run_mti_experiment.py') not in argv
            and 'tools/run_mti_experiment.py' not in argv):
        return None
    if '--variant' not in argv or argv[argv.index('--variant') + 1] != 'baseline':
        return None
    if Path(status.get('work_dir', '')).resolve() != work_dir.resolve():
        raise RuntimeError('launcher metadata work directory differs; refusing to signal processes')
    return pid


def completed_epoch(work_dir):
    rows = read_json(work_dir / 'results.json').get('epochs', [])
    return max((int(row['epoch']) for row in rows), default=0)


def stop_verified_baseline(pid, work_dir):
    if verified_launcher(work_dir) != pid:
        raise RuntimeError('baseline launcher identity changed; refusing to signal')
    children = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or parent_pid(int(path.name)) != pid:
            continue
        argv = process_args(int(path.name))
        if str(ROOT / 'main.py') in argv and str(work_dir / 'launch_config.yaml') in argv:
            children.append((int(path.name), argv))
    if len(children) != 1:
        raise RuntimeError('cannot uniquely identify baseline training child; nothing was stopped')

    outer = parent_pid(pid)
    argv = process_args(outer) if outer else []
    if argv and Path(argv[0]).name in ('bash', 'sh') and '-c' in argv:
        script = argv[argv.index('-c') + 1]
        if '--variant baseline' in script and '--variant mti' in script:
            if 'tools/run_mti_experiment.py' not in script or work_dir.parent.name not in script:
                raise RuntimeError('old queue identity is ambiguous; nothing was stopped')
            os.kill(outer, signal.SIGTERM)
            print(f'Stopped verified old queue PID={outer}; it will not start a second MTI job.', flush=True)

    child, expected = children[0]
    if process_args(child) != expected:
        raise RuntimeError('training child identity changed; refusing to signal')
    os.kill(child, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while process_args(pid):
        if time.monotonic() > deadline:
            raise RuntimeError('old launcher did not exit; MTI was not started')
        time.sleep(1)
    atomic_text(work_dir / 'budget_stop.json', json.dumps({
        'reason': 'intentional epoch-budget stop after an atomic latest checkpoint',
        'completed_epoch': completed_epoch(work_dir), 'stopped_child_pid': child,
    }, indent=2) + '\n')


def launch(variant, root, seed, budget, resume=False):
    command = [sys.executable, '-u', str(ROOT / 'tools/run_mti_experiment.py'),
               '--variant', variant, '--output-root', str(root), '--seed', str(seed),
               '--stop-after-epoch', str(budget)]
    if resume:
        command.append('--resume')
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=4321)
    parser.add_argument('--stop-after-epoch', type=int, default=9)
    args = parser.parse_args()
    if not 1 <= args.stop_after_epoch <= 15:
        parser.error('--stop-after-epoch must be between 1 and 15')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / '.comparison.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('a comparison controller is already running here')
        baseline = root / f'baseline_seed{args.seed}'
        pid = verified_launcher(baseline)
        if pid:
            print(f'Continuing existing baseline PID={pid}; waiting for completed epoch {args.stop_after_epoch}.', flush=True)
            reported = None
            while completed_epoch(baseline) < args.stop_after_epoch:
                if verified_launcher(baseline) != pid:
                    if completed_epoch(baseline) >= args.stop_after_epoch:
                        break
                    raise RuntimeError('baseline exited before the requested budget; inspect its original log')
                done = completed_epoch(baseline)
                if done != reported:
                    print(f'BASELINE_PROGRESS completed={done}, budget={args.stop_after_epoch}', flush=True)
                    reported = done
                time.sleep(2)
            # results.json is written AFTER runs-latest.pt has been replaced.
            if verified_launcher(baseline) == pid:
                stop_verified_baseline(pid, baseline)
        elif completed_epoch(baseline) < args.stop_after_epoch:
            launch('baseline', root, args.seed, args.stop_after_epoch,
                   resume=(baseline / 'runs-latest.pt').is_file())

        print(f'BASELINE BUDGET READY: completed={completed_epoch(baseline)}; starting MTI.', flush=True)
        mti = root / f'mti_seed{args.seed}'
        if completed_epoch(mti) < args.stop_after_epoch:
            launch('mti', root, args.seed, args.stop_after_epoch, resume=(mti / 'runs-latest.pt').is_file())
        print(f'BOTH BUDGETS COMPLETE: compare epochs 1-{args.stop_after_epoch}; '
              'this does not establish stability over epochs 10-15.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f'COMPARISON STOPPED: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
