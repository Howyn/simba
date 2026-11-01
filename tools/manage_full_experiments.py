"""Run and inspect detached experiment suites on AutoDL Linux.

Only the standard library is imported. Read-only commands never create files.
The launcher uses the current Python environment and a detached session.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SUITE = 'ssm20'
MODES = ('baseline', 'ssm_bidir', 'ssm_motion', 'ssm_bidir_motion')
EPOCHS = 20
QUEUE_REL = Path('work_dir/ntu60/xsub/ssm_full_20ep_queue')


def configure_suite(name):
    """Select a profile before building CLI choices or spawning the worker."""
    global SUITE, MODES, EPOCHS, QUEUE_REL
    if name == 'bc10':
        MODES = ('ssm_bc_norm',)
        EPOCHS = 10
        QUEUE_REL = Path('work_dir/ntu60/xsub/ssm_bc_norm_10ep_queue')
    elif name == 'ssm20':
        MODES = ('baseline', 'ssm_bidir', 'ssm_motion', 'ssm_bidir_motion')
        EPOCHS = 20
        QUEUE_REL = Path('work_dir/ntu60/xsub/ssm_full_20ep_queue')
    else:
        raise ValueError(f'Unknown experiment suite: {name}')
    SUITE = name


def group_dir(root, mode):
    return root / 'work_dir' / 'ntu60' / 'xsub' / f'full_{mode}_{EPOCHS}ep'


def config_path(root, mode):
    return root / 'config' / 'nturgbd-cross-subject' / f'joint_{mode}_{EPOCHS}ep.yaml'


def read_state(queue):
    path = queue / 'state.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def write_state(queue, **state):
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = queue / 'state.json.tmp'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(queue / 'state.json')


def process_start(pid):
    try:
        # The comm field can itself contain spaces or parentheses.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def manager_alive(state):
    pid = state.get('manager_pid')
    return bool(pid and state.get('manager_start') and
                process_start(pid) == state['manager_start'])


def training_processes(root):
    """Find Python main.py processes from THIS checkout, without killing any."""
    found = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            arguments = directory.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = directory.joinpath('cwd').resolve(strict=True)
            for argument in arguments[1:]:
                if argument and Path(argument).name == 'main.py':
                    candidate = Path(argument)
                    if not candidate.is_absolute():
                        candidate = cwd / candidate
                    if candidate.resolve() == (root / 'main.py').resolve():
                        found.append(int(directory.name))
                        break
        except (OSError, ValueError):
            continue
    return sorted(found)


def check_sources(root):
    """Catch an old Model that silently swallows new SSM keys through **kwargs."""
    model_argument = 'ssm_bc_norm_layers' if SUITE == 'bc10' else 'ssm_bidir_layers'
    block_argument = 'bc_norm' if SUITE == 'bc10' else 'bidirectional'
    expected = ((root / 'model/Hyperformer.py', 'Model', model_argument),
                (root / 'model/Hyperformer.py', 'TCN_ViT_unit', 'ssm_args'),
                (root / 'mamba_blocks.py', 'MambaBlock', block_argument))
    for path, class_name, argument in expected:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        constructor = next((node for cls in classes for node in cls.body
                            if isinstance(node, ast.FunctionDef) and node.name == '__init__'), None)
        names = [] if constructor is None else [node.arg for node in constructor.args.args + constructor.args.kwonlyargs]
        if argument not in names:
            raise RuntimeError(f'Outdated source: {path} must explicitly accept {argument}; synchronize SSM code first.')
    training = ast.parse((root / 'main.py').read_text(encoding='utf-8-sig'))
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'add_argument' and node.args
               and isinstance(node.args[0], ast.Constant)
               and node.args[0].value == '--save-latest' for node in ast.walk(training)):
        raise RuntimeError('Outdated main.py: --save-latest is missing; synchronize the training code first.')
    if SUITE == 'bc10' and not any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'add_argument' and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == '--ssm-protect-addons' for node in ast.walk(training)):
        raise RuntimeError('Outdated main.py: --ssm-protect-addons is missing; synchronize the training code first.')
    for mode in MODES:
        if not config_path(root, mode).is_file():
            raise RuntimeError(f'Missing configuration: {config_path(root, mode)}')


def parse_result(directory, epoch_limit=None):
    """Read metrics; an explicit limit treats only that training prefix as reference.

    Prefix summaries deliberately ignore the final run's best-accuracy/checkpoint
    metadata, which can describe a later epoch outside the comparison budget.
    """
    log = directory / 'log.txt'
    result = dict(mode=directory.name, trained_epoch=0, eval_epoch=0,
                  best_top1=None, best_epoch=None, last_top1=None,
                  best_top5=None, final_top1=None, finished=False)
    if not log.is_file():
        return result
    scores, top5, eval_epoch = {}, {}, None
    final_best = None
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        if match := re.search(r'Training epoch:\s*(\d+)', line):
            result['trained_epoch'] = max(result['trained_epoch'], int(match[1]))
        if match := re.search(r'Eval epoch:\s*(\d+)', line):
            eval_epoch = int(match[1])
            result['eval_epoch'] = max(result['eval_epoch'], eval_epoch)
        if match := re.search(r'Top1:\s*([\d.]+)%', line):
            if eval_epoch is not None and (epoch_limit is None or eval_epoch <= epoch_limit):
                scores[eval_epoch] = float(match[1])
        if match := re.search(r'Top5:\s*([\d.]+)%', line):
            if eval_epoch is not None and (epoch_limit is None or eval_epoch <= epoch_limit):
                top5[eval_epoch] = float(match[1])
        if (match := re.search(r'Best accuracy:\s*([\d.eE+-]+)', line)) and epoch_limit is None:
            final_best = float(match[1]) * 100
        if (match := re.search(r'Epoch number:\s*(\d+)', line)) and epoch_limit is None:
            result['best_epoch'] = int(match[1])
    if scores:
        best_epoch = max(scores, key=scores.get)
        result['best_top1'] = scores[best_epoch]
        result['best_epoch'] = result['best_epoch'] or best_epoch
        result['last_top1'] = scores[max(scores)]
        result['final_top1'] = scores.get(EPOCHS if epoch_limit is None else epoch_limit)
        result['best_top5'] = top5.get(result['best_epoch'])
    if final_best is not None:
        result['best_top1'] = final_best
    result['finished'] = (final_best is not None and result['trained_epoch'] >= EPOCHS
                          and (directory / 'runs-best.pt').is_file())
    if epoch_limit is not None:
        result['trained_epoch'] = min(result['trained_epoch'], epoch_limit)
        result['eval_epoch'] = min(result['eval_epoch'], epoch_limit)
        result['finished'] = epoch_limit in scores
    return result


def plan_mode(root, mode):
    directory = group_dir(root, mode)
    if parse_result(directory)['finished']:
        return 'skip', None
    latest = directory / 'runs-latest.pt'
    if latest.is_file():
        return 'resume', latest
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f'{directory} contains an unfinished run without runs-latest.pt. '
                           'Keep these files and inspect the failure; automatic overwrite is refused.')
    return 'new', None


def preflight(root, modes=None):
    modes = MODES if modes is None else modes
    check_sources(root)
    busy = training_processes(root)
    if busy:
        raise RuntimeError(f'Other main.py processes are active in this checkout (PIDs {busy}). '
                           'Stop the old foreground experiment chain before starting this queue.')
    for mode in modes:
        plan_mode(root, mode)


def lock_queue(queue):
    import fcntl
    queue.mkdir(parents=True, exist_ok=True)
    handle = (queue / 'queue.lock').open('a')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('The full-experiment queue is already running; use status/logs.') from None
    return handle


def detached_process(command, root, output, lock_fd):
    return subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                            stdout=output, stderr=subprocess.STDOUT,
                            start_new_session=True, pass_fds=(lock_fd,))


def start_queue(root, queue, modes=None):
    modes = MODES if modes is None else modes
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Start the queue on AutoDL Linux using the existing training Python environment.')
    with lock_queue(queue) as handle:
        preflight(root, modes)
        write_state(queue, status='starting', current_mode=None, modes=list(modes), suite=SUITE)
        command = [sys.executable, str(Path(__file__).resolve()), '--suite', SUITE,
                   'run', '--lock-fd', str(handle.fileno()),
                   '--modes', *modes]
        with (queue / 'runner.log').open('a', buffering=1, encoding='utf-8') as output:
            process = detached_process(command, root, output, handle.fileno())
    print(f'Queue launched: PID {process.pid}; Python: {sys.executable}')
    print(f'Queue log: {queue / "runner.log"}')
    print('Use status to confirm startup. The detached queue survives SSH disconnection; '
          'a stopped/restarted cloud instance requires starting the queue again.')


def run_modes(root, queue, modes=None, spawn=None):
    """Core queue logic; injected subprocesses allow CPU-only behavioral tests."""
    modes = MODES if modes is None else modes
    spawn = spawn or subprocess.Popen
    pid = os.getpid()
    state = dict(status='running', manager_pid=pid, manager_start=process_start(pid),
                 current_mode=None, training_pid=None, modes=list(modes), suite=SUITE)
    write_state(queue, **state)
    for mode in modes:
        action, checkpoint = plan_mode(root, mode)
        if action == 'skip':
            print(f'SKIP completed: {mode}', flush=True)
            continue
        busy = training_processes(root)
        if busy:
            raise RuntimeError(f'Another main.py process appeared (PIDs {busy}); queue stopped.')
        directory = group_dir(root, mode)
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(root / 'main.py'), '--config', str(config_path(root, mode))]
        if checkpoint is not None:
            command.extend(('--resume', str(checkpoint)))
        print(f'{action.upper()}: {mode}; console: {directory / "console.log"}', flush=True)
        state.update(current_mode=mode, training_pid=None, action=action)
        write_state(queue, **state)
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        if not environment.get('OMP_NUM_THREADS', '').isdigit() or int(environment.get('OMP_NUM_THREADS', '0')) < 1:
            environment['OMP_NUM_THREADS'] = '1'
        with (directory / 'console.log').open('a', buffering=1, encoding='utf-8') as output:
            child = spawn(command, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                          stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            state.update(training_pid=child.pid)
            write_state(queue, **state)
            handlers = {}

            def stop(signum, frame):
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

            for signum in (signal.SIGTERM, signal.SIGINT):
                handlers[signum] = signal.signal(signum, stop)
            try:
                code = child.wait()
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
        if code != 0 or not parse_result(directory)['finished']:
            state.update(status='failed', training_pid=None, exit_code=code,
                         error=f'{mode} did not finish successfully. Inspect its console.log; later groups were not started.')
            write_state(queue, **state)
            print(state['error'], flush=True)
            return code if code > 0 else 1
        print(f'COMPLETED: {mode}', flush=True)
    state.update(status='completed', current_mode=None, training_pid=None, exit_code=0)
    write_state(queue, **state)
    print(f'All requested {EPOCHS}-epoch experiments completed: {", ".join(modes)}.', flush=True)
    return 0


def run_queue(root, queue, lock_fd, modes=None):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    handle = os.fdopen(lock_fd, 'a') if lock_fd is not None else lock_queue(queue)
    with handle:
        try:
            preflight(root, modes)
            return run_modes(root, queue, modes)
        except Exception as error:
            state = read_state(queue)
            state.update(status='failed', training_pid=None, error=str(error), manager_pid=os.getpid(),
                         manager_start=process_start(os.getpid()))
            write_state(queue, **state)
            print(f'QUEUE FAILED: {error}', flush=True)
            return 1


def print_results(root):
    print(f'mode\tstatus\ttrained_epoch\tbest_top1_%\tbest_epoch\tbest_top5_%\tepoch{EPOCHS}_top1_%')
    reference = None
    if SUITE == 'bc10':
        reference = parse_result(root / 'work_dir/ntu60/xsub/full_baseline_20ep', epoch_limit=10)
        values = ['baseline_first10', 'reference' if reference['finished'] else 'reference_incomplete',
                  reference['trained_epoch'], reference['best_top1'], reference['best_epoch'],
                  reference['best_top5'], reference['final_top1']]
        print('\t'.join('-' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
                        for value in values))
    for mode in MODES:
        directory = group_dir(root, mode)
        result = parse_result(directory)
        status = 'completed' if result['finished'] else ('incomplete' if directory.exists() else 'not_started')
        values = [mode, status, result['trained_epoch'], result['best_top1'], result['best_epoch'],
                  result['best_top5'], result['final_top1']]
        print('\t'.join('-' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
                        for value in values))
        if (reference is not None and reference['finished'] and result['finished']
                and reference['best_top1'] is not None and result['best_top1'] is not None):
            print(f'{mode} best-vs-baseline-first10: '
                  f'{result["best_top1"] - reference["best_top1"]:+.4f} percentage points')
    if SUITE == 'bc10':
        print('Reference uses only epochs 1-10 from full_baseline_20ep/log.txt; '
              'later epochs/final summaries and baseline weights are not used.')


def show_logs(root, queue, mode, lines, follow):
    def select_path():
        current = mode or read_state(queue).get('current_mode')
        return group_dir(root, current) / 'console.log' if current else queue / 'runner.log'

    current, stream = None, None
    try:
        while True:
            selected = select_path()
            if selected != current:
                if stream:
                    stream.close()
                if not selected.is_file():
                    if not follow:
                        print(f'No log yet: {selected}')
                        return
                    time.sleep(0.5)
                    continue
                current = selected
                print(f'--- {current} ---', flush=True)
                stream = current.open(encoding='utf-8', errors='replace')
                print(''.join(deque(stream, maxlen=lines)), end='', flush=True)
            if not follow:
                return
            chunk = stream.read()
            if chunk:
                print(chunk, end='', flush=True)
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if stream:
            stream.close()


def main():
    suite_parser = argparse.ArgumentParser(add_help=False)
    suite_parser.add_argument('--suite', choices=('ssm20', 'bc10'), default='ssm20')
    suite_args, _ = suite_parser.parse_known_args()
    configure_suite(suite_args.suite)
    parser = argparse.ArgumentParser(description=__doc__, parents=[suite_parser])
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start', help='launch/resume the sequential queue in a detached session')
    start.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    internal = commands.add_parser('run', help='foreground queue worker (normally launched by start)')
    internal.add_argument('--lock-fd', type=int, default=None, help=argparse.SUPPRESS)
    internal.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    commands.add_parser('status', help='read queue state and per-group progress')
    logs = commands.add_parser('logs', help='read current console; follows the next group automatically')
    logs.add_argument('--mode', choices=MODES)
    logs.add_argument('--lines', type=int, default=40)
    logs.add_argument('--follow', action='store_true')
    commands.add_parser('results', help='print a tab-separated summary; no training is launched')
    args = parser.parse_args()
    queue = ROOT / QUEUE_REL
    try:
        if args.command == 'start':
            if len(set(args.modes)) != len(args.modes):
                parser.error('--modes must not contain duplicates')
            start_queue(ROOT, queue, tuple(args.modes))
        elif args.command == 'run':
            if len(set(args.modes)) != len(args.modes):
                parser.error('--modes must not contain duplicates')
            return run_queue(ROOT, queue, args.lock_fd, tuple(args.modes))
        elif args.command == 'status':
            state = read_state(queue)
            print(json.dumps(dict(state, manager_alive=manager_alive(state)), ensure_ascii=False, indent=2))
            print_results(ROOT)
        elif args.command == 'logs':
            if args.lines < 1:
                parser.error('--lines must be positive')
            show_logs(ROOT, queue, args.mode, args.lines, args.follow)
        else:
            print_results(ROOT)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
