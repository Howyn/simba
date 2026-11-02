"""Launch explicitly selected low-budget algorithm runs and summarize curves.

The manager imports only the Python standard library.  Training and CUDA
signal checks use the user's activated PyTorch environment in separate runs.
Read-only status/results/logs commands do not create directories.
"""
import argparse
import ast
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import manage_full_experiments as legacy

MODES = ('baseline', 'joint_state', 'dual_scale',
         'context_write', 'retain_write', 'context_retain_write')
CANDIDATES = MODES[1:]
CONTEXT_MODES = ('context_write', 'retain_write', 'context_retain_write')
COMBINATION_MODE = 'context_retain_write'
STAGE_MODES = ('baseline',) + CONTEXT_MODES
COMBINATION_MARGIN = 0.2
PAIRING_FIELDS = ('schema_version', 'seed', 'source_sha256', 'dataset_sha256',
                  'recipe_fingerprint', 'device')
# Parameter-layout proofs are one CPU model construction each; cache per process.
_ARCHITECTURE_CACHE = {}


def group_dir(root, mode, seed=4321):
    return root / 'work_dir/ntu60/xsub' / f'algorithm_{mode}_seed{seed}'


def queue_dir(root, seed=4321):
    return root / 'work_dir/ntu60/xsub' / f'algorithm_screening_seed{seed}_queue'


def config_path(root, mode):
    return root / 'config/nturgbd-cross-subject' / f'joint_algorithm_{mode}.yaml'


def receipt_path(root, mode, seed):
    return root / 'work_dir/ntu60/xsub/algorithm_checks' / f'{mode}_seed{seed}.json'


def parse_curve(directory):
    """Accept only evaluations paired with an explicit training epoch.

    main.py evaluates its best checkpoint once more as Eval epoch: 1 before
    the final summary.  That evaluation must not overwrite true epoch one.
    """
    path = directory / 'log.txt'
    result = dict(scores={}, top5={}, trained_epoch=0, summary_epoch=0,
                  training_seconds={}, peak_memory_mib={}, parameters=None)
    if not path.is_file():
        return result
    active_train, active_eval, expect_parameters = None, None, False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if 'Parameters:' in line:
            expect_parameters = True
            continue
        if expect_parameters and line.strip():
            try:
                value = ast.literal_eval(line.strip())
                if isinstance(value, dict):
                    result['parameters'] = value
            except (ValueError, SyntaxError):
                pass
            expect_parameters = False
        if match := re.search(r'Training epoch:\s*(\d+)', line):
            active_train = int(match[1])
            active_eval = None
            result['trained_epoch'] = max(result['trained_epoch'], active_train)
        if match := re.search(r'Eval epoch:\s*(\d+)', line):
            candidate = int(match[1])
            active_eval = candidate if candidate == active_train and candidate > 0 else None
        if active_eval is not None:
            if match := re.search(r'\bTop1:\s*([\d.]+)%', line):
                result['scores'][active_eval] = float(match[1])
            if match := re.search(r'\bTop5:\s*([\d.]+)%', line):
                result['top5'][active_eval] = float(match[1])
        if active_train is not None:
            if match := re.search(r'Training:\s*([\d.]+)\s+seconds', line):
                result['training_seconds'][active_train] = float(match[1])
            if match := re.search(r'Peak allocated CUDA memory:\s*([\d.]+)\s+MiB', line):
                result['peak_memory_mib'][active_train] = float(match[1])
            if re.search(r'Best accuracy:\s*[\d.eE+-]+', line):
                result['summary_epoch'] = active_train
    return result


def complete(curve, epochs):
    return (all(epoch in curve['scores'] for epoch in range(1, epochs + 1))
            and curve['summary_epoch'] >= epochs)


def metrics(curve, epochs=10):
    scores = curve['scores']
    available = {epoch: score for epoch, score in scores.items() if 1 <= epoch <= epochs}
    result = dict(completed=complete(curve, epochs), epochs=len(available),
                  e10=scores.get(10), e20=scores.get(20) if epochs >= 20 else None,
                  final=scores.get(epochs), last3=None, last5=None,
                  post_second_decay=None, best=None, best_epoch=None,
                  best_top5=None, training_hours=None, peak_memory_mib=None)
    if available:
        best_epoch = max(available, key=lambda epoch: (available[epoch], -epoch))
        result.update(best=available[best_epoch], best_epoch=best_epoch,
                      best_top5=curve['top5'].get(best_epoch))
    for key, window in (('last3', range(epochs - 2, epochs + 1)),
                        ('last5', range(epochs - 4, epochs + 1))):
        if all(epoch in scores for epoch in window):
            result[key] = sum(scores[epoch] for epoch in window) / len(window)
    if epochs >= 20 and 19 in scores and 20 in scores:
        result['post_second_decay'] = (scores[19] + scores[20]) / 2
    seconds = curve['training_seconds']
    if all(epoch in seconds for epoch in range(1, epochs + 1)):
        result['training_hours'] = sum(seconds[epoch] for epoch in range(1, epochs + 1)) / 3600
    memory = [value for epoch, value in curve['peak_memory_mib'].items() if epoch <= epochs]
    if memory:
        result['peak_memory_mib'] = max(memory)
    return result


def compare(candidate, baseline, epochs=10):
    current, reference = metrics(candidate, epochs), metrics(baseline, epochs)
    result = dict(current)
    for field in ('e10', 'e20', 'final', 'last3', 'last5', 'best', 'post_second_decay'):
        result['delta_' + field] = (None if current[field] is None or reference[field] is None
                                   else current[field] - reference[field])
    pairs = [epoch for epoch in range(1, epochs + 1)
             if epoch in candidate['scores'] and epoch in baseline['scores']]
    result['wins'] = sum(candidate['scores'][epoch] > baseline['scores'][epoch] for epoch in pairs)
    result['paired_epochs'] = len(pairs)
    result['last3_deltas'] = [candidate['scores'][epoch] - baseline['scores'][epoch]
                              for epoch in range(epochs - 2, epochs + 1)
                              if epoch in candidate['scores'] and epoch in baseline['scores']]
    return result


def classify(comparison, signal_passed=True):
    if signal_passed is False:
        return '先修模块'
    if not comparison['completed'] or comparison['paired_epochs'] != 10:
        return '未完成'
    delta = comparison['delta_last3']
    if delta is None:
        return '未完成'
    final_wins = sum(value > 0 for value in comparison['last3_deltas'])
    strong = delta >= 0.5 - 1e-9 and comparison['wins'] >= 6 and final_wins >= 2
    borderline = (delta >= 0.3 - 1e-9 and delta < 0.5 - 1e-9
                  and comparison['wins'] >= 7 and final_wins == 3)
    if strong or borderline:
        return '值得追加'
    return '观察' if delta > 0 else '暂缓'


def choose_winner(comparisons):
    eligible = [mode for mode, result in comparisons.items()
                if mode in CANDIDATES and result.get('signal_passed', True) is True
                and result.get('comparison_valid', True)
                and classify(result) == '值得追加']
    if not eligible:
        return None
    def order(mode):
        result = comparisons[mode]
        cost = result.get('training_hours')
        return (result['delta_last3'], result['wins'], -(cost if cost is not None else float('inf')))
    return max(eligible, key=order)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def classify5(comparison, signal_passed=True):
    """Pre-registered five-epoch rule for the context/retain stage.

    Promotion needs all three conditions: mean Top1 over epochs 3-5 at least
    0.5 percentage points above the paired baseline, at least four of the five
    epochs ahead, and epochs 3, 4 and 5 all ahead.
    """
    if signal_passed is False:
        return '先修模块'
    if not comparison['completed'] or comparison['paired_epochs'] != 5:
        return '未完成'
    delta = comparison['delta_last3']
    if delta is None:
        return '未完成'
    final_wins = sum(value > 0 for value in comparison['last3_deltas'])
    if delta >= 0.5 - 1e-9 and comparison['wins'] >= 4 and final_wins == 3:
        return '值得追加'
    return '观察' if delta > 0 else '暂缓'


def eligible5(comparisons):
    """Eligible candidates after the combination-margin preference.

    A combination must beat the stronger single mechanism by about 0.2 points
    on the epochs 3-5 mean; otherwise the single mechanism is preferred, as
    pre-registered. Keeping the single one also makes the winning claim about
    one mechanism instead of an unexplained sum.
    """
    eligible = {mode: result for mode, result in comparisons.items()
                if mode in CONTEXT_MODES and result.get('signal_passed', True) is True
                and result.get('comparison_valid', True)
                and classify5(result) == '值得追加'}
    if COMBINATION_MODE not in eligible:
        return eligible
    singles = {mode: result for mode, result in eligible.items() if mode != COMBINATION_MODE}
    if not singles:
        return eligible
    best_single = max(singles.values(), key=lambda result: result['delta_last3'])
    if eligible[COMBINATION_MODE]['delta_last3'] < best_single['delta_last3'] + COMBINATION_MARGIN - 1e-9:
        return singles
    return eligible


def choose_winner5(comparisons):
    eligible = eligible5(comparisons)
    if not eligible:
        return None
    def order(mode):
        result = eligible[mode]
        cost = result.get('training_hours')
        return (result['delta_last3'], result['wins'], -(cost if cost is not None else float('inf')))
    return max(eligible, key=order)


def baseline_directory(root, seed):
    reference = read_json(queue_dir(root, seed) / 'baseline_reference.json')
    return Path(reference['directory']) if reference else group_dir(root, 'baseline', seed)


def protocol_module():
    # Lazy import keeps read-only reporting usable without a GPU or PyTorch.
    from tools import algorithm_protocol
    return algorithm_protocol


def stored_comparable(candidate_dir, baseline_dir, legacy=False):
    a = read_json(candidate_dir / 'run_manifest.json')
    b = read_json(baseline_dir / 'run_manifest.json')
    if bool(a and b and all(field in a and field in b and a[field] == b[field]
                            for field in PAIRING_FIELDS)):
        return True
    # A frozen legacy baseline may predate the current source revision. Its
    # recipe, dataset, device and parameter layout were verified once when the
    # reference was frozen, so only the candidate's own manifest is required.
    return bool(legacy and a)


def validate_pairing(identities):
    """Reject candidate configuration drift before spending baseline epochs."""
    baseline = identities['baseline']
    for mode, identity in identities.items():
        differing = [field for field in PAIRING_FIELDS
                     if field not in identity or identity[field] != baseline.get(field)]
        if differing:
            raise RuntimeError(f'{mode}: paired identity differs from baseline: {", ".join(differing)}. '
                               'Use the same data, device, seed and training recipe for every group.')


def frozen_reference(root, seed):
    return read_json(queue_dir(root, seed) / 'baseline_reference.json')


def load_comparisons(root, seed, epochs=10, candidates=None):
    base_dir = baseline_directory(root, seed)
    legacy = bool(frozen_reference(root, seed).get('legacy'))
    baseline = parse_curve(base_dir)
    if candidates is None:
        candidates = [mode for mode in CANDIDATES
                      if group_dir(root, mode, seed).exists()
                      or receipt_path(root, mode, seed).is_file()]
    results = {}
    for mode in candidates:
        result = compare(parse_curve(group_dir(root, mode, seed)), baseline, epochs)
        receipt = read_json(receipt_path(root, mode, seed))
        result['signal_passed'] = receipt.get('passed') is True
        result['comparison_valid'] = stored_comparable(group_dir(root, mode, seed), base_dir, legacy)
        results[mode] = result
    return results


def show_results(root, seed, as_json=False):
    baseline_dir = baseline_directory(root, seed)
    baseline = parse_curve(baseline_dir)
    comparisons = load_comparisons(root, seed)
    output = {'seed': seed, 'baseline_directory': str(baseline_dir),
              'baseline10': metrics(baseline, 10), 'candidates': {},
              'recommended_extension': choose_winner(comparisons)}
    for mode, result in comparisons.items():
        receipt = read_json(receipt_path(root, mode, seed))
        result['decision'] = (classify(result, receipt.get('passed') is True) if receipt
                              else '先修模块' if result['completed'] else '未检查')
        if result['completed'] and not result['comparison_valid']:
            result['decision'] = '配方未核验'
        curve = parse_curve(group_dir(root, mode, seed))
        result['twenty_epochs'] = compare(curve, baseline, 20)
        output['candidates'][mode] = result
    if as_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return output
    print('mode\tE10\tlast3_8-10\tbest10\tdelta_E10_pp\tdelta_last3_pp\tdelta_best_pp\twins/10\tdecision')
    def number(value):
        return '-' if value is None else f'{value:.4f}'
    base = output['baseline10']
    print('\t'.join(('baseline', number(base['e10']), number(base['last3']), number(base['best']),
                     '0', '0', '0', '-', 'reference' if base['completed'] else 'incomplete')))
    for mode, result in output['candidates'].items():
        print('\t'.join((mode, number(result['e10']), number(result['last3']), number(result['best']),
                         number(result['delta_e10']), number(result['delta_last3']),
                         number(result['delta_best']), str(result['wins']), result['decision'])))
        print(f'  last3 deltas: {result["last3_deltas"]}; '
              f'train-hours(excludes eval): {number(result["training_hours"])}; '
              f'peak-memory-MiB: {number(result["peak_memory_mib"])}')
        twenty = result['twenty_epochs']
        if twenty['e20'] is not None:
            print(f'  E20={number(twenty["e20"])}; mean18-20={number(twenty["last3"])} '
                  f'(delta={number(twenty["delta_last3"])} pp); '
                  f'mean16-20={number(twenty["last5"])} '
                  f'(delta={number(twenty["delta_last5"])} pp); '
                  f'mean19-20={number(twenty["post_second_decay"])} '
                  f'(delta={number(twenty["delta_post_second_decay"])} pp)')
    print('Recommended single extension:', output['recommended_extension'] or 'none')
    print('These are exploratory budget rules, not statistical significance or proof of final effectiveness.')
    return output


def show_stage_results(root, seed, as_json=False, epochs=5):
    """Report the context/retain stage and any ten-epoch extension already run."""
    baseline_dir = baseline_directory(root, seed)
    baseline = parse_curve(baseline_dir)
    reference = frozen_reference(root, seed)
    comparisons = load_comparisons(root, seed, epochs, CONTEXT_MODES)
    output = {'seed': seed, 'stage_epochs': epochs,
              'baseline_directory': str(baseline_dir),
              'baseline_legacy_reuse': bool(reference.get('legacy')),
              'baseline': metrics(baseline, epochs), 'candidates': {},
              'recommended_extension': choose_winner5(comparisons)}
    baseline_ten = metrics(baseline, 10)
    for mode, result in comparisons.items():
        receipt = read_json(receipt_path(root, mode, seed))
        result['decision'] = (classify5(result, receipt.get('passed') is True) if receipt
                              else '先修模块' if result['completed'] else '未检查')
        if result['completed'] and not result['comparison_valid']:
            result['decision'] = '配方未核验'
        result['ten_epochs'] = compare(parse_curve(group_dir(root, mode, seed)), baseline, 10)
        output['candidates'][mode] = result
    if as_json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return output
    def number(value):
        return '-' if value is None else f'{value:.4f}'
    base = output['baseline']
    print(f'mode\tE{epochs}\tmean3-{epochs}\tbest{epochs}\tdelta_E{epochs}_pp'
          f'\tdelta_mean3-{epochs}_pp\twins/{epochs}\tdecision')
    print('\t'.join(('baseline', number(base['final']), number(base['last3']), number(base['best']),
                     '0', '0', '-', 'reference' if base['completed'] else 'incomplete')))
    for mode, result in output['candidates'].items():
        print('\t'.join((mode, number(result['final']), number(result['last3']), number(result['best']),
                         number(result['delta_final']), number(result['delta_last3']),
                         str(result['wins']), result['decision'])))
        print(f'  last3 deltas: {result["last3_deltas"]}; '
              f'train-hours(excludes eval): {number(result["training_hours"])}; '
              f'peak-memory-MiB: {number(result["peak_memory_mib"])}')
        ten = result['ten_epochs']
        if ten['completed']:
            print(f'  E10={number(ten["final"])} (delta={number(ten["delta_final"])} pp); '
                  f'mean8-10={number(ten["last3"])} (delta={number(ten["delta_last3"])} pp); '
                  f'wins/10={ten["wins"]}; baseline10 E10={number(baseline_ten["final"])}')
    print('Recommended single extension:', output['recommended_extension'] or 'none')
    print('Five-epoch rules are exploratory budget rules, not statistical significance.')
    return output


def validate_receipt(root, mode, seed, identity):
    path = receipt_path(root, mode, seed)
    receipt = read_json(path)
    if not receipt:
        raise RuntimeError(f'Missing signal-check receipt: {path}. Run check_algorithm_signal.py first.')
    if receipt.get('passed') is not True or receipt.get('checked_steps', 0) < 40:
        raise RuntimeError(f'{mode}: signal check did not pass 40 updates; fix the module before training.')
    protocol_module().validate_identity(receipt, identity, kind='receipt')
    return receipt


def checkpoint_header(path):
    """Read only safe metadata using the activated training Python, on CPU."""
    script = (
        "import json,sys,torch; "
        "c=torch.load(sys.argv[1],map_location='cpu',weights_only=True); "
        "print(json.dumps({'epoch':c.get('epoch'),'has_rng':bool(c.get('rng_state')),"
        "'has_optimizer':'optimizer_state_dict' in c,'has_model':'model_state_dict' in c}))"
    )
    process = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True,
                             text=True, encoding='utf-8', errors='replace')
    if process.returncode:
        raise RuntimeError(f'Cannot safely inspect {path} on CPU: {process.stderr[-2000:]}')
    try:
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(f'Invalid checkpoint metadata returned for {path}') from None


def validate_checkpoint(path, target, require_tenth=False, required_epoch=None):
    header = checkpoint_header(path)
    epoch = header.get('epoch')
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0 or epoch >= target
            or not header.get('has_model') or not header.get('has_optimizer')):
        raise RuntimeError(f'{path}: checkpoint epoch/model/optimizer does not match target {target}')
    if require_tenth and epoch != 9:
        raise RuntimeError(f'{path}: extending to 20 requires the actual epoch-10 latest checkpoint, got {epoch + 1}')
    if required_epoch is not None and epoch + 1 != required_epoch:
        raise RuntimeError(f'{path}: extending to {target} requires the actual epoch-{required_epoch} '
                           f'latest checkpoint, got {epoch + 1}')
    if not header.get('has_rng'):
        raise RuntimeError(f'{path}: RNG state is missing. Strict paired continuation is refused; '
                           'do not silently resume a legacy/best checkpoint.')
    return epoch + 1


def plan_mode(root, mode, seed, target, identity):
    directory = group_dir(root, mode, seed)
    curve = parse_curve(directory)
    if directory.exists() and any(directory.iterdir()):
        protocol = protocol_module()
        manifest = protocol.load_manifest(directory)
        protocol.validate_identity(manifest, identity, kind='manifest')
        if complete(curve, target):
            return 'skip', None
        checkpoint = directory / 'runs-latest.pt'
        if not checkpoint.is_file():
            raise RuntimeError(f'{directory}: unfinished run has no runs-latest.pt; preserving files, refusing overwrite.')
        require_tenth = target == 20 and complete(curve, 10) and max(curve['scores'], default=0) <= 10
        required_epoch = None
        if target == 10 and complete(curve, 5) and max(curve['scores'], default=0) <= 5:
            required_epoch = 5
        validate_checkpoint(checkpoint, target, require_tenth=require_tenth,
                            required_epoch=required_epoch)
        return 'resume', checkpoint
    if target == 20:
        raise RuntimeError(f'{mode}: a complete 10-epoch run is required before extending to 20.')
    if target == 10 and mode in CONTEXT_MODES:
        raise RuntimeError(f'{mode}: a complete 5-epoch stage is required before extending to 10.')
    return 'new', None


def verify_baseline_architecture(directory):
    """Compare the saved baseline checkpoint layout with the current code."""
    script = ROOT / 'tools/verify_baseline_architecture.py'
    process = subprocess.run([sys.executable, str(script), '--directory', str(directory)],
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode or not lines:
        detail = (process.stderr or process.stdout).strip()[-2000:]
        raise RuntimeError(f'Baseline architecture proof failed for {directory}: {detail}')
    try:
        record = json.loads(lines[-1])
    except ValueError:
        raise RuntimeError(f'Baseline architecture proof returned invalid JSON for {directory}') from None
    if record.get('ok') is not True:
        differing = record.get('mismatched') or record.get('missing') or record.get('unexpected')
        raise RuntimeError('Baseline checkpoint layout differs from the current baseline network: '
                           f'{record.get("error") or differing}')
    return record


def validate_reference(root, seed, identity, target=10):
    reference = frozen_reference(root, seed)
    if not reference:
        return None
    directory = Path(reference['directory'])
    legacy = bool(reference.get('legacy'))
    if legacy:
        if reference.get('architecture_verified') is not True:
            raise RuntimeError('The frozen legacy baseline has no architecture proof; freeze it again.')
        protocol_module().validate_legacy_baseline(root, directory, seed, identity)
        if not complete(parse_curve(directory), target):
            raise RuntimeError(f'The frozen legacy baseline has fewer than {target} complete epochs; '
                               'external runs are never extended or overwritten.')
    else:
        protocol_module().validate_baseline_manifest(root, directory, seed, identity)
        if not complete(parse_curve(directory), 20):
            raise RuntimeError('A reused external baseline must have all 20 epochs; '
                               'external runs are never extended or overwritten.')
    return directory


def reuse_baseline(root, seed, directory):
    directory = directory.resolve()
    if legacy.manager_alive(legacy.read_state(queue_dir(root, seed))) or legacy.training_processes(root):
        raise RuntimeError('Finish the current queue/training before freezing a baseline reference.')
    protocol = protocol_module()
    identity = protocol.current_identity(root, 'baseline', seed)
    protocol.validate_baseline_manifest(root, directory, seed, identity)
    if not complete(parse_curve(directory), 20):
        raise RuntimeError('Reuse requires a manifest-verified complete 20-epoch baseline; its first 10 epochs will be used for screening.')
    queue = queue_dir(root, seed)
    previous = read_json(queue / 'baseline_reference.json')
    if previous and Path(previous['directory']) != directory:
        raise RuntimeError('A different baseline reference is already frozen for this seed; do not change it after screening.')
    local = group_dir(root, 'baseline', seed)
    if local.exists() and any(local.iterdir()) and local.resolve() != directory:
        raise RuntimeError('This seed already has its own baseline run; refusing a second reference.')
    queue.mkdir(parents=True, exist_ok=True)
    record = dict(directory=str(directory), seed=seed, validated_at=datetime.now(timezone.utc).isoformat())
    (queue / 'baseline_reference.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Frozen reusable baseline: {directory}; no source results were modified.')


def validate_baseline_for_pairing(root, seed, identity, target=10):
    """Validate the baseline a candidate is paired with, once per process.

    A frozen reference goes through ``validate_reference``. Otherwise the local
    baseline group is checked through the legacy-compatible path plus a cached
    parameter-layout proof, so a baseline trained before this source revision
    can still be reused without retraining it.
    """
    reference = frozen_reference(root, seed)
    if reference:
        return validate_reference(root, seed, identity, target)
    directory = group_dir(root, 'baseline', seed)
    protocol_module().validate_legacy_baseline(root, directory, seed, identity)
    key = str(directory.resolve())
    if key not in _ARCHITECTURE_CACHE:
        _ARCHITECTURE_CACHE[key] = verify_baseline_architecture(directory)
    return directory


def reuse_legacy_baseline(root, seed, directory):
    """Freeze a baseline trained before the current source revision.

    Adding the context/retain modules changes every mode's source fingerprint,
    which would otherwise force a redundant baseline retrain. Reuse is allowed
    only after proving the saved actual recipe, dataset, device and the
    checkpoint's parameter layout still match the current baseline network. The
    reference records exactly what was and was not compared.
    """
    directory = directory.resolve()
    if legacy.manager_alive(legacy.read_state(queue_dir(root, seed))) or legacy.training_processes(root):
        raise RuntimeError('Finish the current queue/training before freezing a baseline reference.')
    protocol = protocol_module()
    identity = protocol.current_identity(root, 'baseline', seed)
    notes = protocol.validate_legacy_baseline(root, directory, seed, identity)
    curve = parse_curve(directory)
    if not complete(curve, 5):
        raise RuntimeError('Legacy reuse requires at least the first five evaluated epochs of that run.')
    proof = verify_baseline_architecture(directory)
    queue = queue_dir(root, seed)
    previous = frozen_reference(root, seed)
    if previous and Path(previous['directory']) != directory:
        raise RuntimeError('A different baseline reference is already frozen for this seed; do not change it after screening.')
    local = group_dir(root, 'baseline', seed)
    if local.exists() and any(local.iterdir()) and local.resolve() != directory:
        raise RuntimeError('This seed already has its own baseline run; refusing a second reference.')
    queue.mkdir(parents=True, exist_ok=True)
    record = dict(directory=str(directory), seed=seed, legacy=True,
                  architecture_verified=True, architecture=proof, baseline=notes,
                  available_epochs=len(curve['scores']),
                  validated_at=datetime.now(timezone.utc).isoformat())
    (queue / 'baseline_reference.json').write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                                   encoding='utf-8')
    print(f'Frozen reusable legacy baseline: {directory} ({record["available_epochs"]} scored epochs).')
    print('Verified: actual recipe, seed, dataset fingerprint, GPU/runtime, algorithms off, '
          f'and {proof["parameters"]} parameters in an identical checkpoint layout.')
    if notes.get('extra_model_args'):
        print('Tolerated extra model_args keys (identical layout proves their values): '
              + ', '.join(f'{key}={value}' for key, value in notes['extra_model_args'].items()))
    print('Not compared: source_sha256 of the old run (the new candidate modules changed it). '
          'No source result was modified.')


def preflight(root, seed, modes, target, inspect_processes=True):
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Choose a nonempty list of distinct modes explicitly')
    if inspect_processes:
        busy = legacy.training_processes(root)
        if busy:
            raise RuntimeError(f'Another main.py is active in this checkout (PIDs {busy}); finish it before starting.')
    protocol = protocol_module()
    identities = {mode: protocol.current_identity(root, mode, seed) for mode in set(modes) | {'baseline'}}
    validate_pairing(identities)
    if target == 5:
        return stage5_preflight(root, seed, tuple(modes), identities)
    if target == 10 and any(mode in CONTEXT_MODES for mode in modes):
        return extension10_preflight(root, seed, tuple(modes), identities)
    reference = validate_reference(root, seed, identities['baseline'], target)
    if target == 10:
        if reference and 'baseline' in modes:
            modes = tuple(mode for mode in modes if mode != 'baseline')
        if 'baseline' in modes and modes[0] != 'baseline':
            raise RuntimeError('When building the baseline in this queue, list baseline first.')
        if any(mode in CANDIDATES for mode in modes) and not reference:
            base_curve = parse_curve(group_dir(root, 'baseline', seed))
            if 'baseline' not in modes and not complete(base_curve, 10):
                raise RuntimeError('Run baseline first, or explicitly include it first in --modes.')
            if complete(base_curve, 10):
                validate_baseline_for_pairing(root, seed, identities['baseline'], target)
        for mode in modes:
            if mode in CANDIDATES:
                validate_receipt(root, mode, seed, identities[mode])
    else:
        selected = [mode for mode in modes if mode in CANDIDATES]
        if len(selected) != 1:
            raise RuntimeError('A 20-epoch extension must contain exactly one candidate.')
        allowed = extension_modes(root, seed, selected[0])
        if tuple(modes) != allowed:
            raise RuntimeError(f'Extension mode list must be {list(allowed)}.')
        if not reference:
            validate_baseline_for_pairing(root, seed, identities['baseline'], target)
        validate_receipt(root, selected[0], seed, identities[selected[0]])
    plans = {mode: plan_mode(root, mode, seed, target, identities[mode]) for mode in modes}
    return tuple(modes), identities, plans


def stage5_preflight(root, seed, modes, identities):
    """Preflight for the five-epoch context/retain stage."""
    unknown = [mode for mode in modes if mode not in STAGE_MODES]
    if unknown:
        raise ValueError(f'start5 accepts only {list(STAGE_MODES)}; got {unknown}')
    reference = validate_reference(root, seed, identities['baseline'], 5)
    if reference and 'baseline' in modes:
        modes = tuple(mode for mode in modes if mode != 'baseline')
        print('Frozen baseline reference is in use; the baseline group is skipped.')
    if 'baseline' in modes and modes[0] != 'baseline':
        raise RuntimeError('When building the baseline in this queue, list baseline first.')
    if not modes:
        raise RuntimeError('Nothing to run: the frozen baseline already covers the five-epoch stage.')
    if 'baseline' not in modes and not reference:
        if not complete(parse_curve(group_dir(root, 'baseline', seed)), 5):
            raise RuntimeError('Run baseline first, or explicitly include it first in --modes.')
        validate_baseline_for_pairing(root, seed, identities['baseline'], 5)
    for mode in modes:
        if mode in CONTEXT_MODES:
            validate_receipt(root, mode, seed, identities[mode])
    plans = {mode: plan_mode(root, mode, seed, 5, identities[mode]) for mode in modes}
    return modes, identities, plans


def extension10_preflight(root, seed, modes, identities):
    """Preflight for extending the single five-epoch winner to ten epochs."""
    selected = [mode for mode in modes if mode in CONTEXT_MODES]
    if len(selected) != 1:
        raise RuntimeError('A 10-epoch extension must contain exactly one context candidate.')
    allowed = extension_modes_stage(root, seed, selected[0])
    if modes != allowed:
        raise RuntimeError(f'Extension mode list must be {list(allowed)}.')
    base_dir = validate_baseline_for_pairing(root, seed, identities['baseline'], 10)
    if not complete(parse_curve(base_dir), 5):
        raise RuntimeError('The 10-epoch extension requires a baseline with five complete epochs.')
    validate_receipt(root, selected[0], seed, identities[selected[0]])
    plans = {mode: plan_mode(root, mode, seed, 10, identities[mode]) for mode in modes}
    return modes, identities, plans


def extension_modes_stage(root, seed, selected):
    """Only the pre-registered five-epoch winner may be extended to ten."""
    comparisons = load_comparisons(root, seed, 5, CONTEXT_MODES)
    winner = choose_winner5(comparisons)
    if selected != winner:
        raise RuntimeError(f'Only the single eligible five-epoch winner may be extended. '
                           f'Current recommendation: {winner or "none"}.')
    selection = read_json(queue_dir(root, seed) / 'stage_extension.json')
    if selection and selection.get('mode') != selected:
        raise RuntimeError(f'This seed already allocated its five-epoch extension to '
                           f'{selection.get("mode")}; refusing a second candidate.')
    base_dir = baseline_directory(root, seed)
    if complete(parse_curve(base_dir), 10):
        return (selected,)
    if base_dir != group_dir(root, 'baseline', seed):
        raise RuntimeError('Reused external baseline is not complete at 10; it will not be modified.')
    return ('baseline', selected)


def extension_modes(root, seed, selected):
    comparisons = load_comparisons(root, seed)
    winner = choose_winner(comparisons)
    if selected != winner:
        raise RuntimeError(f'Only the single eligible winner may be extended. Current recommendation: {winner or "none"}.')
    selection = read_json(queue_dir(root, seed) / 'extension_selection.json')
    if selection and selection.get('mode') != selected:
        raise RuntimeError(f'This seed already allocated its extension to {selection.get("mode")}; refusing a second candidate.')
    base_dir = baseline_directory(root, seed)
    if complete(parse_curve(base_dir), 20):
        return (selected,)
    if base_dir != group_dir(root, 'baseline', seed):
        raise RuntimeError('Reused external baseline is not complete at 20; it will not be modified.')
    return ('baseline', selected)


def global_lock_dir(root):
    # One lock across seeds prevents two new queues using one GPU concurrently.
    return root / 'work_dir/ntu60/xsub/algorithm_screening_global_lock'


def start_queue(root, seed, modes, target, selected=None):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Launch training on AutoDL Linux using the activated training environment.')
    queue = queue_dir(root, seed)
    with legacy.lock_queue(global_lock_dir(root)) as handle:
        modes, _, _ = preflight(root, seed, modes, target)
        if not modes:
            print('Baseline reference is already complete; no training was requested.')
            return
        queue.mkdir(parents=True, exist_ok=True)
        legacy.write_state(queue, status='starting', current_mode=None, modes=list(modes), target=target, seed=seed)
        if selected:
            selection_name = ('stage_extension.json'
                              if target == 10 and any(mode in CONTEXT_MODES for mode in modes)
                              else 'extension_selection.json')
            (queue / selection_name).write_text(json.dumps(dict(mode=selected, seed=seed), indent=2), encoding='utf-8')
        command = [sys.executable, str(Path(__file__).resolve()), '--seed', str(seed),
                   '_run', '--target', str(target), '--lock-fd', str(handle.fileno()), '--modes', *modes]
        with (queue / 'runner.log').open('a', buffering=1, encoding='utf-8') as output:
            process = legacy.detached_process(command, root, output, handle.fileno())
    print(f'Queue launched: PID {process.pid}; target={target}; modes={list(modes)}; Python={sys.executable}')
    print('Use status/logs to confirm progress. The queue survives SSH disconnection, but not cloud instance shutdown.')


def run_modes(root, seed, modes, target, spawn=None):
    spawn = spawn or subprocess.Popen
    queue = queue_dir(root, seed)
    queue.mkdir(parents=True, exist_ok=True)
    modes, identities, _ = preflight(root, seed, modes, target)
    pid = os.getpid()
    state = dict(status='running', manager_pid=pid, manager_start=legacy.process_start(pid),
                 current_mode=None, training_pid=None, modes=list(modes), target=target, seed=seed)
    legacy.write_state(queue, **state)
    for mode in modes:
        # A queue may run for hours. Recheck fingerprints after each earlier
        # group so a git pull/config edit cannot mix old receipts with new code.
        current = protocol_module().current_identity(root, mode, seed)
        protocol_module().validate_identity(current, identities[mode], kind='queue source/config/data')
        if mode in CANDIDATES:
            validate_receipt(root, mode, seed, current)
        action, checkpoint = plan_mode(root, mode, seed, target, identities[mode])
        if action == 'skip':
            print(f'SKIP completed: {mode}', flush=True)
            continue
        if legacy.training_processes(root):
            raise RuntimeError('Another main.py appeared; stopping before the next group.')
        if mode in CANDIDATES:
            if validate_reference(root, seed, current, target) is None:
                protocol_module().validate_legacy_baseline(
                    root, group_dir(root, 'baseline', seed), seed, current)
        directory = group_dir(root, mode, seed)
        directory.mkdir(parents=True, exist_ok=True)
        if action == 'new':
            protocol_module().write_manifest(directory, identities[mode])
        command = [sys.executable, '-u', str(root / 'main.py'), '--config', str(config_path(root, mode)),
                   '--work-dir', str(directory), '--seed', str(seed), '--num-epoch', str(target)]
        if checkpoint is not None:
            command.extend(('--resume', str(checkpoint)))
        state.update(current_mode=mode, training_pid=None, action=action)
        legacy.write_state(queue, **state)
        environment = os.environ.copy()
        environment['PYTHONUNBUFFERED'] = '1'
        if not environment.get('OMP_NUM_THREADS', '').isdigit() or int(environment.get('OMP_NUM_THREADS', '0')) < 1:
            environment['OMP_NUM_THREADS'] = '1'
        print(f'{action.upper()}: {mode}; console: {directory / "console.log"}', flush=True)
        with (directory / 'console.log').open('a', buffering=1, encoding='utf-8') as output:
            child = spawn(command, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                          stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            state.update(training_pid=child.pid)
            legacy.write_state(queue, **state)
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
        if code != 0 or not complete(parse_curve(directory), target):
            state.update(status='failed', training_pid=None, exit_code=code,
                         error=f'{mode} did not complete {target} epochs; later modes were not started.')
            legacy.write_state(queue, **state)
            print(state['error'], flush=True)
            return code if code > 0 else 1
    state.update(status='completed', current_mode=None, training_pid=None, exit_code=0)
    legacy.write_state(queue, **state)
    print('All explicitly requested groups completed.', flush=True)
    return 0


def run_worker(root, seed, modes, target, lock_fd):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    handle = os.fdopen(lock_fd, 'a') if lock_fd is not None else legacy.lock_queue(global_lock_dir(root))
    with handle:
        try:
            return run_modes(root, seed, modes, target)
        except Exception as error:
            queue = queue_dir(root, seed)
            state = legacy.read_state(queue)
            state.update(status='failed', training_pid=None, error=str(error), manager_pid=os.getpid(),
                         manager_start=legacy.process_start(os.getpid()))
            legacy.write_state(queue, **state)
            print(f'QUEUE FAILED: {error}', flush=True)
            return 1


def logs(root, seed, mode=None, lines=40, follow=False):
    queue = queue_dir(root, seed)
    current, stream = None, None
    try:
        while True:
            selected_mode = mode or legacy.read_state(queue).get('current_mode')
            selected = (group_dir(root, selected_mode, seed) / 'console.log'
                        if selected_mode else queue / 'runner.log')
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
                print(f'--- {selected} ---', flush=True)
                stream = selected.open(encoding='utf-8', errors='replace')
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=4321)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start10', help='explicitly launch selected 10-epoch groups after receipt checks')
    start.add_argument('--modes', nargs='+', choices=MODES, required=True)
    start5 = commands.add_parser('start5', help='launch the five-epoch context/retain stage')
    start5.add_argument('--modes', nargs='+', choices=STAGE_MODES, required=True)
    extend10 = commands.add_parser('extend10', help='extend the single eligible five-epoch winner to ten epochs')
    extend10.add_argument('--mode', choices=CONTEXT_MODES, required=True)
    extend = commands.add_parser('extend20', help='explicitly extend the single eligible winner and its baseline')
    extend.add_argument('--mode', choices=CANDIDATES, required=True)
    reuse = commands.add_parser('reuse-baseline', help='freeze an existing manifest-verified complete baseline20')
    reuse.add_argument('--directory', type=Path, required=True)
    legacy_reuse = commands.add_parser(
        'reuse-legacy-baseline',
        help='freeze a baseline trained before this source revision after recipe/layout verification')
    legacy_reuse.add_argument('--directory', type=Path, required=True)
    worker = commands.add_parser('_run', help=argparse.SUPPRESS)
    worker.add_argument('--modes', nargs='+', choices=MODES, required=True)
    worker.add_argument('--target', type=int, choices=(5, 10, 20), required=True)
    worker.add_argument('--lock-fd', type=int)
    commands.add_parser('status')
    results = commands.add_parser('results')
    results.add_argument('--json', action='store_true')
    results5 = commands.add_parser('results5', help='summarize the five-epoch context/retain stage')
    results5.add_argument('--json', action='store_true')
    log = commands.add_parser('logs')
    log.add_argument('--mode', choices=MODES)
    log.add_argument('--lines', type=int, default=40)
    log.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'start5':
            start_queue(ROOT, args.seed, tuple(args.modes), 5)
        elif args.command == 'extend10':
            modes = extension_modes_stage(ROOT, args.seed, args.mode)
            start_queue(ROOT, args.seed, modes, 10, selected=args.mode)
        elif args.command == 'start10':
            start_queue(ROOT, args.seed, tuple(args.modes), 10)
        elif args.command == 'extend20':
            modes = extension_modes(ROOT, args.seed, args.mode)
            start_queue(ROOT, args.seed, modes, 20, selected=args.mode)
        elif args.command == 'reuse-baseline':
            reuse_baseline(ROOT, args.seed, args.directory)
        elif args.command == 'reuse-legacy-baseline':
            reuse_legacy_baseline(ROOT, args.seed, args.directory)
        elif args.command == '_run':
            return run_worker(ROOT, args.seed, tuple(args.modes), args.target, args.lock_fd)
        elif args.command == 'status':
            state = legacy.read_state(queue_dir(ROOT, args.seed))
            print(json.dumps(dict(state, manager_alive=legacy.manager_alive(state)), ensure_ascii=False, indent=2))
            show_results(ROOT, args.seed)
        elif args.command == 'results':
            show_results(ROOT, args.seed, args.json)
        elif args.command == 'results5':
            show_stage_results(ROOT, args.seed, args.json)
        else:
            if args.lines < 1:
                parser.error('--lines must be positive')
            logs(ROOT, args.seed, args.mode, args.lines, args.follow)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
