"""Standard-library identity checks for the budget SSM screening protocol.

Importing this module never imports torch, YAML, or the model. GPU/runtime
metadata is queried in a short subprocess with the active Python executable.
New protocol configurations are JSON documents, which are also valid YAML.
"""

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SCHEMA_VERSION = 1
MODES = ('baseline', 'joint_state', 'dual_scale', 'context_write',
         'retain_write', 'context_retain_write')
MANIFEST_NAME = 'run_manifest.json'
SOURCE_FILES = (
    'main.py', 'mamba_blocks.py', 'model/Hyperformer.py', 'layers.py',
    'shiftgcn_model/shift_gcn.py', 'graph/ntu_rgb_d.py', 'graph/tools.py',
    'mamba_ssm/modules/mamba_simple.py',
    'mamba_ssm/modules/mamba_joint_state.py',
    'mamba_ssm/modules/mamba_dual_scale.py',
    'mamba_ssm/modules/mamba_context_write.py',
    'mamba_ssm/ops/selective_scan_interface.py',
    'feeders/feeder_ntu.py', 'feeders/tools.py',
    'tools/check_algorithm_signal.py', 'tools/ssm_optimizer.py',
    'tools/training_rng.py', 'tools/algorithm_protocol.py',
    'tools/verify_baseline_architecture.py',
    'requirements-autodl.txt',
)
ALGORITHM_MODEL_KEYS = frozenset((
    'ctr_layers', 'fr_layers', 'fr_layer_weights', 'fr_loss_weight',
    'fr_hidden_channels', 'fr_input_frames',
    'ssm_bidir_layers', 'ssm_motion_layers', 'ssm_motion_rank',
    'ssm_motion_scale', 'ssm_bidir_init', 'ssm_bc_norm_layers',
    'ssm_bc_norm_gain', 'ssm_bc_norm_bias', 'ssm_bc_norm_eps',
    'ssm_joint_state_layers', 'ssm_joint_state_weight',
    'ssm_dual_scale_layers', 'ssm_slow_scale', 'ssm_scale_gate_init',
    'ssm_context_layers', 'ssm_retain_layers', 'ssm_context_rank',
    'ssm_context_kernels', 'ssm_context_scale', 'ssm_retain_init',
    'ssm_retain_floor', 'ssm_write_init', 'ssm_write_max',
))
IGNORED_RECIPE_KEYS = frozenset((
    'work_dir', 'model_saved_name', 'config', 'phase', 'num_epoch',
    'start_epoch', 'resume', 'weights', 'ignore_weights',
    'save_interval', 'save_epoch', 'save_latest', 'save_score',
    'print_log', 'show_topk', 'log_interval', 'joint_label',
    'lambda_1', 'lambda_2',
))
# Defaults whose command-line overrides must not escape actual-config checks
# merely because a compact input configuration omitted the default key.
RECIPE_DEFAULTS = {
    'optimizer': 'SGD', 'lr_type': 'cosine', 'momentum': 0.9,
    'nesterov': False, 'base_lr': 0.025, 'step': [110, 120],
    'warm_up_epoch': 0, 'weight_decay': 0.0005, 'lr_decay_rate': 0.1,
    'batch_size': 256, 'test_batch_size': 256, 'num_worker': 48,
    'precision': 'fp32', 'tf32': None, 'persistent_workers': False,
    'prefetch_factor': 16, 'ema': False, 'ssm_protect_addons': False,
    'eval_interval': 5, 'device': [0, 1],
}
IDENTITY_KEYS = ('schema_version', 'mode', 'seed', 'config_sha256',
                 'source_sha256', 'dataset_sha256', 'recipe_fingerprint', 'device')


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _text_bytes(path):
    return Path(path).read_text(encoding='utf-8-sig').replace('\r\n', '\n').encode('utf-8')


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_config(root, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown algorithm mode: {mode}')
    path = Path(root) / 'config/nturgbd-cross-subject' / f'joint_algorithm_{mode}.yaml'
    try:
        config = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f'Cannot read protocol JSON/YAML configuration {path}: {error}') from error
    if not isinstance(config, dict) or not isinstance(config.get('model_args'), dict):
        raise ValueError(f'{path} must contain a configuration object and model_args object')
    return path, config


def canonical_recipe(config, seed):
    """Remove only operational differences and explicitly named algorithms."""
    recipe = {key: value for key, value in config.items() if key not in IGNORED_RECIPE_KEYS}
    for key, value in RECIPE_DEFAULTS.items():
        recipe.setdefault(key, value)
    recipe['seed'] = int(seed)
    model_args = recipe.get('model_args')
    if not isinstance(model_args, dict):
        raise ValueError('model_args must be an object')
    recipe['model_args'] = {key: value for key, value in model_args.items()
                            if key not in ALGORITHM_MODEL_KEYS}
    return recipe


def source_fingerprint(root):
    root = Path(root)
    records = []
    for relative in sorted(SOURCE_FILES):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f'Missing protocol source: {path}; synchronize the complete code update')
        records.append((relative, _sha256(_text_bytes(path))))
    return _sha256(_canonical_bytes(records))


def _hash_binary(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def dataset_fingerprint(root, path):
    """Stream a dataset hash once, then reuse it only for the same file stat."""
    root = Path(root)
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'Dataset must be a file: {path}')
    signature = _signature(path)
    cache_dir = root / 'work_dir/ntu60/xsub/algorithm_checks/.cache'
    cache_file = cache_dir / (_sha256(str(path).encode('utf-8')) + '.json')
    try:
        cached = json.loads(cache_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cached = {}
    digest = cached.get('sha256')
    valid_digest = (isinstance(digest, str) and len(digest) == 64
                    and all(character in '0123456789abcdef' for character in digest))
    if valid_digest and all(cached.get(key) == value for key, value in signature.items()):
        return digest
    digest = _hash_binary(path)
    if signature != _signature(path):
        raise RuntimeError(f'Dataset changed while hashing: {path}; retry after data preparation finishes')
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cache_file, dict(signature, sha256=digest))
    return digest


@lru_cache(maxsize=1)
def _runtime_identity_cached():
    code = '''import json, platform, torch
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required to identify this experiment")
print("SIMBA_RUNTIME_IDENTITY=" + json.dumps({
    "name": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "torch": str(torch.__version__), "cuda": torch.version.cuda,
    "python": platform.python_version(), "cudnn": torch.backends.cudnn.version(),
}, sort_keys=True))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=120, check=False)
    marker = 'SIMBA_RUNTIME_IDENTITY='
    matches = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
    if result.returncode != 0 or len(matches) != 1:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f'Cannot query CUDA/runtime identity with {sys.executable}: {detail}')
    try:
        identity = json.loads(matches[0])
    except ValueError as error:
        raise RuntimeError('CUDA/runtime identity subprocess returned invalid JSON') from error
    if not all(key in identity for key in ('name', 'capability', 'torch', 'cuda', 'python')):
        raise RuntimeError('CUDA/runtime identity is incomplete')
    return identity


def runtime_identity():
    """Return a fresh dict; cache the costly torch import within this process."""
    return json.loads(json.dumps(_runtime_identity_cached()))


def current_identity(root, mode, seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    root = Path(root).resolve()
    config_path, config = _read_config(root, mode)
    # The signal check and strict RNG checkpoint continuation implement this
    # published recipe. Reject incompatible edits before GPU queries or NPZ
    # hashing, including baseline-only runs which do not need a probe receipt.
    if (config.get('precision', 'fp32') != 'fp32' or config.get('tf32') is not False
            or config.get('optimizer', 'SGD') != 'SGD'
            or config.get('ema', False) is not False
            or config.get('persistent_workers', False) is not False):
        raise ValueError('This screening protocol requires FP32, TF32=false, SGD, '
                         'EMA=false and persistent_workers=false; keep the published recipe')
    if config.get('device', [0]) != [0]:
        raise ValueError('The protocol currently supports one GPU at device [0]')
    train = config.get('train_feeder_args', {}).get('data_path')
    test = config.get('test_feeder_args', {}).get('data_path')
    if not train or train != test:
        raise ValueError('Protocol train and test must use the same NTU NPZ data_path')
    recipe = canonical_recipe(config, seed)
    device = runtime_identity()
    return dict(schema_version=SCHEMA_VERSION, mode=mode, seed=seed,
                config_sha256=_sha256(_text_bytes(config_path)),
                source_sha256=source_fingerprint(root),
                dataset_sha256=dataset_fingerprint(root, train),
                recipe_fingerprint=_sha256(_canonical_bytes(recipe)),
                device=device)


def _identity(record):
    if isinstance(record, dict) and 'schema_version' not in record and isinstance(record.get('identity'), dict):
        return record['identity']
    if not isinstance(record, dict):
        raise ValueError('Identity record must be a JSON object')
    return record


def validate_identity(record, expected, kind='receipt'):
    record, expected = _identity(record), _identity(expected)
    missing = [key for key in IDENTITY_KEYS if key not in record or key not in expected]
    if missing:
        raise RuntimeError(f'{kind} identity is incomplete: {", ".join(missing)}')
    if record['schema_version'] != SCHEMA_VERSION or expected['schema_version'] != SCHEMA_VERSION:
        raise RuntimeError(f'{kind} identity uses an unsupported schema_version')
    mismatches = [key for key in IDENTITY_KEYS if record[key] != expected[key]]
    if mismatches:
        raise RuntimeError(f'{kind} identity mismatch: {", ".join(mismatches)}')
    return True


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                   allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_manifest(directory, identity):
    validate_identity(identity, identity, kind='new run manifest')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    if path.is_file():
        validate_identity(load_manifest(directory), identity, kind='existing run manifest')
        return path
    record = dict(_identity(identity))
    record['created_at'] = datetime.now(timezone.utc).isoformat()
    _write_json(path, record)
    return path


def load_manifest(directory):
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError(f'No {MANIFEST_NAME} in {directory}; baseline identity cannot be verified. '
                           'Keep the old results; this command will not overwrite or automatically retrain them.')
    try:
        record = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Cannot read run manifest {path}: {error}') from error
    if not isinstance(record, dict):
        raise RuntimeError(f'Run manifest must be a JSON object: {path}')
    return record


def _saved_config(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'Baseline actual config.yaml is missing: {path}')
    text = path.read_text(encoding='utf-8-sig')
    try:
        value = json.loads(text)
    except ValueError:
        # The trainer saves PyYAML syntax. Parse safely in the active runtime
        # while keeping this module's own imports and CPU tests stdlib-only.
        code = ('import json,sys,yaml; '
                'print(json.dumps(yaml.safe_load(open(sys.argv[1],encoding="utf-8-sig"))))')
        result = subprocess.run([sys.executable, '-c', code, str(path)],
                                capture_output=True, text=True, timeout=30, check=False)
        if result.returncode != 0:
            raise RuntimeError(f'Cannot parse baseline actual config safely: {result.stderr.strip()[-1000:]}')
        try:
            value = json.loads(result.stdout)
        except ValueError as error:
            raise RuntimeError('Actual config parser returned invalid JSON') from error
    if not isinstance(value, dict):
        raise ValueError('Baseline actual config.yaml must contain an object')
    return value


def _assert_baseline_algorithms_off(config):
    args = config.get('model_args', {})
    if not isinstance(args, dict):
        raise ValueError('Baseline actual model_args must be an object')
    enabled = [name for name, value in args.items()
               if (name in ('ctr_layers', 'fr_layers')
                   or name.startswith('ssm_') and name.endswith('_layers')) and value]
    if config.get('ssm_protect_addons', False):
        enabled.append('ssm_protect_addons')
    if enabled:
        raise RuntimeError('Baseline actual config enables algorithm changes: ' + ', '.join(enabled))


def validate_legacy_baseline(root, directory, seed, expected_identity):
    """Verify a baseline produced before the current source revision.

    Adding new candidate modules changes ``source_sha256`` for every mode, so a
    strict manifest comparison would force a redundant baseline retrain even
    though the baseline network is untouched. This check proves everything that
    still decides comparability and reports what it could not compare:

    * the saved actual trainer configuration, with every algorithm flag off;
    * seed, training recipe, dataset fingerprint and GPU/runtime identity;
    * an existing manifest, reporting ``code_drift`` instead of accepting it
      silently.

    Architecture equality is not decided here: the caller must additionally
    compare the baseline checkpoint's parameter names/shapes against a freshly
    constructed baseline model (``tools/verify_baseline_architecture.py``).
    Nothing is modified or overwritten.
    """
    directory = Path(directory)
    if not (directory / 'config.yaml').is_file():
        raise FileNotFoundError(f'Baseline actual config.yaml is missing: {directory / "config.yaml"}')
    expected = _identity(expected_identity)
    for key in ('dataset_sha256', 'recipe_fingerprint', 'device', 'seed'):
        if key not in expected:
            raise RuntimeError('The expected baseline identity is incomplete')
    if expected.get('seed') != seed:
        raise RuntimeError('The expected baseline identity uses a different seed')
    actual = _saved_config(directory / 'config.yaml')
    _assert_baseline_algorithms_off(actual)
    if actual.get('seed') != seed:
        raise RuntimeError('Baseline actual config seed does not match the requested seed')
    _, nominal = _read_config(root, 'baseline')
    expected_recipe = canonical_recipe(nominal, seed)
    actual_recipe = canonical_recipe(actual, seed)
    actual_recipe = {key: actual_recipe.get(key) for key in expected_recipe}
    if actual_recipe != expected_recipe:
        differing = [key for key in expected_recipe if actual_recipe.get(key) != expected_recipe[key]]
        raise RuntimeError('Baseline actual training recipe differs: ' + ', '.join(differing))
    train = actual.get('train_feeder_args', {}).get('data_path')
    test = actual.get('test_feeder_args', {}).get('data_path')
    if not train or train != test:
        raise RuntimeError('Baseline actual config must train and test on one NTU NPZ data_path')
    dataset = dataset_fingerprint(root, train)
    if dataset != expected.get('dataset_sha256'):
        raise RuntimeError('Baseline dataset fingerprint differs from the current dataset')
    device = runtime_identity()
    if device != expected.get('device'):
        raise RuntimeError('Baseline GPU/runtime identity differs from the current device')
    notes = dict(directory=str(directory.resolve()), seed=seed,
                 dataset_sha256=dataset, recipe_verified=True,
                 algorithms_off=True, device_verified=True,
                 manifest_present=(directory / MANIFEST_NAME).is_file(),
                 source_sha256_compared=False, code_drift=False,
                 recorded_source_sha256=None)
    if notes['manifest_present']:
        record = _identity(load_manifest(directory))
        notes['recorded_source_sha256'] = record.get('source_sha256')
        if record.get('source_sha256') == expected.get('source_sha256'):
            notes['source_sha256_compared'] = True
        else:
            notes['code_drift'] = True
    return notes


def validate_baseline_manifest(root, directory, seed, expected_identity):
    """Verify a reusable baseline without reading or unpickling checkpoints.

    Mode and raw config hash differ between baseline and candidate. Every
    common recipe, source, dataset, runtime, and seed identity must agree.
    The saved *actual* trainer configuration is checked as well, so a CLI
    override cannot be hidden behind a correct nominal manifest.
    """
    manifest = _identity(load_manifest(directory))
    expected = _identity(expected_identity)
    required = ('schema_version', 'mode', 'seed', 'config_sha256', 'source_sha256', 'dataset_sha256',
                'recipe_fingerprint', 'device')
    if any(key not in manifest for key in required):
        raise RuntimeError('Baseline manifest identity is incomplete')
    if manifest['schema_version'] != SCHEMA_VERSION or expected.get('schema_version') != SCHEMA_VERSION:
        raise RuntimeError('Baseline manifest uses an unsupported schema_version')
    if manifest['mode'] != 'baseline' or manifest['seed'] != seed or expected.get('seed') != seed:
        raise RuntimeError('Baseline manifest mode/seed does not match the requested baseline')
    common = ('source_sha256', 'dataset_sha256', 'recipe_fingerprint', 'device')
    mismatches = [key for key in common if manifest[key] != expected.get(key)]
    if mismatches:
        raise RuntimeError('Baseline identity mismatch: ' + ', '.join(mismatches))
    _, nominal = _read_config(root, 'baseline')
    expected_recipe = canonical_recipe(nominal, seed)
    if _sha256(_canonical_bytes(expected_recipe)) != expected['recipe_fingerprint']:
        raise RuntimeError('Current baseline nominal recipe differs from the candidate recipe')
    actual = _saved_config(Path(directory) / 'config.yaml')
    _assert_baseline_algorithms_off(actual)
    if actual.get('seed') != seed:
        raise RuntimeError('Baseline actual config seed does not match the requested seed')
    actual_recipe = canonical_recipe(actual, seed)
    # argparse emits additional unused/default operational keys. Compare all
    # nominal recipe keys plus the important training defaults above.
    actual_recipe = {key: actual_recipe.get(key) for key in expected_recipe}
    if actual_recipe != expected_recipe:
        differing = [key for key in expected_recipe if actual_recipe.get(key) != expected_recipe[key]]
        raise RuntimeError('Baseline actual training recipe differs: ' + ', '.join(differing))
    return manifest
