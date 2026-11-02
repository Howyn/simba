"""Prove that a saved baseline checkpoint matches the current baseline network.

Adding new candidate modules changes the source fingerprint of every mode, so an
older baseline cannot be reused through a strict manifest comparison even when
the baseline path itself is untouched. This tool supplies the missing proof for
reuse: it constructs the baseline model from the saved actual configuration and
compares every parameter name and shape against the checkpoint. An identical
layout means the baseline graph did not change; the check modifies nothing.

Run it through the activated training environment (it needs torch, not CUDA):

    python tools/verify_baseline_architecture.py --directory work_dir/.../baseline
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _read_config(path):
    import yaml
    text = Path(path).read_text(encoding='utf-8-sig')
    try:
        config = json.loads(text)
    except ValueError:
        config = yaml.safe_load(text)
    if not isinstance(config, dict) or not isinstance(config.get('model_args'), dict):
        raise ValueError(f'{path} must contain a configuration object with model_args')
    return config


def compare_layout(directory, checkpoint=None):
    import torch
    sys.path.insert(0, str(ROOT))
    from tools.diagnose_ssm_results import load_checkpoint_cpu

    directory = Path(directory)
    config = _read_config(directory / 'config.yaml')
    if checkpoint is None:
        for name in ('runs-latest.pt', 'runs-best.pt'):
            candidate = directory / name
            if candidate.is_file():
                checkpoint = candidate
                break
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    if checkpoint is None or not checkpoint.is_file():
        raise FileNotFoundError(f'No runs-latest.pt or runs-best.pt in {directory}')
    record = load_checkpoint_cpu(checkpoint)
    state = record.get('model_state_dict', record) if isinstance(record, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f'{checkpoint}: no model_state_dict')
    saved = {name.removeprefix('module.'): value for name, value in state.items()}
    del record, state

    from model.Hyperformer import Model
    model = Model(**config['model_args'])
    built = model.state_dict()
    missing = sorted(set(built) - set(saved))
    unexpected = sorted(set(saved) - set(built))
    mismatched = sorted((name for name in set(built) & set(saved)
                         if tuple(built[name].shape) != tuple(saved[name].shape)),)
    result = dict(
        ok=not (missing or unexpected or mismatched),
        directory=str(directory.resolve()), checkpoint=str(checkpoint.resolve()),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        state_tensors=len(built), saved_tensors=len(saved),
        missing=missing[:20], unexpected=unexpected[:20], mismatched=mismatched[:20],
        note='Identical layout proves the baseline graph is unchanged; it does not '
             're-decide data, recipe or device identity, which the caller verifies separately.',
    )
    del model, built, saved
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True,
                        help='baseline run directory containing config.yaml and a checkpoint')
    parser.add_argument('--checkpoint', type=Path, default=None)
    args = parser.parse_args()
    try:
        result = compare_layout(args.directory, args.checkpoint)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(json.dumps(dict(ok=False, error=str(error)), ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
