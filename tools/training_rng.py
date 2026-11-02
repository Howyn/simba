"""Checkpoint-safe global RNG state for epoch-boundary training recovery.

Only built-in containers/scalars and torch byte tensors are serialized, so
``torch.load(..., weights_only=True)`` does not need a NumPy allowlist.
This covers random, the legacy global np.random generator, Torch CPU and all
visible CUDA generators. It does not capture arbitrary user-created Generator
objects or live persistent DataLoader worker processes. Restoring RNG state
does not make nondeterministic CUDA kernels bitwise deterministic.
"""

import random
import warnings

import numpy as np
import torch


def _byte_state(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f'{name} must be a nonempty one-dimensional torch.uint8 tensor')
    # A checkpoint loaded with map_location='cuda:0' can move even RNG tensors
    # to CUDA; torch.set_rng_state and CUDA RNG setters expect CPU byte states.
    return value.detach().cpu().contiguous().clone()


def capture_rng_state():
    """Capture independent copies without consuming the captured generators."""
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        'format_version': 1,
        'python': random.getstate(),
        'numpy': {
            'algorithm': str(algorithm),
            'keys': [int(value) for value in keys],
            'position': int(position),
            'has_gauss': int(has_gauss),
            'cached_gaussian': float(cached_gaussian),
        },
        'torch_cpu': torch.get_rng_state().detach().cpu().clone(),
        'torch_cuda': [value.detach().cpu().clone() for value in cuda_states],
    }


def restore_rng_state(state, *, strict_cuda=True):
    """Restore global generators, returning False for legacy ``None`` state.

    Call after constructing/loading models and optimizers and immediately
    before creating the next epoch's DataLoader iterator. For this repository's
    non-persistent workers and default sampler, their RNG is derived from the
    restored main-process generators. An explicit DataLoader generator needs
    its own separately captured state.

    With CUDA state present, the default requires the same visible device
    count. ``strict_cuda=False`` permits a CPU-only diagnostic to skip CUDA
    state, or restores only common devices after a device-count change, with a
    warning. That relaxed mode is not an exact training continuation.
    """
    if state is None:
        return False
    if not isinstance(state, dict) or type(state.get('format_version')) is not int or state['format_version'] != 1:
        raise ValueError('unsupported RNG checkpoint format')
    required = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    if not required.issubset(state):
        raise ValueError(f'RNG checkpoint is missing {sorted(required.difference(state))}')

    # Validate isolated generators first, so ordinary malformed metadata is
    # rejected before changing the running process's random streams.
    python_state = state['python']
    if not isinstance(python_state, (tuple, list)) or len(python_state) != 3:
        raise ValueError('invalid Python RNG state')
    try:
        python_state = (python_state[0], tuple(python_state[1]), python_state[2])
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid Python RNG state') from error

    values = state['numpy']
    if not isinstance(values, dict):
        raise ValueError('invalid NumPy RNG state')
    try:
        keys = values['keys']
        if isinstance(keys, torch.Tensor):
            keys = keys.detach().cpu().tolist()
        if not isinstance(keys, (list, tuple)) or any(type(value) is not int or not 0 <= value < 2 ** 32 for value in keys):
            raise ValueError('NumPy keys must be uint32-range Python integers')
        numpy_state = (str(values['algorithm']), np.asarray(keys, dtype=np.uint32),
                       int(values['position']), int(values['has_gauss']), float(values['cached_gaussian']))
        np.random.RandomState().set_state(numpy_state)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('invalid NumPy RNG state') from error

    cpu_state = _byte_state(state['torch_cpu'], 'torch_cpu')
    try:
        torch.Generator(device='cpu').set_state(cpu_state)
    except RuntimeError as error:
        raise ValueError('invalid Torch CPU RNG state') from error
    if not isinstance(state['torch_cuda'], (tuple, list)):
        raise ValueError('torch_cuda must be a list of byte tensors')
    cuda_states = [_byte_state(value, f'torch_cuda[{index}]')
                   for index, value in enumerate(state['torch_cuda'])]
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cuda_states and len(cuda_states) != visible:
        message = (f'RNG checkpoint has {len(cuda_states)} CUDA generators but the current process has '
                   f'{visible} visible CUDA devices; exact CUDA RNG restore is unavailable')
        if strict_cuda:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)
    if cuda_states and visible:
        # Restore only common devices in explicitly relaxed diagnostic mode.
        if len(cuda_states) == visible:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            for index, value in enumerate(cuda_states[:visible]):
                torch.cuda.set_rng_state(value, device=index)
    return True
