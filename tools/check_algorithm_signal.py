"""Forty real training updates, followed by an execution/signal gate.

This does not train an experiment checkpoint or test accuracy. All updated
weights are discarded; only a fingerprinted pass/fail JSON receipt is saved.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def assess_signal(rows, gradient_rows, *, initial_parity=True):
    """Engineering execution gate, deliberately independent of accuracy."""
    reasons = []
    if not initial_parity:
        reasons.append('Common backbone initialization or RNG differs from the baseline')
    late = [row for row in rows if row['step'] >= 20]
    if len(late) < 2:
        reasons.append('Missing observations at both step 20 and step 40')
    for row in late:
        numeric = ('correction_ratio', 'logit_relative', 'numerical_floor')
        if any(not math.isfinite(row[name]) for name in numeric):
            reasons.append(f"step {row['step']}: non-finite signal")
            continue
        if not 1e-4 <= row['correction_ratio'] <= 1.0:
            reasons.append(f"step {row['step']}: effective correction is too small or dominates the original path")
        if row['logit_relative'] <= max(1e-6, 10 * row['numerical_floor']):
            reasons.append(f"step {row['step']}: logit change does not exceed the numerical noise floor")
        if not row.get('mechanism_ok', False):
            reasons.append(f"step {row['step']}: state mechanism invariant failed")
    if not gradient_rows:
        reasons.append('No task gradient measurements')
    else:
        names = set(gradient_rows[-1]['parameters'])
        for name in sorted(names):
            gradients = [item['parameters'][name]['task_grad_rms']
                         for item in gradient_rows if item['step'] >= 20]
            if not gradients or max(gradients) <= 1e-12 or not all(map(math.isfinite, gradients)):
                reasons.append(f'No measurable late task gradient for {name}')
    return not reasons, reasons


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('joint_state', 'dual_scale'))
    parser.add_argument('--seed', type=int, default=4321)
    args = parser.parse_args()
    receipt = None
    receipt_path = ROOT / 'work_dir/ntu60/xsub/algorithm_checks' / f'{args.mode}_seed{args.seed}.json'
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from tools.algorithm_protocol import current_identity
        if not torch.cuda.is_available():
            raise RuntimeError('Run this check in the AutoDL virtual environment on the selected GPU')
        torch.cuda.set_device(0)
        identity = current_identity(ROOT, args.mode, args.seed)
        receipt = dict(identity, created_at=datetime.now(timezone.utc).isoformat(),
                       checked_steps=0, passed=False, failure_reasons=['Check not yet completed'])
        atomic_json(receipt_path, receipt)
        config = json.loads((ROOT / f'config/nturgbd-cross-subject/joint_algorithm_{args.mode}.yaml').read_text(encoding='utf-8'))
        base_config = json.loads((ROOT / 'config/nturgbd-cross-subject/joint_algorithm_baseline.yaml').read_text(encoding='utf-8'))
        if (config.get('precision') != 'fp32' or config.get('tf32') is not False
                or config.get('optimizer') != 'SGD' or config.get('ema', False)
                or config.get('persistent_workers') is not False):
            raise ValueError('This screening check requires FP32, TF32=false, SGD, EMA=false '
                             'and persistent_workers=false; keep the published recipe')
        from model.Hyperformer import Model
        from feeders.feeder_ntu import Feeder
        from tools.ssm_optimizer import algorithm_parameter_groups

        def seed_all():
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False

        # Verify all common weights, including later layers and classifier.
        seed_all()
        base_model = Model(**base_config['model_args']).cuda()
        common = {name: value.detach().cpu().clone() for name, value in base_model.state_dict().items()}
        expected_cpu_rng = torch.get_rng_state().clone()
        expected_cuda_rng = torch.cuda.get_rng_state().cpu().clone()
        del base_model
        torch.cuda.empty_cache()
        seed_all()
        model = Model(**config['model_args']).cuda()
        parity = (torch.equal(expected_cpu_rng, torch.get_rng_state())
                  and torch.equal(expected_cuda_rng, torch.cuda.get_rng_state().cpu()))
        state = model.state_dict()
        parity = parity and all(name in state and torch.equal(value, state[name].detach().cpu())
                                for name, value in common.items())
        del common, state
        if not parity:
            raise RuntimeError('Candidate changed common backbone initialization/RNG; fix before training')

        block = model.l4.mamba
        captured = {}
        hook = block.register_forward_pre_hook(
            lambda module, values: captured.update(features=values[0].detach().clone()))
        loader = torch.utils.data.DataLoader(
            Feeder(**config['train_feeder_args']), batch_size=config['batch_size'],
            shuffle=True, num_workers=0, drop_last=True)
        if len(loader) < 40:
            raise ValueError('The real training loader must contain at least 40 full batches')
        iterator = iter(loader)
        first = next(iterator)
        fixed_data = first[0].cuda().float()
        fixed_labels = first[1].cuda().long()
        onehot_fixed = F.one_hot(fixed_labels, model.num_class)
        lr = config['base_lr'] / max(1, config['warm_up_epoch'])
        optimizer = torch.optim.SGD(algorithm_parameter_groups(model, config['weight_decay']),
                                    lr=lr, momentum=config.get('momentum', 0.9),
                                    nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        targets = {name: parameter for name, parameter in model.named_parameters()
                   if (args.mode == 'joint_state' and '.joint_state.' in name)
                   or (args.mode == 'dual_scale' and name.endswith('dual_scale_logit'))}
        if not targets:
            raise RuntimeError('Expected new algorithm parameters were not constructed')
        initial = {name: parameter.detach().clone() for name, parameter in targets.items()}
        weight_decays = {id(parameter): group['weight_decay'] for group in optimizer.param_groups
                         for parameter in group['params']}

        def relative(a, b):
            return float((a.detach().double().norm() / b.detach().double().norm().clamp_min(1e-12)).item())

        def observe(step):
            model.eval()
            with torch.no_grad():
                logits, _ = model(fixed_data, onehot_fixed)
                features = captured['features'][:2]
                if args.mode == 'joint_state':
                    global_output = block.mamba(features)
                    parts = block.joint_state.probe_components(features)
                    ratio = relative(block.joint_state_weight * parts['output'], global_output)
                    block.joint_state_enabled = False
                    try:
                        off, _ = model(fixed_data, onehot_fixed)
                        repeated, _ = model(fixed_data, onehot_fixed)
                    finally:
                        block.joint_state_enabled = True
                    floor = relative(off - repeated, off)
                    mechanism_ok = bool(torch.isfinite(parts['state_readout']).all().item()
                                        and 'D' not in dict(block.joint_state.named_parameters()))
                    extra = {'state_readout_rms': float(parts['state_readout'].float().square().mean().sqrt().item())}
                else:
                    inner = block.mamba
                    parts = inner.probe_components(features)
                    ratio = relative(parts['output'] - parts['output_base'], parts['output_base'])
                    inner.correction_enabled = False
                    try:
                        off, _ = model(fixed_data, onehot_fixed)
                        inner.dual_scale_enabled = False
                        original, _ = model(fixed_data, onehot_fixed)
                    finally:
                        inner.correction_enabled = True
                        inner.dual_scale_enabled = True
                    floor = relative(off - original, original)
                    delta_ratio = float((parts['delta_slow'] / parts['delta_base'].clamp_min(1e-12)).mean().item())
                    slow_scale = config['model_args']['ssm_slow_scale']
                    retention_error = float((parts['retention_slow'] - parts['retention_base'].pow(slow_scale)).abs().max().item())
                    mechanism_ok = abs(delta_ratio - slow_scale) < 1e-4 and retention_error < 1e-5
                    extra = {'delta_ratio': delta_ratio, 'retention_power_error': retention_error,
                             'state_correction_ratio': relative(parts['state_correction'], parts['state_base'])}
                row = dict(step=step, correction_ratio=ratio, logit_relative=relative(logits - off, off),
                           numerical_floor=floor, mechanism_ok=mechanism_ok, **extra)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Non-finite fixed-input logits')
            model.train()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            return row

        observations = [observe(0)]
        gradient_rows = []
        torch.cuda.reset_peak_memory_stats()
        print('Checking 40 updates on real training data; these weights will be discarded.', flush=True)
        for step in range(1, 41):
            data, labels, _ = first if step == 1 else next(iterator)
            data, labels = data.cuda().float(), labels.cuda().long()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(data, F.one_hot(labels, model.num_class))
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite CE loss at step {step}')
            loss.backward()
            finite = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not torch.stack(finite).all().item():
                raise RuntimeError(f'Non-finite task gradient at step {step}')
            if step in (10, 20, 40):
                values = {}
                for name, parameter in targets.items():
                    grad = 0.0 if parameter.grad is None else float(parameter.grad.double().square().mean().sqrt().item())
                    p_rms = float(parameter.detach().double().square().mean().sqrt().item())
                    values[name] = dict(task_grad_rms=grad, decay_grad_rms=weight_decays[id(parameter)] * p_rms)
                gradient_rows.append(dict(step=step, parameters=values))
            optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in targets.values()]).all().item():
                raise RuntimeError(f'Non-finite updated parameters at step {step}')
            receipt['checked_steps'] = step
            if step in (10, 20, 40):
                observations.append(observe(step))
        hook.remove()
        passed, reasons = assess_signal(observations, gradient_rows, initial_parity=parity)
        update_ratios = {name: relative(parameter.detach() - initial[name], initial[name])
                         for name, parameter in targets.items()}
        receipt.update(passed=passed, failure_reasons=reasons, observations=observations,
                       task_gradients=gradient_rows, parameter_update_ratios=update_ratios,
                       initial_backbone_parity=parity,
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                       note='Execution/signal check only; no accuracy gain established. Diagnostic weights discarded.')
        atomic_json(receipt_path, receipt)
        print(f'{"PASS" if passed else "先修模块"}: {receipt_path}', flush=True)
        for reason in reasons:
            print('  ' + reason, flush=True)
        return 0 if passed else 2
    except Exception as error:
        if receipt is not None:
            receipt.update(passed=False, failure_reasons=[str(error)])
            atomic_json(receipt_path, receipt)
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
