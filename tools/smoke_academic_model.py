"""Run one synthetic forward/backward/evaluation for an experiment on AutoDL."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.Hyperformer import Model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('baseline', 'ctr', 'fr', 'ctr_fr'))
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required by the Simba backbone for this test')
    path = ROOT / 'config' / 'nturgbd-cross-subject' / f'joint_{args.mode}_10ep.yaml'
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    torch.manual_seed(config['seed'])
    model = Model(**config['model_args']).cuda()
    model.train()
    num_classes = config['model_args']['num_class']
    frames = config['train_feeder_args']['window_size']
    points = config['model_args']['num_point']
    persons = config['model_args']['num_person']
    inputs = torch.randn(args.batch_size, 3, frames, points, persons, device='cuda')
    labels = torch.arange(args.batch_size, device='cuda') % num_classes
    logits, second = model(inputs, F.one_hot(labels, num_classes))
    auxiliary = second if model.fr_head is not None else logits.new_zeros(())
    loss = F.cross_entropy(logits, labels) + auxiliary
    if logits.shape != (args.batch_size, num_classes) or not torch.isfinite(loss):
        raise RuntimeError('Invalid model output or loss')
    loss.backward()
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Non-finite gradients')
    if model.fr_head is not None and not any(p.grad is not None for p in model.fr_head.parameters()):
        raise RuntimeError('FR-Head did not receive gradients')
    memories = {k: v.clone() for k, v in model.state_dict().items() if 'prototypes' in k}
    model.eval()
    with torch.no_grad():
        eval_logits, _ = model(inputs, F.one_hot(labels, num_classes))
    if eval_logits.shape != logits.shape or not torch.isfinite(eval_logits).all():
        raise RuntimeError('Invalid evaluation output')
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in memories.items()):
        raise RuntimeError('FR prototype memory changed during evaluation')
    torch.cuda.synchronize()
    print(f'{args.mode}: forward/backward/eval passed; loss={loss.item():.4f}, '
          f'weighted FR={auxiliary.item():.4f}, peak GPU memory='
          f'{torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB')


if __name__ == '__main__':
    main()
