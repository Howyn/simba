#!/usr/bin/env bash
set -euo pipefail
WEIGHTS="${1:?Usage: bash test.sh /path/to/checkpoint.pt}"
python main.py --phase test --weights "$WEIGHTS" --save-score True \
  --config config/nturgbd-cross-subject/joint_autodl.yaml \
  --model model.Hyperformer.Model --device 0
