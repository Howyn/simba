#!/usr/bin/env bash
set -euo pipefail
WEIGHTS="${1:?Usage: bash test.sh /path/to/checkpoint.pt [config]}"
CONFIG="${2:-config/nturgbd-cross-subject/joint_autodl.yaml}"
# The config decides scan_mode. A negative-eigenvalue checkpoint must be evaluated with
# config/nturgbd-cross-subject/joint_autodl_negative.yaml, otherwise the operator differs
# from the one used in training (main.py also refuses a mismatched checkpoint).
python main.py --phase test --weights "$WEIGHTS" --save-score True \
  --config "$CONFIG" \
  --model model.Hyperformer.Model --device 0
