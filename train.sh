#!/usr/bin/env bash
set -euo pipefail
python main.py --config config/nturgbd-cross-subject/joint_autodl.yaml "$@"
