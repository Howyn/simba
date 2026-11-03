#!/usr/bin/env bash
# Run one group of the negative-eigenvalue screen.
#
#   bash run_negative.sh baseline            # original Simba Mamba
#   bash run_negative.sh negative            # 2 * exp(delta * A) - 1 in l1-l10
#
# Extra arguments go straight to main.py, e.g.
#   bash run_negative.sh negative --num-epoch 10 --work-dir work_dir/ntu60/xsub/neg_10ep
set -euo pipefail

mode="${1:-}"
case "$mode" in
  baseline) config=config/nturgbd-cross-subject/joint_autodl.yaml ;;
  negative) config=config/nturgbd-cross-subject/joint_autodl_negative.yaml ;;
  *)
    echo "usage: bash run_negative.sh {baseline|negative} [main.py arguments]" >&2
    exit 2
    ;;
esac
shift

exec python main.py --config "$config" "$@"
