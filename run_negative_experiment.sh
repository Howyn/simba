#!/usr/bin/env bash
set -euo pipefail
mode="${1:?Usage: bash run_negative_experiment.sh baseline|negative [10|20]}"
end_epoch="${2:-10}"
case "$mode" in baseline|negative) ;; *) echo "Invalid mode" >&2; exit 2;; esac
case "$end_epoch" in 10|20) ;; *) echo "Only 10 or 20 epochs are supported" >&2; exit 2;; esac
config="config/nturgbd-cross-subject/joint_negative_${mode}.yaml"
if [[ "$mode" == negative ]]; then
  python -m unittest discover -s tests -p "test_negative_scan.py" -v
fi
work="work_dir/ntu60/xsub/negative_eigen_${mode}"
if [[ "$end_epoch" == 10 ]]; then
  if [[ -e "$work" ]]; then echo "Refusing to overwrite existing $work" >&2; exit 1; fi
  python main.py --config "$config" --num-epoch 10
else
  checkpoint="$work/runs-latest.pt"
  [[ -f "$checkpoint" ]] || { echo "Missing $checkpoint" >&2; exit 1; }
  python - "$checkpoint" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if checkpoint.get('epoch') != 9:
    raise SystemExit('Expected checkpoint immediately after epoch 10; refusing resume')
PY
  python main.py --config "$config" --num-epoch 20 --resume "$checkpoint" --strict-rng-resume True
fi
