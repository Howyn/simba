#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  baseline|ctr|fr|ctr_fr|ssm_bidir|ssm_motion|ssm_bidir_motion) ;;
  *) echo "Usage: bash run_experiment.sh {baseline|ctr|fr|ctr_fr|ssm_bidir|ssm_motion|ssm_bidir_motion} [main.py options]" >&2; exit 2 ;;
esac
shift
config="config/nturgbd-cross-subject/joint_${mode}_10ep.yaml"
work_dir="work_dir/ntu60/xsub/experiment_${mode}_10ep"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [[ "${args[i]}" == --work-dir ]]; then
    if (( i+1 >= ${#args[@]} )); then echo '--work-dir requires a path' >&2; exit 2; fi
    work_dir="${args[i+1]}"
  elif [[ "${args[i]}" == --work-dir=* ]]; then
    work_dir="${args[i]#--work-dir=}"
  fi
done
if [[ -e "$work_dir/log.txt" ]]; then
  echo "Results already exist in $work_dir. Choose a new --work-dir or resume explicitly with main.py." >&2
  exit 1
fi
python main.py --config "$config" "${args[@]}"
