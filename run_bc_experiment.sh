#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# == 0 )); then
  set -- start
fi
exec python tools/manage_full_experiments.py --suite bc10 "$@"
