#!/usr/bin/env bash
# Five-epoch context/retain stage: start5, results5, extend10, status, logs.
# Baseline reuse: reuse-legacy-baseline --directory <existing baseline run>.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# == 0 )); then
  set -- results5
fi
exec python tools/manage_algorithm_screening.py "$@"
