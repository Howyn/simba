#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# == 0 )); then
  set -- status
fi
exec python tools/manage_algorithm_screening.py "$@"
