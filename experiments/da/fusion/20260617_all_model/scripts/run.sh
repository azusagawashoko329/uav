#!/usr/bin/env bash
# Historical seed-42 recipe; use UAV_RUN_EXP for a new registered experiment.
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
EXP="${UAV_RUN_EXP:-da/fusion/20260617_all_model}"
PY="${PYTHON:-python}"
cd "$ROOT"
if [[ -e "experiments/$EXP/results_fusion.csv" ]] || find "experiments/$EXP/ckpt" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "Existing results at experiments/$EXP; use a new experiment ID" >&2
  exit 2
fi
exec "$PY" fusion/run_all.py --exp "$EXP" --seed 42 "$@"
