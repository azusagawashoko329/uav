#!/usr/bin/env bash
# Historical solar-only code_dim=32 recipe; use UAV_RUN_EXP for a new registered run.
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
EXP="${UAV_RUN_EXP:-da/appearance/codedim/cd32}"
PY="${PYTHON:-python}"
cd "$ROOT"
if [[ -e "experiments/$EXP/results.csv" || -e "experiments/$EXP/ckpt/solar.pth" ]]; then
  echo "Existing results at experiments/$EXP; use a new experiment ID" >&2
  exit 2
fi
"$PY" train.py --scene solar --code_dim 32 --exp "$EXP" --gpu "${GPU:-0}" --epochs 15 "$@"
"$PY" eval.py --exp "$EXP" --scenes solar --gpu "${GPU:-0}"
