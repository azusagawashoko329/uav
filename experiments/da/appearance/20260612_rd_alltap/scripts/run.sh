#!/usr/bin/env bash
# Historical 20260612_rd_alltap recipe. Set UAV_RUN_EXP to a newly registered ID before rerunning.
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export UAV_RUN_EXP="${UAV_RUN_EXP:-da/appearance/20260612_rd_alltap}"
exec bash "$ROOT/experiments/da/appearance/20260611_rd_baseline/scripts/run.sh" --taps 0 1 2 3 4 5 6 7 8 9 10 11 "$@"
