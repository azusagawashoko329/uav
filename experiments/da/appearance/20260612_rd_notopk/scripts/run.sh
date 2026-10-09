#!/usr/bin/env bash
# Historical 20260612_rd_notopk recipe. Set UAV_RUN_EXP to a newly registered ID before rerunning.
set -euo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export UAV_RUN_EXP="${UAV_RUN_EXP:-da/appearance/20260612_rd_notopk}"
exec bash "$ROOT/experiments/da/appearance/20260611_rd_baseline/scripts/run.sh" --loss full "$@"
