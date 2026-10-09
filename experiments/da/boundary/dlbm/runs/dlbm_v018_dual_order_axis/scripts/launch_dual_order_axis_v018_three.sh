#!/usr/bin/env bash
# v018 replaces absolute pseudo localization with an independent pseudo-order axis.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export VERSION="DLBM_v018_dual_order_axis"
export TRAIN_SCRIPT="experiment_DLBM/train_dual_order_axis_dlbm.py"
exec bash "$ROOT/experiments/da/boundary/dlbm/runs/dlbm_v017_dual_axis/scripts/launch_dual_axis_v017_three.sh"
