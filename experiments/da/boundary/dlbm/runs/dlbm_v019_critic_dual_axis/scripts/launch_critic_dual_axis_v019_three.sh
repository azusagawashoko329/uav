#!/usr/bin/env bash
# Five-epoch qualification of the learned pseudo-opening critic axis.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export VERSION="DLBM_v019_critic_dual_axis"
export TRAIN_SCRIPT="experiment_DLBM/train_critic_dual_axis_dlbm.py"
export EPOCHS="5"
export STEPS_PER_EPOCH="20"
export MILESTONES="1 3 5"
exec bash "$ROOT/experiments/da/boundary/dlbm/runs/dlbm_v017_dual_axis/scripts/launch_dual_axis_v017_three.sh"
