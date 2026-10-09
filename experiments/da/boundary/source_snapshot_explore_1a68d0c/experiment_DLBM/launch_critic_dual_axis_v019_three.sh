#!/usr/bin/env bash
# Five-epoch qualification of the learned pseudo-opening critic axis.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VERSION="DLBM_v019_critic_dual_axis"
export TRAIN_SCRIPT="experiment_DLBM/train_critic_dual_axis_dlbm.py"
export EPOCHS="5"
export STEPS_PER_EPOCH="20"
export MILESTONES="1 3 5"
exec bash "$ROOT/experiment_DLBM/launch_dual_axis_v017_three.sh"
