#!/usr/bin/env bash
# Video-consistent v021 qualification; Probe10 remains read-only.
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
VERSION=DLBM_v022_raw_transport_video_consistent \
TRAIN_SCRIPT=experiment_DLBM/train_raw_transport_dual_axis_dlbm.py \
EPOCHS="${EPOCHS:-10}" STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-20}" \
MILESTONES="${MILESTONES:-1 3 5 10}" EXTRA_ARGS="--video_consistency" \
bash "$ROOT/experiments/da/boundary/dlbm/runs/dlbm_v017_dual_axis/scripts/launch_dual_axis_v017_three.sh"
