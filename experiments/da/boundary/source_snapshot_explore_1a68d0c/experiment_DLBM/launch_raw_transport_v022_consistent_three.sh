#!/usr/bin/env bash
# Video-consistent v021 qualification; Probe10 remains read-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=DLBM_v022_raw_transport_video_consistent \
TRAIN_SCRIPT=experiment_DLBM/train_raw_transport_dual_axis_dlbm.py \
EPOCHS="${EPOCHS:-10}" STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-20}" \
MILESTONES="${MILESTONES:-1 3 5 10}" EXTRA_ARGS="--video_consistency" \
bash "$ROOT/experiment_DLBM/launch_dual_axis_v017_three.sh"
