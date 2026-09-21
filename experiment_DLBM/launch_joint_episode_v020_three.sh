#!/usr/bin/env bash
# Three-scene qualification; reuses the read-only Probe10 and predictors.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=DLBM_v020_joint_episode \
TRAIN_SCRIPT=experiment_DLBM/train_joint_episode_dlbm.py \
EPOCHS="${EPOCHS:-10}" STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-20}" \
MILESTONES="${MILESTONES:-1 3 5 10}" \
bash "$ROOT/experiment_DLBM/launch_dual_axis_v017_three.sh"
