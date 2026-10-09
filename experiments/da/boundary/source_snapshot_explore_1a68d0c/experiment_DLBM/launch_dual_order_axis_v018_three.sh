#!/usr/bin/env bash
# v018 replaces absolute pseudo localization with an independent pseudo-order axis.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VERSION="DLBM_v018_dual_order_axis"
export TRAIN_SCRIPT="experiment_DLBM/train_dual_order_axis_dlbm.py"
exec bash "$ROOT/experiment_DLBM/launch_dual_axis_v017_three.sh"
