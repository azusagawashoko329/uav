#!/usr/bin/env bash
# v015 fits each scene-level DLBM on all normal training videos.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VERSION="DLBM_v015_fullscene_order_novelty"
export BOUNDARY_EVIDENCE="novelty"
export TRAIN_FULL_SCENE="1"
exec bash "$ROOT/experiment_DLBM/launch_predictive_order_v013_three.sh"
