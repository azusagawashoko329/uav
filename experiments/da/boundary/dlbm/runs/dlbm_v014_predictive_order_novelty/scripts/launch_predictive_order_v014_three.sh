#!/usr/bin/env bash
# v014 uses learned order novelty as cumulative boundary evidence.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export VERSION="DLBM_v014_predictive_order_novelty"
export BOUNDARY_EVIDENCE="novelty"
exec bash "$ROOT/experiments/da/boundary/dlbm/runs/dlbm_v013_predictive_order_gain/scripts/launch_predictive_order_v013_three.sh"
