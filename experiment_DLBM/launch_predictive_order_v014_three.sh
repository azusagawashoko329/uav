#!/usr/bin/env bash
# v014 uses learned order novelty as cumulative boundary evidence.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VERSION="DLBM_v014_predictive_order_novelty"
export BOUNDARY_EVIDENCE="novelty"
exec bash "$ROOT/experiment_DLBM/launch_predictive_order_v013_three.sh"
