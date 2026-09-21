#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyx/miniconda3/envs/DMAD/bin/python}"

exec "$PYTHON_BIN" "$ROOT/experiment_mask_DLBM/launch_probe10.py" --gpus 0,1
