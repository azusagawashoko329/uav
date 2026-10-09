#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 SCENE GPU" >&2
  exit 2
fi
repo_root="$(git rev-parse --show-toplevel)"
exp_dir="$repo_root/experiments/da/aggregation/20261009_student_adapter_spatial_s42"
python_bin="${UAV_PYTHON:-/home/lyx/miniconda3/envs/DMAD/bin/python}"
cd "$repo_root"
"$python_bin" scripts/run_student_adapter.py --scene "$1" --gpu "$2" --adapter spatial --output "$exp_dir" --seed 42 --epochs 15 --batch 64 --workers 8
