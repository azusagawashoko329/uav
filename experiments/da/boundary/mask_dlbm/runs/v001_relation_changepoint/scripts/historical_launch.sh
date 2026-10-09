#!/usr/bin/env bash
# ARCHIVAL ONLY: original branch paths and runtime commit are not verified after migration.
echo "Historical launcher: inspect the source snapshot and create a new experiment record before rerunning." >&2
exit 2
set -euo pipefail

cd /data1/data8tb/liangyuxuan/code/uav
exec /home/lyx/miniconda3/envs/DMAD/bin/python \
  experiment_mask_DLBM/v001/launch.py --gpus 0,1 "$@"
