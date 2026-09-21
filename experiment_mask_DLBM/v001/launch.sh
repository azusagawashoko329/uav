#!/usr/bin/env bash
set -euo pipefail

cd /data1/data8tb/liangyuxuan/code/uav
exec /home/lyx/miniconda3/envs/DMAD/bin/python \
  experiment_mask_DLBM/v001/launch.py --gpus 0,1 "$@"
