#!/bin/bash
# DYNAMIC multi-GPU scheduler for the LAA appearance experiment. Scenes in ONE shared queue; each GPU
# worker pulls the next scene the moment it frees -> no card idles. Mirrors run_ablations.sh.
# Method: --tap_adapter --pseudo (variant B: all-12 concat into OCBE; learned per-scene layer weights w
# driven by the input-image pseudo-anomaly gap, z-score-standardized; top-k unchanged). Other hyperparams
# = train.py defaults (15 epoch, bs=64, img=256, code_dim=64, depth=12, k=5, AdamW lr=5e-4 wd=5e-2).
#
# Config via env:
#   BASE  experiment root dir          (default experiment_LAA)
#   EXP   experiment name <time>_<desc>(default <date_HHMM>_laa_zscore)
#   GPUS  space-separated CUDA UUIDs OR indices, ONE worker each (default "0 1")
# Usage e.g.: GPUS="GPU-35e64f7f-...." bash run_LAA.sh         # single free card by UUID
set -u
PY=/home/lyx/miniconda3/envs/DMAD/bin/python
UAV=/data1/data8tb/liangyuxuan/code/uav
cd "$UAV" || exit 1
export HF_HUB_OFFLINE=1
BASE=${BASE:-experiment_LAA}
EXP=${EXP:-$(date +%Y%m%d_%H%M)_laa_zscore}
GPUS=${GPUS:-0 1}
ARGS=${ARGS:-"--tap_adapter --pseudo --epochs 15"}
QUEUE="$UAV/experiments/.laa_queue"
LOCK="$UAV/experiments/.laa_queue.lock"

# 7 scenes, heaviest first (crossroads merges sequence1+2+3)
cat > "$QUEUE" <<EOF
crossroads
farmland
highway
bike
vehicle
solar
railway
EOF

pop_job() {                      # atomically read+remove the first queue line
  exec 9>"$LOCK"; flock 9
  local job; job=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$job"
}

worker() {                       # $1 = CUDA device (UUID or index) -> isolated as this proc's cuda:0
  local dev="$1" scene rc
  mkdir -p "$BASE/$EXP/logs" "$BASE/$EXP/ckpt"
  while scene=$(pop_job); [ -n "$scene" ]; do
    echo "[$dev] START $scene  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$dev" $PY "$UAV/train.py" --scene "$scene" --gpu 0 --base "$BASE" --exp "$EXP" $ARGS \
        > "$BASE/$EXP/logs/${scene}.log" 2>&1
    rc=$?
    echo "[$dev] DONE  $scene rc=$rc  $(date +%H:%M:%S)"
  done
  echo "[$dev] queue empty, exit  $(date +%H:%M:%S)"
}

echo "=== LAA START $(date) | BASE=$BASE EXP=$EXP | GPUS=[$GPUS] | $ARGS | dynamic shared queue ==="
for g in $GPUS; do worker "$g" & done
wait
rm -f "$QUEUE" "$LOCK"
echo "=== LAA ALL DONE $(date) ==="
