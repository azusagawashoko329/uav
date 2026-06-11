#!/bin/bash
# DYNAMIC 2-GPU scheduler: one shared job queue; each GPU pulls the next scene the moment it is free.
# No GPU ever idles while scenes remain. Scenes ordered heaviest-first (LPT) for a balanced tail.
# PER-SCENE protocol (MADViT): each scene merges all its sequences into one model (7 scenes total).
# Usage: bash run_experiment.sh <exp_name>
set -u
EXP="${1:-20260611_rd_baseline}"
PY=/home/lyx/miniconda3/envs/DMAD/bin/python
UAV=/data1/data8tb/liangyuxuan/code/uav
EXPDIR="$UAV/experiments/$EXP"
LOG="$EXPDIR/logs"
QUEUE="$EXPDIR/.queue"
LOCK="$EXPDIR/.queue.lock"
mkdir -p "$LOG" "$EXPDIR/ckpt"

# all 7 scenes, heaviest first (by merged train-frame count)
cat > "$QUEUE" << 'EOF'
crossroads
farmland
highway
bike
vehicle
solar
railway
EOF

pop_job() {                      # atomically read+remove the first queue line; echo it (empty if none)
  exec 9>"$LOCK"; flock 9
  local job; job=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$job"
}

worker() {                       # a GPU keeps pulling scenes until the queue is empty
  local gpu="$1" scene rc
  while scene=$(pop_job); [ -n "$scene" ]; do
    echo "[GPU$gpu] START $scene  $(date +%H:%M:%S)"
    "$PY" "$UAV/train.py" --scene "$scene" --gpu "$gpu" --exp "$EXP" \
        > "$LOG/${scene}.log" 2>&1
    rc=$?
    echo "[GPU$gpu] DONE  $scene rc=$rc  $(date +%H:%M:%S)"
  done
  echo "[GPU$gpu] queue empty, worker exit  $(date +%H:%M:%S)"
}

echo "=== experiment $EXP START $(date) | dynamic 2-GPU shared queue (7 scenes, per-scene merge) ==="
worker 0 &
worker 1 &
wait
rm -f "$QUEUE" "$LOCK"
echo "=== experiment $EXP ALL DONE $(date) ==="
