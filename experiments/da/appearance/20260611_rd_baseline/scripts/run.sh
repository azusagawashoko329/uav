#!/bin/bash
# DYNAMIC 2-GPU scheduler: one shared job queue; each GPU pulls the next scene the moment it is free.
# No GPU ever idles while scenes remain. Scenes ordered heaviest-first (LPT) for a balanced tail.
# PER-SCENE protocol (MADViT): each scene merges all its sequences into one model (7 scenes total).
# Usage: bash experiments/da/appearance/20260611_rd_baseline/scripts/run.sh [extra train.py args ...]
set -u
EXP="${UAV_RUN_EXP:-da/appearance/20260611_rd_baseline}"
EXTRA=("$@")                      # forwarded verbatim to every train.py call
PY="${PYTHON:-python}"
UAV="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$UAV" || exit 1              # train.py writes ckpt to relative experiments/<exp>/ckpt
EXPDIR="$UAV/experiments/$EXP"
if [[ -e "$EXPDIR/results.csv" ]] || find "$EXPDIR/ckpt" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
  echo "Existing experiment evidence at $EXPDIR; create a new experiment ID and set UAV_RUN_EXP" >&2
  exit 2
fi
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
  local gpu="$1" scene rc failed=0
  while scene=$(pop_job); [ -n "$scene" ]; do
    echo "[GPU$gpu] START $scene  $(date +%H:%M:%S)"
    "$PY" "$UAV/train.py" --scene "$scene" --gpu "$gpu" --exp "$EXP" "${EXTRA[@]+"${EXTRA[@]}"}" \
        > "$LOG/${scene}.log" 2>&1
    rc=$?
    echo "[GPU$gpu] DONE  $scene rc=$rc  $(date +%H:%M:%S)"
    if (( rc != 0 )); then failed=1; fi
  done
  echo "[GPU$gpu] queue empty, worker exit  $(date +%H:%M:%S)"
  return "$failed"
}

echo "=== experiment $EXP START $(date) | dynamic 2-GPU shared queue (7 scenes, per-scene merge) ==="
worker 0 & pid0=$!
worker 1 & pid1=$!
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
rm -f "$QUEUE" "$LOCK"
if (( status0 != 0 || status1 != 0 )); then
  echo "=== experiment $EXP FAILED $(date) ===" >&2
  exit 1
fi
echo "=== experiment $EXP ALL DONE $(date) ==="
