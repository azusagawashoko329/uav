#!/bin/bash
# DYNAMIC 2-GPU scheduler for the appearance-branch ABLATIONS: 2 exps x 7 scenes = 14 jobs in ONE
# shared queue; each GPU pulls the next job the moment it frees -> no GPU ever idles across BOTH exps.
# Queue line format: <exp>|<scene>|<extra train.py args>.  Usage: bash run_ablations.sh
set -u
PY="${PYTHON:-python}"
UAV="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$UAV" || exit 1
for exp in da/appearance/20260612_rd_alltap da/appearance/20260612_rd_notopk; do
  if [[ -e "experiments/$exp/results.csv" ]] || find "experiments/$exp/ckpt" -type f -name '*.pth' -print -quit 2>/dev/null | grep -q .; then
    echo "Existing experiment evidence at experiments/$exp; run each new variant under a new ID" >&2
    exit 2
  fi
done
QUEUE="$UAV/experiments/da/appearance/ablations/.ablation_queue"
LOCK="$UAV/experiments/da/appearance/ablations/.ablation_queue.lock"
ALLTAP="--taps 0 1 2 3 4 5 6 7 8 9 10 11"     # 消融1: 蒸馏全部 12 层 (vs 选 [5,7,9,11])
NOTOPK="--loss full"                          # 消融2: 全 patch 均值余弦 (vs top-k 难例)

# 14 jobs, heaviest scene first, two exps interleaved
cat > "$QUEUE" <<EOF
da/appearance/20260612_rd_alltap|crossroads|$ALLTAP
da/appearance/20260612_rd_notopk|crossroads|$NOTOPK
da/appearance/20260612_rd_alltap|farmland|$ALLTAP
da/appearance/20260612_rd_notopk|farmland|$NOTOPK
da/appearance/20260612_rd_alltap|highway|$ALLTAP
da/appearance/20260612_rd_notopk|highway|$NOTOPK
da/appearance/20260612_rd_alltap|bike|$ALLTAP
da/appearance/20260612_rd_notopk|bike|$NOTOPK
da/appearance/20260612_rd_alltap|vehicle|$ALLTAP
da/appearance/20260612_rd_notopk|vehicle|$NOTOPK
da/appearance/20260612_rd_alltap|solar|$ALLTAP
da/appearance/20260612_rd_notopk|solar|$NOTOPK
da/appearance/20260612_rd_alltap|railway|$ALLTAP
da/appearance/20260612_rd_notopk|railway|$NOTOPK
EOF

pop_job() {                      # atomically read+remove the first queue line
  exec 9>"$LOCK"; flock 9
  local job; job=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$job"
}

worker() {
  local gpu="$1" job rest exp scene args rc failed=0
  while job=$(pop_job); [ -n "$job" ]; do
    exp=${job%%|*}; rest=${job#*|}; scene=${rest%%|*}; args=${rest#*|}
    mkdir -p "experiments/$exp/logs" "experiments/$exp/ckpt"
    echo "[GPU$gpu] START $exp/$scene  $(date +%H:%M:%S)"
    $PY "$UAV/train.py" --scene "$scene" --gpu "$gpu" --exp "$exp" $args > "experiments/$exp/logs/${scene}.log" 2>&1
    rc=$?
    echo "[GPU$gpu] DONE  $exp/$scene rc=$rc  $(date +%H:%M:%S)"
    if (( rc != 0 )); then failed=1; fi
  done
  echo "[GPU$gpu] queue empty, exit  $(date +%H:%M:%S)"
  return "$failed"
}

echo "=== ablations START $(date) | 14 jobs (alltap + notopk x 7) | dynamic 2-GPU shared queue ==="
worker 0 & pid0=$!
worker 1 & pid1=$!
wait "$pid0"; status0=$?
wait "$pid1"; status1=$?
rm -f "$QUEUE" "$LOCK"
if (( status0 != 0 || status1 != 0 )); then
  echo "=== ablations FAILED $(date) ===" >&2
  exit 1
fi
echo "=== ablations ALL DONE $(date) ==="
