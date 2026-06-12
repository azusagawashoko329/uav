#!/bin/bash
# DYNAMIC 2-GPU scheduler for the appearance-branch ABLATIONS: 2 exps x 7 scenes = 14 jobs in ONE
# shared queue; each GPU pulls the next job the moment it frees -> no GPU ever idles across BOTH exps.
# Queue line format: <exp>|<scene>|<extra train.py args>.  Usage: bash run_ablations.sh
set -u
PY=/home/lyx/miniconda3/envs/DMAD/bin/python
UAV=/data1/data8tb/liangyuxuan/code/uav
cd "$UAV" || exit 1
QUEUE="$UAV/experiments/.ablation_queue"
LOCK="$UAV/experiments/.ablation_queue.lock"
ALLTAP="--taps 0 1 2 3 4 5 6 7 8 9 10 11"     # 消融1: 蒸馏全部 12 层 (vs 选 [5,7,9,11])
NOTOPK="--loss full"                          # 消融2: 全 patch 均值余弦 (vs top-k 难例)

# 14 jobs, heaviest scene first, two exps interleaved
cat > "$QUEUE" <<EOF
20260612_rd_alltap|crossroads|$ALLTAP
20260612_rd_notopk|crossroads|$NOTOPK
20260612_rd_alltap|farmland|$ALLTAP
20260612_rd_notopk|farmland|$NOTOPK
20260612_rd_alltap|highway|$ALLTAP
20260612_rd_notopk|highway|$NOTOPK
20260612_rd_alltap|bike|$ALLTAP
20260612_rd_notopk|bike|$NOTOPK
20260612_rd_alltap|vehicle|$ALLTAP
20260612_rd_notopk|vehicle|$NOTOPK
20260612_rd_alltap|solar|$ALLTAP
20260612_rd_notopk|solar|$NOTOPK
20260612_rd_alltap|railway|$ALLTAP
20260612_rd_notopk|railway|$NOTOPK
EOF

pop_job() {                      # atomically read+remove the first queue line
  exec 9>"$LOCK"; flock 9
  local job; job=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$job"
}

worker() {
  local gpu="$1" job rest exp scene args rc
  while job=$(pop_job); [ -n "$job" ]; do
    exp=${job%%|*}; rest=${job#*|}; scene=${rest%%|*}; args=${rest#*|}
    mkdir -p "experiments/$exp/logs" "experiments/$exp/ckpt"
    echo "[GPU$gpu] START $exp/$scene  $(date +%H:%M:%S)"
    $PY "$UAV/train.py" --scene "$scene" --gpu "$gpu" --exp "$exp" $args > "experiments/$exp/logs/${scene}.log" 2>&1
    rc=$?
    echo "[GPU$gpu] DONE  $exp/$scene rc=$rc  $(date +%H:%M:%S)"
  done
  echo "[GPU$gpu] queue empty, exit  $(date +%H:%M:%S)"
}

echo "=== ablations START $(date) | 14 jobs (alltap + notopk x 7) | dynamic 2-GPU shared queue ==="
worker 0 & worker 1 & wait
rm -f "$QUEUE" "$LOCK"
echo "=== ablations ALL DONE $(date) ==="
