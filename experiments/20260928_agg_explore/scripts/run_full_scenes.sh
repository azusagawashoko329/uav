#!/bin/bash
# 把一个配置跑满 7 个场景(每场景合并全部 sequence 训一个权重, 与仓库口径一致)。
# 两张卡共享一个队列, 谁空谁领下一个场景; 结果按 variant 收集到一个 raw 文件里。
#
# 用法:  bash experiments/20260928_agg_explore/scripts/run_full_scenes.sh <variant> [extra probe args...]
# 例:    bash experiments/20260928_agg_explore/scripts/run_full_scenes.sh base12 --seed 42
#
# 产物:
#   experiments/20260928_agg_explore/raw_<variant>.txt    每场景一行 ">>> [variant] scene: AUC=..."
#   experiments/20260928_agg_explore/logs/<variant>_<scene>.log   完整训练日志
set -u
V="${1:?usage: run_full_scenes.sh <variant> [extra args]}"
shift || true
EXTRA=("$@")
PY="${PYTHON:-python}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UAV="$(cd "$EXP/../.." && pwd)"
OUT="$EXP/raw_$V.txt"
QUEUE="$EXP/.queue_$V"
LOCK="$EXP/.queue_$V.lock"
mkdir -p "$EXP/logs"
cd "$UAV" || exit 1

# 场景按训练帧数从多到少排(LPT), 让两张卡的收尾更齐
cat > "$QUEUE" << 'EOF'
crossroads
farmland
highway
bike
vehicle
solar
railway
EOF

pop_job() {
  exec 9>"$LOCK"; flock 9
  local job; job=$(head -n1 "$QUEUE" 2>/dev/null)
  [ -n "$job" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$job"
}

worker() {
  local gpu="$1" scene
  while scene=$(pop_job); [ -n "$scene" ]; do
    echo "[GPU$gpu] START $V/$scene  $(date +%H:%M:%S)"
    "$PY" "$UAV/experiments/20260928_agg_explore/scripts/probe_e2e.py" --gpu "$gpu" --variant "$V" --scenes "$scene" \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 \
      | tee "$EXP/logs/${V}_${scene}.log" \
      | grep '^>>>' >> "$OUT"
    echo "[GPU$gpu] DONE  $V/$scene  $(date +%H:%M:%S)"
  done
}

echo "=== $V START $(date) ===" >> "$OUT"
worker 0 & worker 1 & wait
rm -f "$QUEUE" "$LOCK"
echo "=== $V ALL DONE $(date) ===" >> "$OUT"
echo "results -> $OUT"
