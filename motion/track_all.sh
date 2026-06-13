#!/usr/bin/env bash
# Render BoT-SORT vehicle-tracking previews for all 7 scenes across 2 GPUs (dynamic, longest-first).
# Uses the DMAD conda env (only env with ultralytics). One mp4 per scene -> tracking_preview/.
set -u
cd /data1/data8tb/liangyuxuan/code/uav
PY=/home/lyx/miniconda3/envs/DMAD/bin/python
OUT=tracking_preview
mkdir -p "$OUT"
QUEUE=(bike crossroads farmland highway vehicle solar railway)   # longest-first → better 2-GPU balance
GPUS=(0 1)
declare -A PID ASSIGN
i=0
launch(){ local g=$1 s=$2; echo "[GPU$g] START $s $(date +%T)";
  "$PY" motion/visualize.py --scenes "$s" --gpu "$g" --out "$OUT" >"$OUT/$s.log" 2>&1 & PID[$g]=$!; ASSIGN[$g]=$s; }
for g in "${GPUS[@]}"; do (( i < ${#QUEUE[@]} )) && { launch "$g" "${QUEUE[$i]}"; ((i++)); }; done
while :; do
  busy=0
  for g in "${GPUS[@]}"; do
    pid=${PID[$g]:-}; [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then busy=1
    else wait "$pid"; echo "[GPU$g] DONE ${ASSIGN[$g]} rc=$? $(date +%T)"; PID[$g]=""
      (( i < ${#QUEUE[@]} )) && { launch "$g" "${QUEUE[$i]}"; ((i++)); busy=1; }
    fi
  done
  (( busy )) || break
  sleep 5
done
echo "=== tracking preview ALL DONE $(date +%T) ==="
ls -la "$OUT"/*.mp4
