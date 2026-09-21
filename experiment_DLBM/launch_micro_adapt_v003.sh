#!/usr/bin/env bash
# Coarse L0/L4/L8/L11 fixed-suffix micro-adaptation target validation; no Final training.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyx/miniconda3/envs/DMAD/bin/python}"
VERSION="DLBM_v003_micro_adapt_target_probe"
RUN_DIR="$ROOT/experiment_DLBM/run/boundary/$VERSION"
PROBE_DIR="$ROOT/experiment_DLBM/run/probe/ckpt"
LOG_DIR="$RUN_DIR/logs"; STATUS_DIR="$RUN_DIR/status"; SCENE_DIR="$RUN_DIR/scenes"
SCENES=(crossroads farmland highway bike vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"; QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
rm -f "$RUN_DIR/COMPLETE"
printf '%s\n' \
  '{' \
  '  "version": "DLBM_v003_micro_adapt_target_probe",' \
  '  "purpose": "validate training-aware boundary targets before DLBM fitting",' \
  '  "final_training": false,' \
  '  "seed": 42,' \
  '  "gpus": [0, 1],' \
  '  "boundaries": [0, 4, 8, 11],' \
  '  "micro_epochs": 5,' \
  '  "milestones": [0, 1, 3, 5],' \
  '  "max_frames_per_video": 48,' \
  '  "batch_size": 8,' \
  '  "learning_rate": 0.0005,' \
  '  "weight_decay": 0.05,' \
  '  "topk": 5' \
  '}' > "$RUN_DIR/config.json"

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_DIR/${scene}.log" out="$SCENE_DIR/$scene"
  rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"; rm -rf "$out"
  {
    echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION"
    "$PYTHON_BIN" "$ROOT/experiment_DLBM/micro_adapt_targets_v003.py" \
      --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" --gpu "$gpu" --seed 42 \
      --boundaries 0 4 8 11 --epochs 5 --milestones 1 3 5 --max_frames_per_video 48 \
      --train_bs 8 --eval_bs 8 --workers 4 --topk 5 --lr 5e-4 --wd 5e-2
    code=$?; echo "[$(date '+%F %T')] end scene=$scene exit=$code"; return "$code"
  } > "$log" 2>&1
}

printf '0\n' > "$QUEUE_INDEX"
claim_scene() {
  local index; exec 9> "$QUEUE_LOCK"; flock 9; index="$(<"$QUEUE_INDEX")"
  if (( index >= ${#SCENES[@]} )); then flock -u 9; return 1; fi
  printf '%s\n' "$((index + 1))" > "$QUEUE_INDEX"; flock -u 9; printf '%s\n' "${SCENES[$index]}"
}
worker() {
  local gpu="$1" scene; echo "[$(date '+%F %T')] worker gpu=$gpu ready"
  while scene="$(claim_scene)"; do
    echo "[$(date '+%F %T')] gpu=$gpu running $scene"
    if run_scene "$scene" "$gpu"; then touch "$STATUS_DIR/${scene}.done"; echo "[$(date '+%F %T')] gpu=$gpu completed $scene"
    else touch "$STATUS_DIR/${scene}.failed"; echo "[$(date '+%F %T')] gpu=$gpu FAILED $scene"; fi
  done
  echo "[$(date '+%F %T')] worker gpu=$gpu queue empty"
}

echo "[$(date '+%F %T')] launch version=$VERSION gpus=${GPUS[*]} queue=${SCENES[*]}"
worker "${GPUS[0]}" & worker0=$!; worker "${GPUS[1]}" & worker1=$!
wait "$worker0"; status0=$?; wait "$worker1"; status1=$?
failed=0
for scene in "${SCENES[@]}"; do
  if [[ ! -f "$STATUS_DIR/${scene}.done" || ! -s "$SCENE_DIR/$scene/result.json" ]]; then failed=1; fi
done
if (( status0 != 0 || status1 != 0 || failed != 0 )); then exit 1; fi
touch "$RUN_DIR/COMPLETE"; echo "[$(date '+%F %T')] all seven $VERSION runs complete"
