#!/usr/bin/env bash
# Three-scene video-crossfit qualification; Probe10 is read-only and Final is not run.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="DLBM_v016_crossfit_order_novelty"
RUN_DIR="${UAV_RUN_DIR:?Set UAV_RUN_DIR to a new experiment output directory}"
if [[ -e "$RUN_DIR/COMPLETE" || -d "$RUN_DIR/scenes" ]]; then echo "Existing run data: $RUN_DIR" >&2; exit 2; fi
PROBE_DIR="$ROOT/experiments/da/boundary/dlbm/probe/ckpt"
PREDICTOR_DIR="$ROOT/experiments/da/boundary/dlbm/diagnostics/_architecture_study"
LOG_DIR="$RUN_DIR/logs"; STATUS_DIR="$RUN_DIR/status"; SCENE_DIR="$RUN_DIR/scenes"
SCENES=(bike crossroads highway); GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"; QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
printf '%s\n' "$$" > "$RUN_DIR/run.pid"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED_REVIEW"
sha256sum "$PROBE_DIR"/*.pth | sort -k2 > "$RUN_DIR/probe_sha256_before.txt"

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_DIR/${scene}.log" out="$SCENE_DIR/$scene"
  local predictor="$PREDICTOR_DIR/aligned_convgru_${scene}/predictor.pth"
  rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"; rm -rf "$out"
  {
    echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION"
    "$PYTHON_BIN" "$SOURCE/experiment_DLBM/train_crossfit_order_novelty.py" \
      --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" \
      --gpu "$gpu" --seed 42 --predictor_checkpoint "$predictor" \
      --topk 5 --max_frames_per_video 48 --cache_bs 4 --workers 4 \
      --pseudo_min_area 0.02 --pseudo_max_area 0.15 --pseudo_mode multi \
      --channels 16 --model_dim 32 --heads 4 \
      --epochs 30 --steps_per_epoch 30 --episode_frames 32 --pair_samples 256 \
      --eval_pairs 2048 --eval_nulls 16 --lr 1e-3 --wd 1e-2 --grad_clip 1.0 \
      --milestones 5 10 20 30
    code=$?; echo "[$(date '+%F %T')] end scene=$scene exit=$code"; return "$code"
  } > "$log" 2>&1
}

printf '0\n' > "$QUEUE_INDEX"
claim_scene() {
  local index; exec 9> "$QUEUE_LOCK"; flock 9; index="$(<"$QUEUE_INDEX")"
  if (( index >= ${#SCENES[@]} )); then flock -u 9; return 1; fi
  printf '%s\n' "$((index + 1))" > "$QUEUE_INDEX"; flock -u 9
  printf '%s\n' "${SCENES[$index]}"
}
worker() {
  local gpu="$1" scene; echo "[$(date '+%F %T')] worker gpu=$gpu ready"
  while scene="$(claim_scene)"; do
    echo "[$(date '+%F %T')] gpu=$gpu running $scene"
    if run_scene "$scene" "$gpu"; then touch "$STATUS_DIR/${scene}.done"; else touch "$STATUS_DIR/${scene}.failed"; fi
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
sha256sum "$PROBE_DIR"/*.pth | sort -k2 > "$RUN_DIR/probe_sha256_after.txt"
if ! cmp "$RUN_DIR/probe_sha256_before.txt" "$RUN_DIR/probe_sha256_after.txt"; then failed=1; fi
if (( status0 != 0 || status1 != 0 || failed != 0 )); then touch "$RUN_DIR/FAILED_REVIEW"; exit 1; fi
touch "$RUN_DIR/COMPLETE"
echo "[$(date '+%F %T')] three-scene $VERSION qualification complete"
