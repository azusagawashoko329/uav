#!/usr/bin/env bash
# Scene-wise paired-relation DLBM with read-only Probe10 rollouts; no Final training.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="DLBM_v006_paired_rollout_relation"
RUN_DIR="${UAV_RUN_DIR:?Set UAV_RUN_DIR to a new experiment output directory}"
if [[ -e "$RUN_DIR/COMPLETE" || -d "$RUN_DIR/scenes" ]]; then echo "Existing run data: $RUN_DIR" >&2; exit 2; fi
PROBE_DIR="$ROOT/experiments/da/boundary/dlbm/probe/ckpt"
LOG_DIR="$RUN_DIR/logs"; STATUS_DIR="$RUN_DIR/status"; SCENE_DIR="$RUN_DIR/scenes"
SCENES=(crossroads bike highway farmland vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"; QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED_REVIEW"
sha256sum "$PROBE_DIR"/*.pth | sort -k2 > "$RUN_DIR/probe_sha256_before.txt"

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_DIR/${scene}.log" out="$SCENE_DIR/$scene"
  rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"
  rm -rf "$out"
  {
    echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION"
    "$PYTHON_BIN" "$SOURCE/experiment_DLBM/train_boundary_v006.py" \
      --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" --gpu "$gpu" --seed 42 \
      --candidate_layers 0 1 2 3 4 5 6 7 8 9 10 --topk 5 \
      --max_frames_per_video 48 --cache_bs 4 --workers 4 \
      --pseudo_min_area 0.02 --pseudo_max_area 0.15 \
      --target_episodes 8 --validation_episodes 2 --episode_frames 8 --query_frames 8 \
      --inner_steps 20 --inner_bs 2 --inner_lr 5e-4 --inner_wd 5e-2 \
      --model_dim 32 --heads 4 --frame_depth 1 \
      --epochs 50 --lr 1e-3 --wd 1e-2 --grad_clip 1.0 \
      --milestones 5 10 20 30 50
    code=$?
    echo "[$(date '+%F %T')] end scene=$scene exit=$code"
    return "$code"
  } > "$log" 2>&1
}

printf '0\n' > "$QUEUE_INDEX"
claim_scene() {
  local index
  exec 9> "$QUEUE_LOCK"
  flock 9
  index="$(<"$QUEUE_INDEX")"
  if (( index >= ${#SCENES[@]} )); then
    flock -u 9
    return 1
  fi
  printf '%s\n' "$((index + 1))" > "$QUEUE_INDEX"
  flock -u 9
  printf '%s\n' "${SCENES[$index]}"
}

worker() {
  local gpu="$1" scene
  echo "[$(date '+%F %T')] worker gpu=$gpu ready"
  while scene="$(claim_scene)"; do
    echo "[$(date '+%F %T')] gpu=$gpu running $scene"
    if run_scene "$scene" "$gpu"; then
      touch "$STATUS_DIR/${scene}.done"
      echo "[$(date '+%F %T')] gpu=$gpu completed $scene"
    else
      touch "$STATUS_DIR/${scene}.failed"
      echo "[$(date '+%F %T')] gpu=$gpu FAILED $scene"
    fi
  done
  echo "[$(date '+%F %T')] worker gpu=$gpu queue empty"
}

echo "[$(date '+%F %T')] launch version=$VERSION gpus=${GPUS[*]} queue=${SCENES[*]}"
worker "${GPUS[0]}" & worker0=$!
worker "${GPUS[1]}" & worker1=$!
wait "$worker0"; status0=$?
wait "$worker1"; status1=$?

failed=0
for scene in "${SCENES[@]}"; do
  if [[ ! -f "$STATUS_DIR/${scene}.done" || ! -s "$SCENE_DIR/$scene/result.json" ]]; then
    failed=1
  fi
done
if (( status0 != 0 || status1 != 0 || failed != 0 )); then
  exit 1
fi

sha256sum "$PROBE_DIR"/*.pth | sort -k2 > "$RUN_DIR/probe_sha256_after.txt"
cmp "$RUN_DIR/probe_sha256_before.txt" "$RUN_DIR/probe_sha256_after.txt"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
external = {'bike': 8, 'crossroads': 4, 'farmland': 8, 'highway': 0,
            'railway': 8, 'solar': 8, 'vehicle': 8}
rows = []
for path in sorted((root / 'scenes').glob('*/result.json')):
    data = json.loads(path.read_text())
    scene = data['scene']
    policy = data['final_policy']
    validation = data['validation']
    baseline = data['candidate_constant_baseline']
    row = {
        'scene': scene,
        'final_boundary': policy['boundary'],
        'selected_taps': ' '.join(map(str, policy['selected_taps'])),
        'validation_oracle_boundary': validation['oracle_boundary'],
        'validation_mse': validation['mse'],
        'constant_baseline_mse': baseline['mse'],
        'model_over_constant_mse_ratio': validation['mse'] / max(baseline['mse'], 1e-12),
        'validation_sign_accuracy': validation['sign_accuracy'],
        'paired_input_sensitivity': data['paired_input_sensitivity'],
        'external_coarse_gate_boundary': external[scene],
        'external_boundary_abs_error': abs(policy['boundary'] - external[scene]),
        'rollout_peak_gpu_gib': data['rollout_peak_gpu_gib'],
    }
    for epoch in ('5', '10', '20', '30', '50'):
        row[f'epoch{epoch}_boundary'] = data['milestones'][epoch]['boundary']
        row[f'epoch{epoch}_oracle'] = data['milestones'][epoch]['oracle_boundary']
    rows.append(row)
rows.sort(key=lambda row: row['scene'])
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
with (root / 'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY

touch "$RUN_DIR/COMPLETE"
echo "[$(date '+%F %T')] all seven $VERSION runs complete"

