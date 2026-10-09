#!/usr/bin/env bash
# Seven independent scene-wise DLBM v008 runs; Probe10 is read-only and Final is not trained.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="DLBM_v008_neural_opening_order"
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
    "$PYTHON_BIN" "$SOURCE/experiment_DLBM/train_boundary_v008.py" \
      --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" --gpu "$gpu" --seed 42 \
      --topk 5 --max_frames_per_video 48 --cache_bs 4 --workers 4 \
      --pseudo_min_area 0.02 --pseudo_max_area 0.15 \
      --model_dim 32 --heads 4 --epochs 30 --steps_per_epoch 44 \
      --frame_batch 48 --pair_samples 512 --eval_pairs 4096 --max_nulls 16 \
      --lr 1e-3 --wd 1e-2 --grad_clip 1.0 --milestones 5 10 20 30
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
    query = data['final_policy']
    support = data['support_policy']
    row = {
        'scene': scene,
        'query_boundary': query['boundary'],
        'support_boundary': support['boundary'],
        'query_M_gain': query['middle']['normal_gain'],
        'query_S_gain': query['shallow']['normal_gain'],
        'actual_query_M_gain': query['middle']['actual_normal_gain'],
        'actual_query_S_gain': query['shallow']['actual_normal_gain'],
        'query_M_pseudo_gain': query['middle']['pseudo_gain'],
        'query_S_pseudo_gain': query['shallow']['pseudo_gain'],
        'external_coarse_gate_boundary': external[scene],
        'external_match': query['boundary'] == external[scene],
        'peak_gpu_gib': data['peak_gpu_gib'],
    }
    for epoch in ('5', '10', '20', '30'):
        row[f'epoch{epoch}_boundary'] = data['milestones'][epoch]['boundary']
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
