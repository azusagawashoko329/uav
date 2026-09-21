#!/usr/bin/env bash
# Scene-wise GDAS boundary search with one-step exact hyper-gradients; no Final training.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyx/miniconda3/envs/DMAD/bin/python}"
VERSION="DLBM_v005_gdas_bilevel_rank"
RUN_DIR="$ROOT/experiment_DLBM/run/boundary/$VERSION"
PROBE_DIR="$ROOT/experiment_DLBM/run/probe/ckpt"
LOG_DIR="$RUN_DIR/logs"; STATUS_DIR="$RUN_DIR/status"; SCENE_DIR="$RUN_DIR/scenes"
SCENES=(crossroads bike highway farmland vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"; QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED_REVIEW"
printf '%s\n' \
  '{' \
  '  "version": "DLBM_v005_gdas_bilevel_rank",' \
  '  "stage": "boundary",' \
  '  "final_training": false,' \
  '  "independent_distribution_per_scene": true,' \
  '  "seed": 42,' \
  '  "gpus": [0, 1],' \
  '  "probe": "experiment_DLBM/run/probe/ckpt",' \
  '  "probe_mutated": false,' \
  '  "boundaries": [0, 4, 8, 11],' \
  '  "epochs": 50,' \
  '  "steps_per_epoch": 8,' \
  '  "hypergradient_updates": 400,' \
  '  "support_batch_size": 2,' \
  '  "query_batch_size": 4,' \
  '  "inner_learning_rate": 0.0005,' \
  '  "dlbm_learning_rate": 0.05,' \
  '  "gumbel_temperature": 1.0,' \
  '  "topk": 5,' \
  '  "outer_objective": "RankNet ordering of two within-frame CutPaste views by realized area",' \
  '  "milestones": [5, 10, 20, 30, 50]' \
  '}' > "$RUN_DIR/config.json"

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_DIR/${scene}.log" out="$SCENE_DIR/$scene"
  rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"; rm -rf "$out"
  {
    echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION"
    "$PYTHON_BIN" "$ROOT/experiment_DLBM/train_boundary_v005.py" \
      --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" --gpu "$gpu" --seed 42 \
      --boundaries 0 4 8 11 --epochs 50 --steps_per_epoch 8 \
      --support_bs 2 --query_bs 4 --cache_bs 4 --workers 4 --max_frames_per_video 48 \
      --topk 5 --inner_lr 5e-4 --dlbm_lr 5e-2 --temperature 1.0 --grad_clip 5.0 \
      --milestones 5 10 20 30 50 --pseudo_min_area 0.02 --pseudo_max_area 0.15
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

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
external = {'bike': 8, 'crossroads': 4, 'farmland': 8, 'highway': 0,
            'railway': 8, 'solar': 8, 'vehicle': 8}
rows = []
for path in sorted((root / 'scenes').glob('*/result.json')):
    data = json.loads(path.read_text())
    policy = data['final_policy']; scene = data['scene']
    row = {'scene': scene, 'final_boundary': policy['boundary'],
           'confidence': policy['confidence'], 'normalized_entropy': policy['normalized_entropy'],
           'coarse_gate_boundary_external_only': external[scene],
           'coarse_match': policy['boundary'] == external[scene],
           'peak_gpu_gib': data['peak_gpu_gib']}
    for boundary in (0, 4, 8, 11):
        row[f'L{boundary}_probability'] = policy['probabilities'][str(boundary)]
        row[f'L{boundary}_samples'] = data['sample_counts'][str(boundary)]
    for epoch in ('5', '10', '20', '30', '50'):
        milestone = data['milestones'][epoch]
        row[f'epoch{epoch}_boundary'] = milestone['policy']['boundary']
        row[f'epoch{epoch}_confidence'] = milestone['policy']['confidence']
    rows.append(row)
rows.sort(key=lambda row: row['scene'])
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
with (root / 'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
PY
touch "$RUN_DIR/COMPLETE"
echo "[$(date '+%F %T')] all seven $VERSION runs complete"
