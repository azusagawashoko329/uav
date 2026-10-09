#!/usr/bin/env bash
# Learned patch-localization target probe on four micro-adapted coarse suffixes; no Final training.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="DLBM_v004_mask_localization_target_probe"
RUN_DIR="${UAV_RUN_DIR:?Set UAV_RUN_DIR to a new experiment output directory}"
if [[ -e "$RUN_DIR/COMPLETE" || -d "$RUN_DIR/scenes" ]]; then echo "Existing run data: $RUN_DIR" >&2; exit 2; fi
PROBE_DIR="$ROOT/experiments/da/boundary/dlbm/probe/ckpt"
LOG_DIR="$RUN_DIR/logs"; STATUS_DIR="$RUN_DIR/status"; SCENE_DIR="$RUN_DIR/scenes"
SCENES=(crossroads farmland highway bike vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"; QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
rm -f "$RUN_DIR/COMPLETE" "$RUN_DIR/FAILED_TARGET_REVIEW"
printf '%s\n' \
  '{' \
  '  "version": "DLBM_v004_mask_localization_target_probe",' \
  '  "purpose": "validate learned spatial opening target before DLBM fitting",' \
  '  "final_training": false,' \
  '  "seed": 42,' \
  '  "gpus": [0, 1],' \
  '  "boundaries": [0, 4, 8, 11],' \
  '  "micro_epochs": 5,' \
  '  "localizer_epochs": 20,' \
  '  "localizer_milestones": [5, 10, 20],' \
  '  "steps_per_epoch": 64,' \
  '  "max_frames_per_video": 48,' \
  '  "batch_size": 8,' \
  '  "probe_learning_rate": 0.0005,' \
  '  "probe_weight_decay": 0.05,' \
  '  "localizer_learning_rate": 0.001,' \
  '  "localizer_weight_decay": 0.01,' \
  '  "topk": 5' \
  '}' > "$RUN_DIR/config.json"

run_scene() {
  local scene="$1" gpu="$2" log="$LOG_DIR/${scene}.log" out="$SCENE_DIR/$scene"
  rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"; rm -rf "$out"
  {
    echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION"
    "$PYTHON_BIN" "$SOURCE/experiment_DLBM/mask_localization_targets_v004.py" \
      --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" --gpu "$gpu" --seed 42 \
      --boundaries 0 4 8 11 --micro_epochs 5 --localizer_epochs 20 --milestones 5 10 20 \
      --steps_per_epoch 64 --max_frames_per_video 48 --train_bs 8 --eval_bs 8 \
      --localizer_bs 8 --workers 4 --topk 5 --lr 5e-4 --wd 5e-2 \
      --localizer_lr 1e-3 --localizer_wd 1e-2 --model_dim 16 --localizer_depth 1 --heads 4
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
    row = {'scene': data['scene'], 'best_boundary': data['best_boundary'],
           'coarse_gate_boundary_external_only': external[data['scene']],
           'coarse_match': data['best_boundary'] == external[data['scene']],
           'peak_gpu_gib': data['peak_gpu_gib']}
    for epoch in ('5', '10', '20'):
        metrics = data['milestone_metrics'][epoch]
        choice = max((0, 4, 8, 11), key=lambda b: metrics[str(b)]['patch_auc'])
        row[f'epoch{epoch}_boundary'] = choice
    for boundary in (0, 4, 8, 11):
        row[f'L{boundary}_patch_auc'] = data['final_metrics'][str(boundary)]['patch_auc']
        row[f'L{boundary}_soft_bce'] = data['final_metrics'][str(boundary)]['soft_bce']
    rows.append(row)
rows.sort(key=lambda row: row['scene'])
(root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
with (root / 'summary.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
PY
touch "$RUN_DIR/COMPLETE"
echo "[$(date '+%F %T')] all seven $VERSION runs complete"
