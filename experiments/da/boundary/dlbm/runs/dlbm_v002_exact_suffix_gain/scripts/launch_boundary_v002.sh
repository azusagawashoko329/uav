#!/usr/bin/env bash
# Cache exact suffix responses once, then train seven independent DLBMs for 30 epochs.
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="DLBM_v002_exact_suffix_gain"
RUN_DIR="${UAV_RUN_DIR:?Set UAV_RUN_DIR to a new experiment output directory}"
if [[ -e "$RUN_DIR/COMPLETE" || -d "$RUN_DIR/scenes" ]]; then echo "Existing run data: $RUN_DIR" >&2; exit 2; fi
PROBE_DIR="$ROOT/experiments/da/boundary/dlbm/probe/ckpt"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
SCENE_DIR="$RUN_DIR/scenes"
SCENES=(crossroads farmland highway bike vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"
QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
rm -f "$RUN_DIR/COMPLETE"
printf '%s\n' \
    '{' \
    '  "version": "DLBM_v002_exact_suffix_gain",' \
    '  "stage": "boundary",' \
    '  "final_training": false,' \
    '  "independent_model_per_scene": true,' \
    '  "seed": 42,' \
    '  "gpus": [0, 1],' \
    '  "scheduling": "dynamic_longest_first",' \
    '  "probe": "experiments/da/boundary/dlbm/probe/ckpt",' \
    '  "topk": 5,' \
    '  "exact_suffix_supernet": true,' \
    '  "max_frames_per_video": 48,' \
    '  "cache_batch_size": 4,' \
    '  "pseudo": "within_frame_cutpaste",' \
    '  "dlbm": {"model_dim": 32, "layer_depth": 1, "pair_depth": 1, "heads": 4},' \
    '  "training": {"epochs": 30, "steps_per_epoch": 64, "episode_frames": 8, "validation_episodes": 64, "lr": 0.001, "weight_decay": 0.01},' \
    '  "milestones": [5, 10, 20, 30],' \
    '  "objective": "MSE predicted adjacent gain vs exact-suffix pairwise anomaly-score risk gain"' \
    '}' > "$RUN_DIR/config.json"

run_scene() {
    local scene="$1"
    local gpu="$2"
    local log="$LOG_DIR/${scene}.log"
    local out="$SCENE_DIR/$scene"
    rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"
    rm -rf "$out"
    {
        echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION epochs=30"
        "$PYTHON_BIN" "$SOURCE/experiment_DLBM/train_boundary_v002.py" \
            --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" \
            --gpu "$gpu" --seed 42 --topk 5 --max_frames_per_video 48 \
            --cache_bs 4 --workers 4 \
            --model_dim 32 --layer_depth 1 --pair_depth 1 --heads 4 \
            --epochs 30 --steps_per_epoch 64 --episode_frames 8 --validation_episodes 64 \
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
        if [[ -f "$STATUS_DIR/${scene}.done" && -s "$SCENE_DIR/$scene/result.json" ]]; then
            echo "[$(date '+%F %T')] gpu=$gpu skip completed $scene"
            continue
        fi
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
        echo "[$(date '+%F %T')] incomplete scene=$scene"
        failed=1
    fi
done
if (( status0 != 0 || status1 != 0 || failed != 0 )); then
    echo "[$(date '+%F %T')] $VERSION finished with failures"
    exit 1
fi
touch "$RUN_DIR/COMPLETE"
echo "[$(date '+%F %T')] all seven $VERSION runs complete"
