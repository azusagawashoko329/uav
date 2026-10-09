#!/usr/bin/env bash
# Seven independent per-scene counterfactual-gain probes.  Does not run Final training.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyx/miniconda3/envs/DMAD/bin/python}"
VERSION="DLBM_v001_topk_counterfactual"
RUN_DIR="$ROOT/experiment_DLBM/run/boundary/$VERSION"
PROBE_DIR="$ROOT/experiment_DLBM/run/probe/ckpt"
LOG_DIR="$RUN_DIR/logs"
STATUS_DIR="$RUN_DIR/status"
SCENE_DIR="$RUN_DIR/scenes"
SCENES=(crossroads farmland highway bike vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$RUN_DIR/.queue_index"
QUEUE_LOCK="$RUN_DIR/.queue_lock"

mkdir -p "$LOG_DIR" "$STATUS_DIR" "$SCENE_DIR"
rm -rf "$RUN_DIR/smoke"
rm -f "$RUN_DIR/COMPLETE"
printf '%s\n' \
    '{' \
    '  "version": "DLBM_v001_topk_counterfactual",' \
    '  "stage": "boundary_gain_probe",' \
    '  "final_training": false,' \
    '  "independent_model_per_scene": true,' \
    '  "seed": 42,' \
    '  "gpus": [0, 1],' \
    '  "scheduling": "dynamic_longest_first",' \
    '  "probe": "experiment_DLBM/run/probe/ckpt",' \
    '  "topk": 5,' \
    '  "max_frames_per_video": 48,' \
    '  "pseudo": "within_frame_cutpaste",' \
    '  "evaluator": {"model_dim": 32, "depth": 1, "heads": 4, "epochs": 40, "lr": 0.001, "weight_decay": 0.01},' \
    '  "gain": {"type": "first_order_counterfactual", "episodes": 4, "support_batch": 3, "query_batch": 3, "inner_lr": 0.0005},' \
    '  "fast_parameters": "ocbe+student.proj_in+student.heads",' \
    '  "exact_validation_candidates": [0, 4, 8, 10]' \
    '}' > "$RUN_DIR/config.json"

run_scene() {
    local scene="$1"
    local gpu="$2"
    local log="$LOG_DIR/${scene}.log"
    local out="$SCENE_DIR/$scene"
    rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"
    rm -rf "$out"
    {
        echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu version=$VERSION"
        "$PYTHON_BIN" "$ROOT/experiment_DLBM/run_boundary_v001.py" \
            --scene "$scene" --probe_dir "$PROBE_DIR" --out_dir "$out" \
            --gpu "$gpu" --seed 42 --topk 5 --max_frames_per_video 48 \
            --cache_bs 32 --workers 4 \
            --evaluator_dim 32 --evaluator_depth 1 --evaluator_heads 4 \
            --evaluator_epochs 40 --evaluator_bs 64 --evaluator_lr 1e-3 --evaluator_wd 1e-2 \
            --episodes 4 --support_bs 3 --query_bs 3 --inner_lr 5e-4 \
            --exact_candidates 0 4 8 10
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
echo "[$(date '+%F %T')] all seven $VERSION gain probes complete"
