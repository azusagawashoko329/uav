#!/usr/bin/env bash
# Train the seven reusable all-layer RD probes.  This stage never starts DLBM or final training.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lyx/miniconda3/envs/DMAD/bin/python}"
RUN_DIR="$ROOT/experiment_DLBM/run"
PROBE_DIR="$RUN_DIR/probe"
CKPT_DIR="$PROBE_DIR/ckpt"
LOG_DIR="$PROBE_DIR/logs"
STATUS_DIR="$PROBE_DIR/status"
# Longest-probe-first queue.  Workers claim scenes dynamically, so neither GPU owns a fixed subset.
SCENES=(crossroads farmland highway bike vehicle solar railway)
GPUS=(0 1)
QUEUE_INDEX="$PROBE_DIR/.queue_index"
QUEUE_LOCK="$PROBE_DIR/.queue_lock"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$STATUS_DIR" "$RUN_DIR/boundary"
printf '%s\n' \
    '{' \
    '  "stage": "probe",' \
    '  "scenes": ["bike", "crossroads", "farmland", "highway", "railway", "solar", "vehicle"],' \
    '  "gpus": [0, 1],' \
    '  "scheduling": "dynamic_longest_first",' \
    '  "seed": 42,' \
    '  "epochs": 10,' \
    '  "img": 256,' \
    '  "taps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],' \
    '  "code_dim": 64,' \
    '  "student_depth": 12,' \
    '  "tap_align": "absolute",' \
    '  "k": 5,' \
    '  "loss": "topk",' \
    '  "batch_size": 64,' \
    '  "learning_rate": 0.0005,' \
    '  "weight_decay": 0.05,' \
    '  "workers": 8' \
    '}' > "$PROBE_DIR/config.json"

run_scene() {
    local scene="$1"
    local gpu="$2"
    local log="$LOG_DIR/${scene}.log"
    rm -f "$STATUS_DIR/${scene}.done" "$STATUS_DIR/${scene}.failed"
    {
        echo "[$(date '+%F %T')] start scene=$scene gpu=$gpu seed=42 probe_epochs=10"
        "$PYTHON_BIN" "$ROOT/train.py" \
            --scene "$scene" --gpu "$gpu" --img 256 \
            --taps 0 1 2 3 4 5 6 7 8 9 10 11 \
            --code_dim 64 --depth 12 --tap_align absolute \
            --k 5 --loss topk --epochs 10 --bs 64 \
            --lr 5e-4 --wd 5e-2 --workers 8 --seed 42 \
            --out "$CKPT_DIR"
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
        if [[ -f "$STATUS_DIR/${scene}.done" && -s "$CKPT_DIR/${scene}.pth" ]]; then
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

echo "[$(date '+%F %T')] probe10 launch gpus=${GPUS[*]} queue=${SCENES[*]}"
worker "${GPUS[0]}" & worker0=$!
worker "${GPUS[1]}" & worker1=$!
wait "$worker0"; status0=$?
wait "$worker1"; status1=$?

failed=0
for scene in "${SCENES[@]}"; do
    if [[ ! -f "$STATUS_DIR/${scene}.done" || ! -s "$CKPT_DIR/${scene}.pth" ]]; then
        echo "[$(date '+%F %T')] incomplete scene=$scene"
        failed=1
    fi
done
if (( status0 != 0 || status1 != 0 || failed != 0 )); then
    echo "[$(date '+%F %T')] probe10 finished with failures"
    exit 1
fi
touch "$PROBE_DIR/COMPLETE"
echo "[$(date '+%F %T')] all seven probe checkpoints complete"
