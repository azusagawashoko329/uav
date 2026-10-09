#!/usr/bin/env bash
# Fixed-action outer-objective audit for the completed v005 policies.
set -uo pipefail
ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SOURCE="$ROOT/experiments/da/boundary/source_snapshot_explore_1a68d0c"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${UAV_RUN_DIR:?Set UAV_RUN_DIR to a new experiment output directory}"
if [[ -e "$RUN_DIR/COMPLETE" || -d "$RUN_DIR/scenes" ]]; then echo "Existing run data: $RUN_DIR" >&2; exit 2; fi
PROBE_DIR="$ROOT/experiments/da/boundary/dlbm/probe/ckpt"
SCENES=(crossroads bike highway farmland vehicle solar railway); GPUS=(0 1)
QUEUE="$RUN_DIR/.audit_queue"; LOCK="$RUN_DIR/.audit_lock"; mkdir -p "$RUN_DIR/audit_logs"
printf '0\n' > "$QUEUE"; rm -f "$RUN_DIR/AUDIT_COMPLETE"
claim() { local i; exec 9>"$LOCK"; flock 9; i="$(<"$QUEUE")"; if ((i>=${#SCENES[@]})); then flock -u 9; return 1; fi; printf '%s\n' "$((i+1))" >"$QUEUE"; flock -u 9; printf '%s\n' "${SCENES[$i]}"; }
worker() { local gpu="$1" scene; while scene="$(claim)"; do
  "$PYTHON_BIN" "$SOURCE/experiment_DLBM/evaluate_boundary_v005.py" --scene "$scene" \
    --probe_dir "$PROBE_DIR" --out "$RUN_DIR/scenes/$scene/objective_audit.json" --gpu "$gpu" \
    --seed 42 --boundaries 0 4 8 11 --episodes 32 --support_bs 2 --query_bs 4 \
    --cache_bs 4 --workers 4 --max_frames_per_video 48 --topk 5 --inner_lr 5e-4 \
    > "$RUN_DIR/audit_logs/$scene.log" 2>&1 || return 1
done; }
worker "${GPUS[0]}" & p0=$!; worker "${GPUS[1]}" & p1=$!; wait "$p0"; s0=$?; wait "$p1"; s1=$?
((s0==0 && s1==0)) || exit 1
touch "$RUN_DIR/AUDIT_COMPLETE"
