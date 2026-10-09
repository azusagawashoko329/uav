# Mask DLBM experiment

The normal training frames are partitioned once, independently inside every subvideo, with seed 42:

- `probe_train`: two of four shuffled folds (approximately 1/2);
- `boundary_train`: one shuffled fold (approximately 1/4);
- `boundary_decision`: the remaining shuffled fold (approximately 1/4).

The three subsets are disjoint and cover every normal training frame. Generated manifests live in
`splits/seed42/<scene>/`; `split_summary.json` records counts and SHA-256 fingerprints. Phase two must
consume these same manifests instead of splitting frames again.

Probe checkpoints are written to `run/probe/ckpt/`, with per-scene logs and status records beside
them. `launch_probe10.sh` dynamically schedules the largest remaining scene onto the next free GPU.
`audit_probe.py` loads every completed checkpoint and verifies its fixed-manifest fingerprint,
training contract, epoch log, state layout, and tensor finiteness before phase two reuses it.

## Version layout

Every DLBM iteration owns its implementation, entry points, tests, and review:

```text
v001/
  dlbm.py
  train.py
  launch.py
  launch.sh
  audit.py
  tests/test_dlbm.py
  REVIEW.md
```

Shared data partitions, Probe10 code, and reusable Probe checkpoints remain at the experiment root.
Versioned runtime products remain under `run/dlbm/<version>/`, separate from source code.

## DLBM v001

`v001_relation_changepoint` is the first masked-relation boundary implementation.  For each scene it
uses the frozen Probe to obtain all twelve Teacher/Student patch pairs, predicts every shallower
cosine-distance map from the already-observed deeper suffix, and encodes the learned opening
transition as 256 aligned local tokens plus one global token per layer.  A frozen ordered two-regime
density reads both scales and returns one boundary `B`, therefore the exact suffix `[B, ..., L11]`.

The 1/4 `boundary_train` manifest is split deterministically inside each subvideo into 80% fit and
20% representation validation frames.  The representation has a 30-epoch ceiling, a 10-epoch
minimum, and patience-7 early stopping on that internal validation only.  `boundary_decision` is
never consulted until the representation and density are frozen; no train/decision difference is
computed or used.

Products follow:

```text
run/dlbm/v001_relation_changepoint/
  ckpt/<scene>.pth
  decision/<scene>.json
  logs/<scene>.log
  status/<scene>.done.json
```

Run all scenes dynamically on both GPUs with `experiment_mask_DLBM/v001/launch.sh`; validate the
completed products with `python experiment_mask_DLBM/v001/audit.py`.
