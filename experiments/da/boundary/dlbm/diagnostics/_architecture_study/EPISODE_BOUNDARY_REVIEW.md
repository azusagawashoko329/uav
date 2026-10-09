# Episode-level DLBM internal review

Status: **scalar predictive core retained; all current episode boundary heads rejected**.

## Accepted component

The normal-only 3x3 ConvGRU on full Teacher--Student cosine-discrepancy grids is
still the only representation that passes all low-level controls.  On the saved
scene predictors, independently rolling every layer's patch grid raises held-out
loss by about 1.19x--1.74x.  The predictor remains frozen during every boundary
experiment and Probe10 hashes are unchanged.

## Rejected attachment studies

### v009 real/null episode classifier

The head used predictive innovations, multi-seed permutation-invariant pooling,
and real same-frame openings versus marginal-preserving frame-permuted openings.
The initial ranking loss left D/M/S logit norms unidentified (bike margins grew to
roughly 23/39/100).  Balanced absolute calibration bounded the margins but made
all three tasks saturate; bike oscillated and finished query/support/shifted at L4.
The task detects correspondence, not useful distillation gain.

### v010 same-frame base/opening retrieval

InfoNCE frame retrieval removed the hand-defined flip target.  It learned
non-trivial held-out retrieval, but bike finished query L4/support L0,
crossroads L0/L0, and highway L0/L0.  A coherent or identifiable opening can
still be nuisance-like, so mutual information is not a gain target.

### Channel-preserving vector input

Raw 768-D Teacher/Student patch directions were tested to determine whether the
scalar cosine map had discarded the missing signal.  Direct SmoothL1 prediction
collapsed to the zero residual (only 0.1% better than zero).  A cosine
autoencoder followed by a frozen bottleneck removed that shortcut, but the final
cross-layer roll ratio was only 1.048 and pseudo/normal loss ratio 1.004.  ViT
channel residual direction is not stable enough across layers; this input is
rejected rather than attached to another head.

### v011 pseudo-branch cross-view retrieval

Two independent multi-mode pseudo views were matched only within the pseudo
branch.  Pair derangement reduced MRR to near chance, proving that the network
learned the intended stable same-frame signal.  It nevertheless finished:

| scene | query | support | external coarse audit |
|---|---:|---:|---:|
| bike | L0 | L4 | L8 |
| crossroads | L4 | L0 | L4 |
| highway | L4 | L4 | L0 |

The correspondence target is valid but is not a boundary-gain target.

## Target-level finding

The current UAV Probe10 only stores the stable ep10 OCBE/Student.  The historical
effective Gate used warmup response trajectories and a richer pseudo stress
family.  More importantly, direct audits on the current UAV ep10 features show
that its M/S scalar opening gains do not reproduce the historical coarse choices;
for example the audited highway shallow ordering gain is negative while several
deep-only scenes have positive shallow gain.  A network cannot recover the old
policy by fitting a target that is already inconsistent at its input.

## Decision

Do not run seven-scene Boundary or Final from v009--v011.  Do not add another
pooler, Transformer, pseudo-normal delta, or handcrafted metric combination.
The next experiment must make the candidate opening an actual train-time
intervention and learn its held-out episode effect.  The frozen scalar predictor
can remain the observation encoder; the missing variable is intervention gain,
not feature capacity.
