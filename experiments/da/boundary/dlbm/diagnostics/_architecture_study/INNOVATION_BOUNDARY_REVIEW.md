# Predictive-innovation boundary attachment review

Status: **predictive representation accepted; current pairwise boundary head rejected**.

- Frozen predictive core reproduces the earlier correspondence result on bike:
  normal roll ratio 1.272 and pseudo roll ratio about 2.34.
- Mixed-axis head: query L8, support L8, shifted L0, but normal flip BCE 0.684.
- Separate base/candidate axes: query L8, support L0, shifted L0; normal rank BCE
  0.627 and flip BCE 0.685.
- Alternating end-to-end control: query/support/shifted L0; normal flip BCE
  0.691; predictive correspondence destroyed (roll ratio 1.272 -> 0.994).

No seven-scene run and no Final run are authorized from these results.  Do not
interpret the isolated bike L8 result as a successful boundary.  The next
change is an episode/scene-level distribution head on the frozen innovations,
not another per-frame-pair classifier and not more epochs.
