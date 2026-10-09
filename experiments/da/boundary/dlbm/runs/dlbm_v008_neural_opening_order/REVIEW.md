# DLBM v008 internal review

Status: **core signal validated; boundary policy rejected as a final candidate**.

## What v008 tested

- Input: frozen Probe10 Teacher--Student per-patch cosine discrepancy maps.
- Encoder input: each layer's raw top-5 patch values, coordinates, and ranks.
- Learned targets: base/candidate frame ordering and pairwise ordering flips.
- Calibration: real D->MD and MD->ALL openings versus same-ratio layer-subset
  openings drawn from the current deeper group.
- Readout: cumulative normal-branch calibrated gains for L8, L4, and L0.
- Manual Gate decisions were never used for training or selection.

## Passed checks

- All seven scene jobs completed; all expected result/checkpoint files exist.
- No NaN, Inf, traceback, or failed scene.
- Probe checkpoint SHA256 values are identical before and after the run.
- Full test suite: 44/44 passed.
- Peak allocation: 0.84 GiB per scene.
- Learned gain and direct top-k audit gain agree in sign for 13/14 M/S
  transitions. This validates that the network learned the same opening-order
  primitive instead of only recognizing a layer id or a matched frame.
- Scene outputs no longer collapse to one boundary: L0 for bike, crossroads,
  farmland, vehicle; L4 for highway, railway, solar.

## Failed candidate checks

- External coarse Gate agreement: 0/7.
- Support/query boundary agreement: 5/7. Bike and crossroads choose L4 on
  support but L0 on query.
- Bike oscillates L4->L0->L4->L0 at epochs 5/10/20/30, despite decreasing
  training loss. More DLBM epochs therefore do not solve boundary stability.
- A single null-calibrated ordering-flip gain is not equivalent to the manual
  Gate's final notion of useful opening. The historical policy also used the
  sign/shape of the within-branch opening response, especially pseudo nuisance
  suppression and tail-conditioned effects.

## Decision

Do not run Final with v008 and do not label it a successful boundary module.
Keep it as evidence that the raw patch network can learn the correct underlying
order-change signal. The next version must learn how opening effects are useful
or nuisance-like; adding a hard-coded pseudo veto would merely recreate the old
policy and is explicitly not accepted.
