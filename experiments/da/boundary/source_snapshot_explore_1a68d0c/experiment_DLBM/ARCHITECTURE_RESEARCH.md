# DLBM bottom-level architecture study

Date: 2026-07-15  
Scope: frozen scene-wise Probe10 representations; no Final training and no Gate-label training.

## 1. Why v008 still behaves like a hand statistic

For a frame, the frozen discrepancy tensor is `D in R^(12 x 16 x 16)`.  v008
independently takes five patches from every layer, attention-pools the five
tokens, attention-pools the layers, and only then compares the base and opened
groups.

Instrumentation on the trained bike model found:

- normalized patch-attention entropy: about 0.998;
- normalized layer-attention entropy: about 0.999;
- fixing all top-5 coordinates did not change the chosen boundary;
- replacing the five values by their within-layer mean barely changed the gain.

The learned attention therefore collapsed to nearly uniform averaging.  The
effective computation is still close to `per-layer top-k mean -> layer mean ->
MLP`.  It can reproduce the scalar ordering-flip target, but it cannot retain
which spatial patch generated the response.

## 2. The same ViT patch is not always top-1, but spatial correspondence matters

On bike normal query frames, adjacent-layer top-5 Jaccard overlap ranges from
0.141 to 0.344 and adjacent top-1 equality from 0.115 to 0.375.  Nevertheless,
the full 16x16 discrepancy maps have adjacent-layer spatial correlations of
0.552 to 0.731.  On pseudo frames, top-5 Jaccard rises to 0.557--0.770 and
spatial correlation to 0.766--0.968.

Consequences:

1. Selecting top-k independently in each layer before cross-layer interaction
   destroys useful information.
2. Requiring the L10 maximum to be exactly the L11 maximum is too strong.
3. The network should compare aligned local neighborhoods on the complete grid,
   and perform top-k only after it has formed an opening-response map.

## 3. Tested predictive representation

The prototype reads layers from deep to shallow.  After observing layers
`l+1..11`, it predicts the complete next map before observing layer `l`:

```
Z_l       = SpatialConv3x3(D_l)
D_hat_l   = Decoder3x3(H_{l+1}, layer=l)
R_l       = D_l - D_hat_l
H_l       = ConvGRU3x3(Z_l, H_{l+1})
```

`R_l` is a learned innovation map: the part of the newly opened layer that is
not predictable from the already-open deeper suffix.  No spatial/layer pooling,
top-k score, pseudo label, Gate metric, or boundary label is used to pretrain
this representation.  The model has 19,073 parameters at 16 channels.

### Results after 10 normal-only epochs

| scene | normal query loss | independent-layer-roll / aligned | pseudo / normal | pseudo roll / aligned |
|---|---:|---:|---:|---:|
| bike | 1.60e-5 | 1.281 | 24.71 | 2.408 |
| crossroads | 4.26e-5 | 1.719 | 11.93 | 2.623 |
| highway | 1.42e-4 | 1.188 | 4.55 | 2.014 |

Independent per-layer circular shifts preserve every layer's values and local
texture but destroy cross-layer patch correspondence.  The consistent loss
increase shows that the model has learned aligned cross-layer structure.  The
pseudo branch was never used for training, yet its prediction loss is much
higher and is especially sensitive to correspondence destruction.

On bike, the ConvGRU normal query loss is 28.5% below a per-layer mean-map
predictor and 84.2% below copying the next deeper layer.

### Low-level ablations on bike

| encoder | parameters | normal loss | normal roll ratio | pseudo loss | pseudo roll ratio |
|---|---:|---:|---:|---:|---:|
| 3x3 ConvGRU, C=16 | 19,073 | 1.60e-5 | 1.281 | 3.95e-4 | 2.408 |
| 1x1 ConvGRU, C=16 | 2,561 | 2.16e-5 | 1.012 | 8.91e-4 | 1.022 |
| 1x1 ConvGRU, C=46 | 18,401 | 1.80e-5 | 1.149 | 7.01e-4 | 1.266 |
| 3x3 fixed residual cell, C=16 | 9,825 | 2.89e-5 | 1.022 | 8.75e-4 | 1.039 |

The parameter-matched 1x1 control confirms that local spatial convolution, not
only capacity, is necessary.  The fixed residual control confirms that the
learned recurrent gates/state transition are also necessary.

## 4. Recommended DLBM core

Do not replace this prototype with another unrelated boundary model.  Extend it
in two frozen stages:

### Stage 2A: cross-layer predictive pretraining

- Train the 3x3 ConvGRU predictor on normal Probe maps only.
- Keep all 256 patches through the recurrent stack.
- Use a shared spatial stem, shared ConvGRU, small absolute layer embeddings,
  and a lightweight decoder.
- Select the checkpoint by held-out normal prediction, plus the correspondence
  intervention check.  Do not select it using a boundary.

### Stage 2B: innovation-based boundary learning

Freeze or initially freeze Stage 2A.  For every incremental opening `k` build:

```
T_k = Conv3x3([H_{k+1}, Z_k, R_k, H_{k+1} * Z_k])
```

Then:

1. score/gather top-5 locations from `T_k`, after cross-layer interaction;
2. retain the five feature vectors rather than averaging them immediately;
3. use one shared pairwise relation head across all `k` to learn whether
   `S_{k+1}` and `S_k` change frame ordering;
4. aggregate frame tokens with a permutation-invariant set module only at the
   final scene level and keep videos balanced;
5. calibrate the learned transition against layer-subset structural nulls.

Normal and pseudo branches should pass through the same frozen predictive core
independently.  Fuse their innovation tokens late.  Do not form a raw
`pseudo-normal` layer GAP target.

The predictive pretraining prevents the order loss from taking v008's shortcut:
the encoder must retain aligned spatial information before the boundary head is
allowed to optimize the Gate-related signal.

## 5. Architecture choices not recommended now

- **Early top-k/GAP/attention pooling:** empirically collapses to means.
- **Pure MLP or 1x1 layer mixer:** cannot adequately exploit spatially coherent
  cross-layer responses.
- **A new full Transformer over 12x256 tokens:** unnecessarily data-hungry for
  one scene and does not itself prevent uniform-attention collapse.
- **Fixed 3D convolution only:** does not naturally expose every contiguous
  suffix state or share one incremental opening update from L11 toward L0.
- **Absolute pseudo localization:** already shown by v004 to prefer shallow
  layers for spatial resolution rather than incremental boundary value.

## 6. Method references

- ConvLSTM shows why convolutional state transitions preserve spatial structure
  while modeling an ordered axis: [Shi et al., NeurIPS 2015](https://proceedings.neurips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html).
- Predictive recurrent learning motivates retaining spatial appearance and
  transition information in recurrent memory: [PredRNN, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/e5f6ad6ce374177eef023bf5d0c018b6-Abstract.html).
- Reconstructing missing structure before downstream supervision follows the
  representation-learning principle used by [Masked Autoencoders, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html).
- A final frame-set aggregator can use interaction-aware permutation-invariant
  attention: [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html).
- If the boundary head directly optimizes frame ranks, differentiable ranks or
  Spearman can be implemented without hand-written hard sorting using
  [Fast Differentiable Sorting and Ranking, ICML 2020](https://proceedings.mlr.press/v119/blondel20a.html).

These references support the building blocks, not the conclusion that they
automatically optimize anomaly AUC.  The latter remains an inference requiring
the later boundary/Final experiments.

## 7. Boundary-head attachment study

The predictive core was then frozen and connected to a late-top-k opening head.
Each added layer retained five innovation features, and a layer GRU produced an
opening embedding.  Two variants were tested on bike:

1. one mixed embedding with separate order heads;
2. explicit spatial base state, candidate state, and transition embeddings.

The first variant selected L8 on both query and support, but its normal flip BCE
remained 0.684 (random is 0.693).  Its apparent boundary was therefore rejected.
The explicit-axis variant reached normal rank BCE 0.627 and flip BCE 0.685;
query selected L8 but support selected L0.  It was also rejected.

An alternating joint-finetuning control performed one prediction step followed
by one boundary step, avoiding a weighted multi-loss.  It did not solve the
problem: normal flip BCE stayed 0.691, prediction loss degraded from 1.60e-5 to
4.81e-5, and the correspondence intervention ratio collapsed from 1.272 to
0.994.  Boundary gradients erased the spatial representation.

The error is now localized to task granularity.  Manual Gate rank/flip is a
distributional statistic over hundreds of frames.  Treating it as an
independent sparse XOR label for individual frame pairs is both poorly
conditioned and destructive.  The next head should operate on balanced
frame/video sets or episodes after the frozen innovation encoder.  It should
model the joint base/opening distribution directly; the ConvGRU core should
remain frozen until that scene-level objective passes correspondence and
support/query checks.

## 8. Episode-level and input-capacity audit

Episode pooling fixed the pairwise-XOR optimization problem but did not fix the
target.  Real/null classification saturated, base/opening retrieval measured
identifiability rather than usefulness, and two-pseudo-view retrieval measured
perturbation stability rather than opening gain.  All three passed their direct
intervention controls while failing support/query boundary checks.

A separate channel-preserving predictor tested whether cosine discrepancy had
discarded the needed signal.  Direct vector prediction collapsed to zero; a
cosine autoencoder removed collapse but produced only a 1.048 spatial-roll ratio
and 1.004 pseudo/normal ratio.  The scalar discrepancy ConvGRU therefore remains
the stronger observation encoder.

The complete evidence and rejection criteria are recorded in
`run/boundary/_architecture_study/EPISODE_BOUNDARY_REVIEW.md`.  The next target
must measure the held-out effect of an actual opening intervention; another
passive distribution encoder is not justified by these results.

## 9. Order-transport correction and v012--v016 findings

The preceding last sentence was too strong.  Manual Gate never observes AUC or
an actual Final-training intervention.  Its useful prior is that an opening's
change to the frame/patch order distribution defines a distillation boundary.
The learned module should therefore capture that same train-only signal rather
than invent an unsupervised AUC surrogate.

The following controlled sequence was run on bike, crossroads, and highway
(external coarse references L8, L4, and L0 respectively):

| version | learned quantity | observed qualification result | conclusion |
|---|---|---|---|
| v012 | episode flip-rate target | query L4 / L8 / L8 | rejected: pseudo order change was incorrectly rewarded in the same direction as normal change |
| v013 | reduction in candidate-order NLL after opening | full-scene audit L8 / L4 / L4 | useful representation, but opening-explanation gain hides highway's shallow rank novelty |
| v014 | base-to-candidate order novelty relative to structural nulls | full L8 / L0 / L0 | captures bike and highway; crossroads shallow evidence is too close to zero |
| v015 | same novelty, trained on every scene video | full L8 / L8 / L0 | same-set prediction suppresses crossroads middle novelty; rejected as self-prediction leakage |
| v016 | two-fold out-of-video novelty | cross-fit L0 / L0 / L8 | rejected: normal novelty alone is not the complete Gate signal |

The v013 decomposition is still the strongest bottom-level finding.  Its shared
order head learns `q(candidate pair order | base patch episode)` and the opening
path can explain the candidate ordering without an explicit flip, Spearman,
z-score, tail, or AUC target.  On one fixed v013 model, the learned normal
novelty `(M,S)` was:

- bike `(-0.0225,-0.0089)`;
- crossroads `(+0.0281,-0.0027)`;
- highway `(-0.0375,+0.0428)`.

The cumulative utilities `0`, `G_M`, and `G_M+G_S` then reproduce L8, L4, and
L0, including highway's shallow bridge.  Fresh training and cross-fitting show
that weak evidence around zero is not stable enough by itself, however.  More
importantly, historical Gate metrics show that bike has real normal rank
novelty and is kept at D by its pseudo-nuisance response.  A normal-only model
can therefore never robustly reproduce all three decisions.

The next justified architecture is one scene-level two-axis episode model:

1. a normal branch learns opening order novelty from full patch relations;
2. a pseudo branch independently learns whether the same opening is dominated
   by perturbation response, without feeding a handcrafted pseudo-normal delta;
3. a learned scene/video set head combines the two branch tokens into cumulative
   D/MD/ALL utilities;
4. the loss supervises the meaning of the two axes, while the combination is not
   implemented as Gate thresholds, z-scores, tail rules, or nested if/else.

No seven-scene expansion or Final training is justified until this two-axis
head passes bike/crossroads/highway and weak-evidence stability checks.

## 10. Two-axis qualification (v017--v019)

Three concrete two-axis definitions were implemented and tested on bike,
crossroads, and highway.  In every case the normal axis remained the learned
base-to-candidate order novelty from section 9; only the pseudo axis changed.
Task scales were learned with homoscedastic NLL uncertainty rather than fixed
loss weights.

| version | pseudo axis | training | full-scene boundary (bike/crossroads/highway) | result |
|---|---|---:|---|---|
| v017 | synthetic perturbation-mask localization from pseudo opening maps | 30 epochs | L8 / L8 / L8 | rejected: learns shallow spatial resolution and suppresses highway's real shallow bridge |
| v018 | independent pseudo-only base-to-candidate order novelty | 30 epochs | L4 / L4 / L4 | rejected: pseudo-order novelty sign is not equivalent to nuisance |
| v019 | learned normal-vs-pseudo opening-token two-sample critic | 5-epoch qualification | L8 / L8 / L4 | rejected before long training: query/full disagreement remains and highway shallow novelty drifts |

All three networks trained normally: gradients were finite, the learned task
precisions stayed positive, and v017's mask head clearly separated perturbed
from background patches.  The v017 failure is therefore semantic rather than a
collapse.  It independently reproduces the earlier v004 finding that absolute
pseudo localization selects spatial resolution, not incremental boundary
value.

The combined evidence rules out a late scalar subtraction of two independently
trained axes.  Both axes are observable, but their scene-level meaning depends
on their joint distribution across videos.  The next architecture should form
normal and pseudo opening tokens with shared layer semantics, retain per-video
sets, and learn their interaction before producing an ordinal boundary.  It
should not first reduce each branch to one novelty/nuisance scalar.  No v017--
v019 result justifies seven-scene or Final training.

## 11. Joint episode and raw-transport correction (v020--v022)

v020 tested the late-interaction proposal from section 10.  A shared opening
encoder formed one joint normal/pseudo token for every frame pair; dense heads
predicted normal flips, pseudo flips, and `normal flip AND NOT pseudo flip`, and
a Set-Attention head predicted the video-level event rate.  The result was
full-scene L0 / L8 / L0 for bike/crossroads/highway and was rejected.

The important result was obtained by auditing its targets rather than adding a
larger head.  In all three scenes, the raw joint event increased for M and
decreased for S.  Even a perfect v020 classifier would therefore choose roughly
the same boundary.  At epoch 10 its pair BCEs also remained near 0.67--0.70,
while the episode head merely learned the global positive rate near 0.1.  A
Boolean flip discards the continuous direction and magnitude that distinguish
the manual Gate cases.

The DABG Probe10 policy history confirms the missing structure without implying
that its hard rule should be copied:

- bike has strong M normal rank transport but a strong positive pseudo opening
  response and remains at D;
- crossroads has strong M transport and negative pseudo response and opens MD;
- highway has weak M but strong S transport and negative pseudo response and
  opens ALL.

v021 consequently returned the network input to raw per-layer discrepancy maps.
Top-5 only chooses the hard patch support already used by Probe training.  A
shared spatial convolution, layer attention, pair relation network, and branch
critic learn every subsequent representation.  The two losses are continuous
normal order prediction and normal-vs-pseudo transport discrimination; no
rank-z, tail rule, pseudo threshold, or AUC is used.

| version | full boundary | query boundary | support boundary | conclusion |
|---|---|---|---|---|
| v021 bike | L8 | L8 | L8 | correct and stable |
| v021 crossroads | L4 | L0 | L8 | correct full aggregate, unstable by video split |
| v021 highway | L8 | L0 | L8 | shallow signal appears in query but does not generalize |
| v022 bike | L4 | L4 | L4 | video consistency removes variation but changes the correct decision |
| v022 crossroads | L0 | L0 | L8 | still unstable and full decision regresses |
| v022 highway | L8 | L8 | L8 | shallow bridge is suppressed |

v021 is the first variant in this sequence whose raw learned axes recover the
correct full boundary for both bike and crossroads while retaining highway's
shallow evidence on a held-out video subset.  This is evidence that the input
and task semantics are now substantially closer to the manual Gate signal.
However, it is not a qualified DLBM because the evidence sign is video-domain
dependent.

v022 penalized the variance of the final scalar utility between two videos,
with a learned task uncertainty instead of a fixed weight.  Its failure shows
that forcing the already-combined utility to agree is too late and erases useful
pseudo-veto/shallow-bridge magnitude.  Future stability work should align the
normal and pseudo transport representations separately (or estimate a
hierarchical video-set posterior), not force their final difference to be
equal.  Neither v020, v021, nor v022 justifies seven-scene or Final training.
