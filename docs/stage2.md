# Stage 2 Results: Flow Matching as a Layer

Produced by `notebooks/stage2_flow_matching.ipynb`.

A small velocity field `v(z, t)` is trained to transport a frozen CLIP RN50 image embedding
at t=0 onto its frozen CLIP **text** prototype at t=1. At test time the field is applied to
an **unlabelled** embedding for T Euler steps, and the transported embedding is classified
with the same cosine 1-NN rule as the Stage 1 zero-shot baseline.

| Item | Value |
| --- | --- |
| Datasets | DTD (47 classes, partition 1), FGVC-Aircraft (100 classes, variant level) |
| Encoder | CLIP RN50 (1024-d), frozen; features cached from Stage 1 |
| Target | Frozen CLIP text prototype `p_c` of the true class |
| Baseline | Stage 1 zero-shot CLIP: DTD 0.4005, Aircraft 0.1545 |
| Objectives | Standard FM (`L_FM`) and rolled-out FM (`L_roll`) |
| Euler steps | T in {4, 12} |
| Protocol | K in {5, 10, full} x seeds {0, 1, 2}, reusing the Stage 1 subsets |
| Field | MLP 1025 -> 512 -> 512 -> 1024, SiLU, scalar t concatenated (1.31 M params) |
| Optimiser | AdamW, lr 1e-3 constant, wd 1e-4, batch 256, 300 epochs, best val accuracy |
| Metric | Top-1 accuracy on the complete official test split, at the endpoint `z_T` |
| Runs | 54 trained fields -> 72 result rows (36 standard + 36 rolled-out) |

---

## 1. Method

**Standard flow matching.** Sample `t ~ U[0,1]` per example, interpolate on the straight
conditional-OT path and regress its constant velocity:

```
z_t = (1 - t) z_i + t p_yi        u_i = p_yi - z_i        L_FM = || v(z_t, t) - u_i ||^2
```

**Rolled-out flow matching.** Run the inference loop itself and supervise only where it
lands, backpropagating through all T network calls:

```
z_0 = z_i,  z_{k+1} = z_k + (1/T) v(z_k, k/T)        L_roll = || z_T - p_yi ||^2
```

**Inference (both).** T Euler steps of size 1/T from the test embedding, then cosine 1-NN
against the text prototypes. `z_T` is at t=1 for both T values; T only controls how coarsely
the interval is discretised, so the two step counts are two approximations of the same map.

Standard FM does not depend on T at training time, so one field per (dataset, K, seed) is
trained and scored at both step counts; its checkpoint is selected on the mean validation
accuracy over T=4 and T=12. Rolled-out FM bakes T into the loss, so it gets one field per T,
with the same T at train and test. Architecture, optimiser, epochs and batch size are
identical across the two objectives.

### Small modifications, and why

| Change                                     | Reason |
|--------------------------------------------| --- |
| Output layer zero-initialised              | Makes the untrained field the identity, so t=0 reproduces the Stage 1 baseline exactly. Every diagnostic run asserts `\|acc(t=0) - baseline\| < 5e-3`. |
| Inputs L2-normalised once, before training | Stage 1 classifies with cosine, so length carries no class information, and CLIP's image and text towers have different norm scales. Without it `u_i = p_c - z_i` would be dominated by a length mismatch the classifier then discards. |

---

## 2. Accuracy Table

Mean +/- std over 3 seeds. For K = 5 and K = 10 the seeds select the balanced subset; for
K = full the subset is fixed and the seeds vary initialisation and batch order. `dAcc` is
measured against the Stage 1 zero-shot baseline of the same dataset.

### DTD (baseline 0.4005)

| Variant | T | K | Top-1 accuracy | dAcc |
| --- | --- | --- | --- | --- |
| Standard FM | 4 | 5 | 0.5227 +/- 0.0100 | +0.1222 |
| Standard FM | 12 | 5 | 0.5252 +/- 0.0090 | +0.1246 |
| Rolled-out FM | 4 | 5 | 0.5236 +/- 0.0094 | +0.1230 |
| Rolled-out FM | 12 | 5 | 0.5163 +/- 0.0080 | +0.1158 |
| Standard FM | 4 | 10 | 0.5902 +/- 0.0025 | +0.1897 |
| Standard FM | 12 | 10 | 0.5931 +/- 0.0019 | +0.1926 |
| Rolled-out FM | 4 | 10 | 0.5915 +/- 0.0074 | +0.1910 |
| Rolled-out FM | 12 | 10 | 0.5771 +/- 0.0024 | +0.1766 |
| Standard FM | 4 | full | 0.6466 +/- 0.0016 | +0.2461 |
| Standard FM | 12 | full | 0.6493 +/- 0.0029 | +0.2488 |
| Rolled-out FM | 4 | full | **0.6651 +/- 0.0031** | **+0.2645** |
| Rolled-out FM | 12 | full | 0.6642 +/- 0.0043 | +0.2637 |

### FGVC-Aircraft (baseline 0.1545)

| Variant | T | K | Top-1 accuracy | dAcc |
| --- | --- | --- | --- | --- |
| Standard FM | 4 | 5 | 0.1708 +/- 0.0046 | +0.0163 |
| Standard FM | 12 | 5 | 0.1712 +/- 0.0026 | +0.0167 |
| Rolled-out FM | 4 | 5 | 0.1651 +/- 0.0024 | +0.0106 |
| Rolled-out FM | 12 | 5 | 0.1659 +/- 0.0031 | +0.0114 |
| Standard FM | 4 | 10 | 0.2053 +/- 0.0072 | +0.0508 |
| Standard FM | 12 | 10 | 0.2087 +/- 0.0049 | +0.0542 |
| Rolled-out FM | 4 | 10 | 0.2074 +/- 0.0075 | +0.0529 |
| Rolled-out FM | 12 | 10 | 0.2097 +/- 0.0095 | +0.0552 |
| Standard FM | 4 | full | 0.2319 +/- 0.0044 | +0.0774 |
| Standard FM | 12 | full | 0.2347 +/- 0.0060 | +0.0802 |
| Rolled-out FM | 4 | full | **0.2845 +/- 0.0056** | **+0.1300** |
| Rolled-out FM | 12 | full | 0.2542 +/- 0.0071 | +0.0997 |

The layer helps everywhere. The best setting adds **+26.5 points on DTD**
(0.4005 -> 0.6651, +66% relative) and **+13.0 points on Aircraft** (0.1545 -> 0.2845, +84%
relative), and no cell in either table is below its baseline.

<details>
<summary>Per-run detail (72 rows), with the final training loss of each field</summary>

`Loss` is the mean training loss at epoch 300 of the field that produced the row: a velocity
MSE for standard FM, an endpoint MSE for rolled-out FM. Standard rows share one field per
(dataset, K, seed), hence one loss value across both step counts.

| Variant | Dataset | K | Seed | T | Loss | Test acc | dAcc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | Aircraft | 5 | 0 | 4 | 0.000130 | 0.1677 | +0.0132 |
| Standard | Aircraft | 5 | 0 | 12 | 0.000130 | 0.1713 | +0.0168 |
| Rolled-out | Aircraft | 5 | 0 | 4 | 0.000105 | 0.1674 | +0.0129 |
| Rolled-out | Aircraft | 5 | 0 | 12 | 0.000108 | 0.1695 | +0.0150 |
| Standard | Aircraft | 5 | 1 | 4 | 0.000135 | 0.1761 | +0.0216 |
| Standard | Aircraft | 5 | 1 | 12 | 0.000135 | 0.1737 | +0.0192 |
| Rolled-out | Aircraft | 5 | 1 | 4 | 0.000108 | 0.1626 | +0.0081 |
| Rolled-out | Aircraft | 5 | 1 | 12 | 0.000108 | 0.1644 | +0.0099 |
| Standard | Aircraft | 5 | 2 | 4 | 0.000137 | 0.1686 | +0.0141 |
| Standard | Aircraft | 5 | 2 | 12 | 0.000137 | 0.1686 | +0.0141 |
| Rolled-out | Aircraft | 5 | 2 | 4 | 0.000102 | 0.1653 | +0.0108 |
| Rolled-out | Aircraft | 5 | 2 | 12 | 0.000107 | 0.1638 | +0.0093 |
| Standard | Aircraft | 10 | 0 | 4 | 0.000119 | 0.2124 | +0.0579 |
| Standard | Aircraft | 10 | 0 | 12 | 0.000119 | 0.2112 | +0.0567 |
| Rolled-out | Aircraft | 10 | 0 | 4 | 0.000096 | 0.2145 | +0.0600 |
| Rolled-out | Aircraft | 10 | 0 | 12 | 0.000099 | 0.2178 | +0.0633 |
| Standard | Aircraft | 10 | 1 | 4 | 0.000115 | 0.1980 | +0.0435 |
| Standard | Aircraft | 10 | 1 | 12 | 0.000115 | 0.2031 | +0.0486 |
| Rolled-out | Aircraft | 10 | 1 | 4 | 0.000097 | 0.1995 | +0.0450 |
| Rolled-out | Aircraft | 10 | 1 | 12 | 0.000100 | 0.1992 | +0.0447 |
| Standard | Aircraft | 10 | 2 | 4 | 0.000114 | 0.2055 | +0.0510 |
| Standard | Aircraft | 10 | 2 | 12 | 0.000114 | 0.2118 | +0.0573 |
| Rolled-out | Aircraft | 10 | 2 | 4 | 0.000093 | 0.2082 | +0.0537 |
| Rolled-out | Aircraft | 10 | 2 | 12 | 0.000095 | 0.2121 | +0.0576 |
| Standard | Aircraft | full | 0 | 4 | 0.000108 | 0.2349 | +0.0804 |
| Standard | Aircraft | full | 0 | 12 | 0.000108 | 0.2403 | +0.0858 |
| Rolled-out | Aircraft | full | 0 | 4 | 0.000091 | 0.2829 | +0.1284 |
| Rolled-out | Aircraft | full | 0 | 12 | 0.000106 | 0.2562 | +0.1017 |
| Standard | Aircraft | full | 1 | 4 | 0.000112 | 0.2340 | +0.0795 |
| Standard | Aircraft | full | 1 | 12 | 0.000112 | 0.2355 | +0.0810 |
| Rolled-out | Aircraft | full | 1 | 4 | 0.000088 | 0.2799 | +0.1254 |
| Rolled-out | Aircraft | full | 1 | 12 | 0.000102 | 0.2601 | +0.1056 |
| Standard | Aircraft | full | 2 | 4 | 0.000105 | 0.2268 | +0.0723 |
| Standard | Aircraft | full | 2 | 12 | 0.000105 | 0.2283 | +0.0738 |
| Rolled-out | Aircraft | full | 2 | 4 | 0.000090 | 0.2907 | +0.1362 |
| Rolled-out | Aircraft | full | 2 | 12 | 0.000106 | 0.2463 | +0.0918 |
| Standard | DTD | 5 | 0 | 4 | 0.000149 | 0.5340 | +0.1335 |
| Standard | DTD | 5 | 0 | 12 | 0.000149 | 0.5351 | +0.1346 |
| Rolled-out | DTD | 5 | 0 | 4 | 0.000079 | 0.5282 | +0.1277 |
| Rolled-out | DTD | 5 | 0 | 12 | 0.000087 | 0.5245 | +0.1239 |
| Standard | DTD | 5 | 1 | 4 | 0.000152 | 0.5154 | +0.1149 |
| Standard | DTD | 5 | 1 | 12 | 0.000152 | 0.5176 | +0.1170 |
| Rolled-out | DTD | 5 | 1 | 4 | 0.000077 | 0.5298 | +0.1293 |
| Rolled-out | DTD | 5 | 1 | 12 | 0.000086 | 0.5160 | +0.1154 |
| Standard | DTD | 5 | 2 | 4 | 0.000148 | 0.5186 | +0.1181 |
| Standard | DTD | 5 | 2 | 12 | 0.000148 | 0.5229 | +0.1223 |
| Rolled-out | DTD | 5 | 2 | 4 | 0.000077 | 0.5128 | +0.1122 |
| Rolled-out | DTD | 5 | 2 | 12 | 0.000089 | 0.5085 | +0.1080 |
| Standard | DTD | 10 | 0 | 4 | 0.000134 | 0.5888 | +0.1883 |
| Standard | DTD | 10 | 0 | 12 | 0.000134 | 0.5926 | +0.1920 |
| Rolled-out | DTD | 10 | 0 | 4 | 0.000065 | 0.5915 | +0.1910 |
| Rolled-out | DTD | 10 | 0 | 12 | 0.000068 | 0.5750 | +0.1745 |
| Standard | DTD | 10 | 1 | 4 | 0.000138 | 0.5931 | +0.1926 |
| Standard | DTD | 10 | 1 | 12 | 0.000138 | 0.5952 | +0.1947 |
| Rolled-out | DTD | 10 | 1 | 4 | 0.000066 | 0.5840 | +0.1835 |
| Rolled-out | DTD | 10 | 1 | 12 | 0.000070 | 0.5766 | +0.1761 |
| Standard | DTD | 10 | 2 | 4 | 0.000130 | 0.5888 | +0.1883 |
| Standard | DTD | 10 | 2 | 12 | 0.000130 | 0.5915 | +0.1910 |
| Rolled-out | DTD | 10 | 2 | 4 | 0.000064 | 0.5989 | +0.1984 |
| Rolled-out | DTD | 10 | 2 | 12 | 0.000070 | 0.5798 | +0.1793 |
| Standard | DTD | full | 0 | 4 | 0.000092 | 0.6463 | +0.2457 |
| Standard | DTD | full | 0 | 12 | 0.000092 | 0.6463 | +0.2457 |
| Rolled-out | DTD | full | 0 | 4 | 0.000040 | 0.6628 | +0.2622 |
| Rolled-out | DTD | full | 0 | 12 | 0.000050 | 0.6596 | +0.2590 |
| Standard | DTD | full | 1 | 4 | 0.000090 | 0.6452 | +0.2447 |
| Standard | DTD | full | 1 | 12 | 0.000090 | 0.6495 | +0.2489 |
| Rolled-out | DTD | full | 1 | 4 | 0.000039 | 0.6638 | +0.2633 |
| Rolled-out | DTD | full | 1 | 12 | 0.000046 | 0.6681 | +0.2676 |
| Standard | DTD | full | 2 | 4 | 0.000090 | 0.6484 | +0.2479 |
| Standard | DTD | full | 2 | 12 | 0.000090 | 0.6521 | +0.2516 |
| Rolled-out | DTD | full | 2 | 4 | 0.000039 | 0.6686 | +0.2681 |
| Rolled-out | DTD | full | 2 | 12 | 0.000047 | 0.6649 | +0.2644 |

</details>

---

## 3. Accuracy vs. Training-Set Size

Four FM curves (standard / rolled-out x T = 4 / 12) against the flat zero-shot baseline.
Error bars are the std over 3 seeds.

![Flow accuracy vs K](figures/flow_accuracy_vs_k.png)

- **The layer never hurts.** Even at K = 5 - 235 images on DTD, 500 on Aircraft - every
  variant is above the baseline, and the gain is already large on DTD (+0.12, which is 46% of
  the full-split gain).
- **T barely matters for standard FM.** T=12 is ahead in all six standard cells, so the
  finer discretisation does help - but by at most 0.0034, which is comparable to or below the
  seed spread (0.0013 - 0.0100) in every one of them. The learned velocity is therefore close
  to constant along the path: 4 Euler steps already resolve the map, and discretisation error
  is not what limits accuracy.
- **T matters for rolled-out FM, and fewer steps are better.** On Aircraft K = full,
  T=4 scores 0.2845 against 0.2542 for T=12, a 0.030 gap that is ~5x the seed std. On DTD
  K = 10 the same direction gives +0.014. T=4 is never meaningfully worse than T=12.
- **Rolled-out training needs data.** It wins clearly only at K = full (+0.0158 on DTD,
  +0.0498 on Aircraft over the best standard setting). At K = 5 it is level with or slightly
  behind standard FM on both datasets - the endpoint objective has more freedom in how it
  reaches the prototype, and with 5 shots per class that freedom is spent on the training set.
- **Spread shrinks with K**, as in Stage 1: DTD std falls from ~0.009 at K = 5 to ~0.003 at
  K = full.
- Aircraft remains much harder than DTD, and the ordering of the two datasets is unchanged
  from Stage 1: 100 fine-grained variants whose text prompts are close together give the flow
  a much weaker target geometry than 47 texture names.

---

## 4. Training-Loss Curves

Representative K = full, seed 0 runs. The two objectives are on separate panels on purpose:
standard FM minimises a velocity MSE against `p_y - z_i`, rolled-out FM an endpoint MSE
against `p_y`. They are different quantities and overlaying them would be meaningless.

Both are reported as per-element means over 1024 dimensions, so multiplying by 1024 turns
them into squared distances: the DTD rolled-out field at T = 4, seed 0 ends at 4.0e-5, i.e.
`||z_T - p_y||^2` = 0.041, an RMS distance of 0.20 from the prototype in a space where every
vector has unit norm.

**DTD**

![DTD training curves](figures/flow_training_curves_dtd.png)

**FGVC-Aircraft**

![Aircraft training curves](figures/flow_training_curves_aircraft.png)

- Training is stable for both objectives at lr 1e-3: the loss falls smoothly over all 300
  epochs with no oscillation or divergence, and backpropagating through 12 sequential Euler
  steps did not need gradient clipping or a smaller learning rate.
- **The deeper rollout optimises worse.** In 17 of the 18 rolled-out pairs the T=12 field
  ends at a *higher* endpoint loss than its T=4 twin, and the eighteenth is a tie at the
  printed precision - it is never lower (DTD K = full: 4.6e-5 - 5.0e-5 vs 3.9e-5 - 4.0e-5;
  Aircraft K = full: 1.02e-4 - 1.06e-4 vs 8.8e-5 - 9.1e-5). Twelve chained network calls are
  simply a harder optimisation problem than four, and on Aircraft that shows up directly in
  test accuracy.
- Loss falls with K for both objectives, as expected: more examples per class make the
  displacement field easier to fit, not harder.

---

## 5. Feature Space Before and After the Layer

Original features, after standard FM and after rolled-out FM, for 8 fixed classes with
T = 12, K = full, seed 0. The PCA is fitted **once** over all three feature sets and the
prototypes together and then applied to each, so the three panels share one coordinate
system and the movement between them is real rather than a re-projection. Stars are the text
prototypes.

**DTD**

![DTD feature comparison](figures/flow_feature_comparison_dtd.png)

**FGVC-Aircraft**

![Aircraft feature comparison](figures/flow_feature_comparison_aircraft.png)

- The left panel is the Stage 1 picture: the image cloud and the text prototypes sit in two
  separate regions, the CLIP modality gap described in `stage1.md`. The layer's job is
  precisely to cross that gap, and the training loss says it does - the rolled-out fields
  drawn here (T = 12, K = full) land their training features at an RMS distance of ~0.22
  (DTD) and ~0.33 (Aircraft) from the target prototype, in a space where every vector has
  unit norm.
- What to read off the two transported panels is whether the per-class groups stay
  *separated* while they concentrate. Concentration alone is not enough: the classifier is a
  cosine 1-NN against the prototypes, so a field that drove every embedding onto the
  prototype barycentre would score worse than the baseline. The accuracy table shows this did
  not happen.
- Aircraft is the harder case to inspect: 100 prototypes packed into a narrow cone, so
  neighbouring variants can land in the same region even when the endpoint distance is small.
  That is consistent with Aircraft gaining 13 points where DTD gains 26.

---

## 6. Flow Trajectories for Individual Examples

Individual test features traced through the T = 12 Euler steps, K = full, seed 0. Circle =
original feature, every intermediate state marked, square = transported endpoint, star = the
class prototype; 4 classes x 2 examples.

**DTD, rolled-out FM**

![DTD trajectories, rolled](figures/flow_trajectories_dtd_rolled.png)

**DTD, standard FM**

![DTD trajectories, standard](figures/flow_trajectories_dtd_standard.png)

**FGVC-Aircraft, rolled-out FM**

![Aircraft trajectories, rolled](figures/flow_trajectories_aircraft_rolled.png)

- The standard-FM paths should be close to straight and close to evenly spaced, because the
  field is regressed onto the constant velocity `p_y - z_i` of a straight path. The
  T=4 vs T=12 agreement in the accuracy table (<= 0.0034 everywhere) is the quantitative
  version of the same statement.
- Rolled-out FM is under no such constraint: nothing in `L_roll` says the intermediate states
  have to lie on the segment, only that `z_T` lands on the prototype. Its paths are free to
  bend, and its intermediate states have no interpretation as "partially transported"
  features.
- Both variants are label-free at inference: the field is applied to the embedding without
  knowing which star it should be heading for.

---

## 7. Diagnostic: Accuracy Along t

Not a deliverable - the brief scores the endpoint `z_T` - but it is the figure that explains
whether the flow overshoots. Finely integrated (51 report times, 50 sub-steps per unit time),
standard FM on DTD, K = full, seed 0.

![DTD accuracy along t](figures/flow_accuracy_vs_t_dtd_standard.png)

- The curve starts **exactly** on the zero-shot baseline. This is enforced, not observed: the
  output layer is zero-initialised, and the run asserts
  `|acc(t=0) - 0.4005| < 5e-3` before the figure is drawn.
- The same field scored at the brief's endpoint gives 0.6463 for both T = 4 and T = 12.
- The peak of the curve is annotated on the figure. Any gap between that peak and the
  endpoint is the amount left on the table by integrating all the way to t=1; the brief does
  not allow stopping early, so it is reported here as diagnosis rather than as a result.

---

## 8. Optional: Reverse Flow, Rendered by Retrieval

Flow matching is time-symmetric, so the same field run from t=1 down to t=0 carries each
class *text* prototype back into image-embedding space. Each intermediate point is rendered
as the nearest real training image in cosine similarity - no decoder, and nothing leaves the
CLIP RN50 space the field was trained in. Rolled-out field, T = 12, K = full, seed 0.

![DTD reverse flow](figures/flow_reverse_dtd.png)

Read it as "what does this point in embedding space look like", not as generation: the
forward map is many-to-one, so the reverse path from exactly `p_c` is a single deterministic
trajectory towards an average class image. Titles are green when the retrieved image belongs
to the row's class and red otherwise, with the cosine similarity of the match.

---

## 9. Discussion

**Against Stage 1.** The FM layer turns zero-shot CLIP RN50 into a competitive few-shot
classifier without touching the encoder or the prototypes:

| Setting | DTD | Aircraft |
| --- | --- | --- |
| Zero-shot CLIP (Stage 1) | 0.4005 | 0.1545 |
| Best FM layer, K = 5 | 0.5252 | 0.1712 |
| Best FM layer, K = full | **0.6651** | **0.2845** |
| Linear probe, ResNet-18, K = full (Stage 1) | 0.6284 | 0.3662 |
| Linear probe, DINOv2, K = full (Stage 1) | 0.7637 | - |

On DTD the layer overtakes the ResNet-18 probe trained on the same full split, while still
classifying by cosine against frozen text prototypes. On Aircraft it does not: 0.2845 against
0.3662. The layer can only move embeddings towards a fixed target geometry it does not
control, and on 100 fine-grained variants that geometry is the bottleneck - the prototypes
themselves are too close together. A probe, by contrast, is free to place its own decision
boundaries.

**Which objective to prefer.** Rolled-out FM at T = 4, if there is enough data. It is the
best cell in both tables at K = full and is the only variant that clearly separates from the
others (+0.05 on Aircraft). At K = 5 the two objectives are indistinguishable, and standard
FM is cheaper: one training per (dataset, K, seed) instead of one per T, and one network call
per step instead of T chained calls with a retained graph.

**Why fewer Euler steps win for the rolled-out objective.** Two effects point the same way.
The T=12 rollout is a 12-deep chain of the same network, which optimises measurably worse -
its final endpoint loss is at least as high as T=4's in all 18 pairs, and strictly higher in
17. And with 4 steps the map from `z_i` to `z_T` is a coarser, more constrained function,
which is a form of regularisation on a layer that has 1.31 M parameters and, at K = full on
DTD, only 1880 training examples.

**Caveats.**
- The field is far larger than the Stage 1 probe it is compared against (1.31 M parameters
  vs 48 k for a CLIP linear probe), yet at K = 5 it improves only on the *zero-shot*
  baseline, not on any probe. The comparison that matters is FM vs zero-shot, which is the
  one the brief asks for.
- K = full seeds vary only initialisation and batch order, since the subset is the whole
  split - so those error bars measure optimisation noise, not sampling noise.
- Everything is a single encoder and a single prompt template, inherited from Stage 1.

---

## Deliverable Checklist

| Brief item | Where |
| --- | --- |
| Standard FM loss, `z_t` and `u_i` per spec | Section 1, `models/flow_matching_clip.py` |
| Euler inference `z_{k+1} = z_k + (1/T) v(z_k, k/T)`, T in {4, 12} | Section 1, `models/flow_ode.rollout` |
| Rolled-out training, same T at train and test | Section 1, `train/train_flow_clip.batch_loss` |
| Small MLP, 2 x 512, SiLU, scalar t concatenated | Config table, `models/flow_matching_clip.VelocityField` |
| K in {5, 10, full}, Stage 1 subsets and seeds | Section 2 |
| Accuracy table + acc-vs-K plot with error bars and dAcc | Sections 2 and 3 |
| Training-loss curves | Section 4 |
| 3-way feature visualisation, joint projection | Section 5 |
| Flow trajectories for individual examples | Section 6 |
| Optional: reverse flow from prototypes | Section 8 |

---

## Reproducing

Run `notebooks/stage2_flow_matching.ipynb` on Colab. It only needs the cached `clip_rn50`
features from Stage 1 (`train`, `val`, `test` and the text prototypes per dataset); cell 4
rebuilds them if they are missing, so the notebook is standalone. The datasets themselves are
needed only for the reverse-retrieval figure in Section 8.

Results are appended to the shared `runs.csv` under `method = "fm_clip_standard"` or
`"fm_clip_rolled"` with T in the `steps` column, and aggregated into
`<results>/flow_accuracy_table.csv`. Each run also writes its full history to
`<results>/flow_curves/` and its field to `<results>/flow_ckpt/`, so every figure can be
rebuilt without retraining.

Subset indices come from the same persisted Stage 1 files, so the K-shot subsets are
identical to the ones the baselines used; initialisation, batch order and time sampling are
seeded per run.
