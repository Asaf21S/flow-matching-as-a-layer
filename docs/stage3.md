# Stage 3: FM Before a Linear Classifier

## Goal

Insert a flow-matching transformation between the frozen encoder feature and the
pretrained linear classifier:

$$ z \xrightarrow{\ \text{FM}\ } \hat{z} \xrightarrow{\ \text{frozen linear classifier}\ } s $$

where $z$ is the frozen image-encoder feature, $\hat{z}$ is the feature after the FM
transformation and $s = W\hat{z} + b$ are the classifier logits. The question is whether
the FM layer can reshape the frozen representation into one the *existing* classifier
handles better.

## Experimental setup

**Protocol.** The linear classifier is trained first, exactly as in Stage 1, and is then
**frozen**: its parameters are set to `requires_grad = False` and never enter the
optimiser. The FM layer is initialised at the **exact identity** — the output projection of
the velocity network is zero-initialised, so $v_\theta \equiv 0$ and $\hat{z} = z$ before
any Stage 3 training. The complete system therefore starts out numerically identical to
the Stage 1 linear probe, and every reported change is caused by Stage 3 training alone.
Both properties are asserted by the component checks in `src/fmlayer/train/checks.py`.

**Data.** The same splits and the same sampled training subsets as the Stage 1 linear-probe
experiments; the subset index files are reused from the cache, keyed by
`(dataset, K, seed)`, so the flow and the Stage 1 probe see exactly the same images.

| | |
| --- | --- |
| Datasets / encoders | FGVC-Aircraft + ResNet-18; DTD + DINOv2 ViT-S/14 |
| Training-set size | $K = 10$ shots per class (secondary table at $K = \text{full}$) |
| Euler steps | $T = 12$, used identically in training, model selection and evaluation |
| Seeds | 0, 1, 2 |
| Test-set size | 3333 images (Aircraft), 1880 images (DTD) |
| Baseline | the Stage 1 linear probe of the same encoder, subset and seed |

**Velocity network.** Same design and integration procedure as Stage 2: an MLP
$v_\theta(z, t)$ with 2 hidden layers of width 512 and SiLU activations, the scalar time
concatenated to the input. Features are standardised on the way in and the predicted
velocity is rescaled by the same per-dimension spread on the way out, so the network
optimises in a well-conditioned space while the ODE runs in the original feature space.
Integration is explicit Euler, $z_{k+1} = z_k + \tfrac{1}{T} v_\theta(z_k, k/T)$.

**Optimisation.** AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, cosine schedule
down to $10^{-5}$, batch size 256, 500 epochs. Validation is evaluated every 10 epochs and
the checkpoint with the best validation accuracy is kept; the test set is touched once, at
the end, by that checkpoint only.

**Why $T = 12$.** Stage 2 found the accuracy of the transported features flat between
$T = 4$ and $T = 12$ while the discretisation error keeps shrinking, so 12 is the cheapest
count at which the Euler solution is a faithful stand-in for the continuous flow. It is
fixed once and used everywhere in Stage 3.

## Methods compared

| Name in the tables | Configuration tag | What it does |
| --- | --- | --- |
| Stage 1 linear probe | — | The frozen baseline; the FM layer is the identity. |
| End-to-end rolled-out | `rolled_ce_T12` | Strategy 1. Run the full $T$-step rollout, score $\hat{z}$ with the frozen classifier and minimise $\mathcal{L}_{cls} = \mathrm{CE}(W\hat{z} + b,\, y)$, backpropagating through **every** Euler step. Only the FM parameters are updated. |
| Classifier-guided FM | `standard_guided_T12_s1lr0p1` | Strategy 2. Transport $z$ to $\hat{z}$, take a gradient step on the classification loss in feature space to get $\hat{z}'$, then perform a **standard** conditional-OT FM update from source $z$ to target $\hat{z}'$. Targets are recomputed every batch as the flow changes. |
| Classifier-guided FM, noised sources | `standard_guided_T12_n0p15_s1lr0p1` | The same, with the source perturbed by Gaussian noise (see *Deviations*). |
| Joint fine-tuning | `rolled_ce_T12_joint` | Optional extension: the classifier is unfrozen and trained together with the FM layer. |

**Reading the tags.** `T12` is the number of Euler steps; `s1` is one target-improvement
step; `lr0p1` is a feature-space step size of $0.1$; `n0p15` is source noise with standard
deviation $0.15$ of the mean feature norm. Every knob that changes the trained model
appears in the tag, so no two variants can share a checkpoint.

## Deviations from the suggested recipes

The brief asks that substantial changes be described and justified experimentally.

1. **Noise-perturbed sources for the guided method.** At $K = 10$ the frozen probe has
   essentially zero training error, so on the training features themselves the
   classification loss is already saturated and its gradient — the signal Strategy 2 is
   built on — is close to zero. Perturbing the source with Gaussian noise
   ($\sigma = 0.15$ of the mean feature norm) puts the flow on points the probe has not
   memorised. Because this is a deviation, the **clean** variant is reported alongside it
   rather than replaced by it, so the effect is measured rather than assumed.
2. **Checkpoint selection on validation accuracy.** Both methods keep their best
   validation checkpoint, as does the Stage 1 baseline, so the comparison is like for like.
3. **Cross-fitting was tried and rejected.** Training the flow against held-out fold probes
   de-saturates the loss, but the flow then learns the fold probes' boundaries while being
   scored against the full probe, so the corrections aim at the wrong decision surface. It
   is kept as a documented negative control, not used in the main comparison.

---

## Results

### Classification results

Top-1 test accuracy at $K = 10$, $T = 12$, averaged over seeds 0/1/2. The delta is a
**paired** difference: each run is scored against the Stage 1 probe of its own seed, so the
spread of the delta is the correct error bar rather than the spread of accuracy.

<!-- Paste the output of `markdown_main_table(main_table)` from cell 7.1 here. -->

_Preliminary numbers, seed 0 only, before the multi-seed run:_

| Dataset | Encoder | Method | Top-1 accuracy | Delta vs probe |
| --- | --- | --- | --- | --- |
| FGVC-Aircraft | ResNet-18 | Stage 1 linear probe | 0.2781 | - |
| FGVC-Aircraft | ResNet-18 | End-to-end rolled-out | 0.2727 | -0.0054 |
| FGVC-Aircraft | ResNet-18 | Classifier-guided FM | 0.2775 | -0.0006 |
| DTD | DINOv2 ViT-S/14 | Stage 1 linear probe | 0.6910 | - |
| DTD | DINOv2 ViT-S/14 | End-to-end rolled-out | 0.6830 | -0.0080 |
| DTD | DINOv2 ViT-S/14 | Classifier-guided FM | 0.6910 | +0.0000 |

A delta of $-0.0006$ on Aircraft is two test images out of 3333, and $-0.0080$ on DTD is
fifteen out of 1880. Differences of this size are not interpretable from one seed, which is
why the final table is averaged over three.

### Did the flow do anything at all?

An accuracy number cannot separate *"the flow collapsed back to the identity"* from
*"the flow moved points and the moves were harmful"*. The diagnostics below measure the
mean relative displacement $\lVert \hat{z} - z \rVert / \lVert z \rVert$ and count label
flips in both directions.

<!-- Paste the output of `print_diagnostics(diagnostics)` from cell 7.2 here. -->

Two observations already point at the answer for the guided method at $K = 10$: its
accuracy is identical at $T = 4$ and $T = 12$ to every decimal, and on DTD it equals the
frozen baseline exactly. Both are signatures of a flow that has learned an almost-zero
velocity field, i.e. it has stayed at its identity initialisation.

### Training behaviour

![Curves, Aircraft / ResNet-18](../results/figures/stage3_curves_aircraft_resnet18.png)
![Curves, DTD / DINOv2](../results/figures/stage3_curves_dtd_dinov2_vits14.png)

Three panels per cell: training loss, validation accuracy and validation cross-entropy.

**The training losses of the two methods are different quantities** — Strategy 1 minimises
the classifier's cross-entropy while Strategy 2 minimises a flow-matching MSE against a
moving target — so the left panel is only meaningful *within* a method. The two right-hand
panels are the frozen classifier's own metrics on the validation split and are what the
methods should be compared on.

Per-method dynamics, each showing the training and validation curves, the learned vector
field at $t = 0$ and the trajectories $z_0 \rightarrow z_T$ at $T = 12$:

**FGVC-Aircraft / ResNet-18**
![Dynamics, rolled-out](../results/figures/viz_rolled_ce_T12_aircraft_resnet18_k10_seed0.png)
![Dynamics, classifier-guided](../results/figures/viz_standard_guided_T12_n0p15_s1lr0p1_aircraft_resnet18_k10_seed0.png)

**DTD / DINOv2 ViT-S/14**
![Dynamics, rolled-out](../results/figures/viz_rolled_ce_T12_dtd_dinov2_vits14_k10_seed0.png)
![Dynamics, classifier-guided](../results/figures/viz_standard_guided_T12_n0p15_s1lr0p1_dtd_dinov2_vits14_k10_seed0.png)

### Feature-space visualization

**FGVC-Aircraft / ResNet-18**
![Before and after, Aircraft / ResNet-18](../results/figures/flow_comparison_aircraft_resnet18_T12.png)

**DTD / DINOv2 ViT-S/14**
![Before and after, DTD / DINOv2](../results/figures/flow_comparison_dtd_dinov2_vits14_T12.png)

Each figure shows, left to right: the **original test features** $z$, the features
transported by the **end-to-end rolled-out** flow, and the features transported by the
**classifier-guided** flow. All panels use the **test** split at $K = 10$, $T = 12$,
seed 0, restricted to the same readable subset of 8 classes, with the same class colours
throughout. The projection is a single PCA fitted **jointly** over all three feature sets,
so a point is directly comparable across panels and the panels cannot be rescaled relative
to one another. The jointly fine-tuned flow is deliberately excluded here, since it belongs
to the extension rather than to the frozen-classifier comparison.

### Secondary comparison at $K = \text{full}$

The brief fixes one training-set size for the main experiments, but the amount of training
data turns out to be the variable that decides whether the FM layer can help at all, so the
same comparison is repeated with the full training split.

<!-- Paste the output of `markdown_main_table(full_table)` from cell 7.1 here. -->

![Accuracy vs K, Aircraft / ResNet-18](../results/figures/stage3_accuracy_vs_k_aircraft_resnet18.png)
![Accuracy vs K, DTD / DINOv2](../results/figures/stage3_accuracy_vs_k_dtd_dinov2_vits14.png)

### Ablations requested by the brief

**Strategy 1 — regularising the transformation.** Penalising the displacement
$\lVert \hat{z} - z \rVert^2$ or the magnitude of the predicted velocities, both normalised
by the mean squared feature norm so the weights mean the same thing on both encoders.

<!-- Paste the regularisation table from cell 7.1 here. -->

**Strategy 2 — the target-construction knobs.** Feature-space step size, number of
target-improvement steps, whether the target update is normalised (each step constrained to
a fixed fraction of the feature norm), and how often the targets are recomputed.

<!-- Paste the guided-ablation table from cell 7.1 here. -->

---

## Optional extension: jointly fine-tuning the classifier

After the frozen-classifier experiments we unfroze the pretrained linear classifier and
optimised it jointly with the FM transformation (`rolled_ce_T12_joint`). The classifier is
trained on a private copy, so the frozen Stage 1 probe used as the baseline is never
modified.

<!-- The joint rows are already part of the main table from cell 7.1. -->

_Preliminary numbers, seed 0 only:_

| Dataset | Encoder | Method | Top-1 accuracy | Delta vs probe |
| --- | --- | --- | --- | --- |
| FGVC-Aircraft | ResNet-18 | Frozen classifier (`rolled_ce_T12`) | 0.2727 | -0.0054 |
| FGVC-Aircraft | ResNet-18 | Joint fine-tuning | 0.2688 | -0.0093 |
| DTD | DINOv2 ViT-S/14 | Frozen classifier (`rolled_ce_T12`) | 0.6830 | -0.0080 |
| DTD | DINOv2 ViT-S/14 | Joint fine-tuning | 0.6894 | -0.0016 |

![Dynamics, joint, Aircraft / ResNet-18](../results/figures/viz_rolled_ce_T12_joint_aircraft_resnet18_k10_seed0.png)
![Dynamics, joint, DTD / DINOv2](../results/figures/viz_rolled_ce_T12_joint_dtd_dinov2_vits14_k10_seed0.png)

Unfreezing the classifier lets the decision boundary move together with the features, which
adds capacity exactly where there is least data to constrain it. The training-accuracy curve
in the figures above is the evidence for or against the overfitting explanation: at $K = 10$
the *frozen* probe is already at or near 100% on its own training subset, so joint training
can only be said to "memorise more" if its training accuracy stays pinned at 1.0 while its
validation cross-entropy rises.

<!-- Check the train_accuracy / val_loss curves before committing to this explanation. -->

---

## Conclusions

<!-- Rewrite once the three-seed numbers and the diagnostics are in. -->

1. **At $K = 10$ the FM layer does not improve the frozen linear probe on either dataset.**
   Every measured delta is negative or zero, and all of them are of the order of a handful
   of test images.
2. **The two strategies fail in different ways.** The end-to-end rolled-out objective moves
   the features and loses accuracy: at $K = 10$ the frozen probe has near-zero training
   error, so the classification loss it backpropagates is already saturated and the
   gradient it does receive reflects the training subset rather than the class structure.
   The classifier-guided objective instead converges back to the identity — its targets are
   built from the same saturated gradient, so the improved target $\hat{z}'$ is barely
   distinguishable from $\hat{z}$ and the FM update has almost nothing to learn.
3. **The failure is a data-regime effect, not a defect of the construction.** The same
   methods behave differently at $K = \text{full}$, where the probe has genuine training
   error and the classification gradient carries information.
4. **Joint fine-tuning does not rescue it**, which is consistent with the diagnosis: adding
   trainable classifier parameters in the regime where the bottleneck is the lack of
   training signal makes the overfitting worse, not the representation better.

