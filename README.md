# flow-matching-as-a-layer

Generative **Flow Matching (FM)** repurposed as a trainable structural layer inside an image
classifier.

## Executive summary

Flow Matching is normally a *generative* tool: it learns a velocity field $v_\theta(z,t)$ that
transports one distribution into another. This project asks whether that same machinery can be
made *discriminative* — whether a learned ODE, applied to a frozen encoder's embedding, can
reshape the feature manifold into one an existing classifier handles better.

Everything upstream of the flow stays frozen: the encoder, and (in Stage 3) the classifier. The
FM layer is initialised at the exact identity, so any change in accuracy is caused by the flow
alone.

- **Stage 1** — frozen-encoder baselines (linear probe, zero-shot CLIP). No flow matching.
- **Stage 2** — FM transports a CLIP image embedding onto the CLIP **text** prototype of its
  class; classification is cosine 1-NN. Large gains.
- **Stage 3** — FM inserted **before a frozen linear probe**. Small but statistically supported
  gains, plus a measurement of *when* the layer can help at all.

**Setup:** DTD (47 classes) and FGVC-Aircraft (100 classes) · ResNet-18 / DINOv2 ViT-S/14 /
CLIP RN50 · K ∈ {5, 10, full} shots × seeds {0, 1, 2} · top-1 on the official test split.

## Consolidated results

### Stage 1 — baselines (top-1, K = full)

| Dataset | Encoder | Linear probe | Zero-shot CLIP RN50 |
| --- | --- | --- | --- |
| DTD | ResNet-18 | 0.6284 ± 0.0045 | 0.4005 |
| DTD | DINOv2 ViT-S/14 | **0.7637 ± 0.0083** | 0.4005 |
| FGVC-Aircraft | ResNet-18 | **0.3662 ± 0.0027** | 0.1545 |

### Stage 2 — zero-shot CLIP + FM (best: rolled-out, T = 4, K = full)

| Dataset | Zero-shot baseline | + FM layer | Δ |
| --- | --- | --- | --- |
| DTD | 0.4005 | **0.6668 ± 0.0011** | **+0.2663** |
| FGVC-Aircraft | 0.1545 | **0.3299 ± 0.0034** | **+0.1754** |

### Stage 3 — linear probe + FM (T = 12, paired Δ, `*` = Δ larger than 2σ)

| Dataset / encoder | K | Best method | Accuracy | Δ vs probe |
| --- | --- | --- | --- | --- |
| Aircraft / ResNet-18 | full | Joint fine-tuning (extension) | 0.3772 ± 0.0046 | +0.0110 ± 0.0048 \* |
| Aircraft / ResNet-18 | full | End-to-end rolled-out (frozen probe) | 0.3751 ± 0.0011 | +0.0089 ± 0.0023 \* |
| Aircraft / ResNet-18 | 10 | Displacement-regularised rolled-out | — (probe 0.2720) | +0.0096 ± 0.0039 \* |
| DTD / DINOv2 ViT-S/14 | 10 | Probe-weight target | — (probe 0.6952) | +0.0094 ± 0.0020 \* |
| DTD / DINOv2 ViT-S/14 | full | — | 0.7637 ± 0.0083 | no method beats the probe |

**Takeaways**

- Stage 2 is the headline: **+26.6 points on DTD** and **+17.5 points on Aircraft** over
  zero-shot CLIP, with the encoder and the text prototypes both frozen.
- Stage 3 gains are small because at K = 10 the frozen probe sits at **100 % training accuracy**
  — its cross-entropy gradient is numerically zero, so a flow initialised at the identity
  correctly stays there. Quantifying this identity collapse is the stage's main contribution.
- The prediction that follows is confirmed at K = full: the *same* objective that lost 0.0119 at
  K = 10 gains **+0.0089** on Aircraft, a swing of +0.0208 driven purely by training-set size.
- Overall, **the FM layer helps in proportion to the headroom the frozen classifier leaves** —
  most on a weak encoder with plenty of data, not at all on a strong encoder already at its
  linear ceiling (DTD / DINOv2 at 76 %).

## Navigation

- [Stage 1 — classification baselines](docs/stage1.md)
- [Stage 2 — FM onto CLIP text prototypes](docs/stage2.md)
- [Stage 3 — FM before a linear probe](docs/stage3.md)

Code lives in `src/fmlayer/`, runnable pipelines in `notebooks/`, figures and tables in
`results/`.
