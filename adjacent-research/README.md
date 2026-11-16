# Adjacent research

Papers and notes for work that this project must position itself against.
Each entry should answer three questions:

1. What did they actually show?
2. How is their setup different from ours?
3. What does our project need to reproduce, cite, or avoid claiming?

---

## MOSAIC — *When Do Diffusion Models Learn to Generate Multiple Objects?*

**Paper:** `jeong-2026-mosaic.pdf` (this folder)
**Authors:** Yujin Jeong, Arnas Uselis, Iro Laina, Seong Joon Oh, Anna Rohrbach
**Venue:** ICML 2026, PMLR 306:51250–51282
**Links:**
- Proceedings: https://proceedings.mlr.press/v306/jeong26b.html
- PDF: https://raw.githubusercontent.com/mlresearch/v306/main/assets/jeong26b/jeong26b.pdf
- Code/data: https://github.com/eugene6923/MOSAIC

### Citation

```bibtex
@InProceedings{pmlr-v306-jeong26b,
  title     = {When Do Diffusion Models Learn to Generate Multiple Objects?},
  author    = {Jeong, Yujin and Uselis, Arnas and Laina, Iro and Oh, Seong Joon and Rohrbach, Anna},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {51250--51282},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

### The one-paragraph version

MOSAIC is a controlled synthetic dataset and training protocol for asking
**why text-to-image diffusion models fail at multi-object generation**. It
isolates three compositional skills — colour attribution, spatial relations, and
counting — and trains small latent diffusion models from scratch on controlled
data distributions that vary dataset size, concept imbalance, scene complexity,
and the number of held-out concept combinations. Its headline findings are that
**scene complexity matters more than concept imbalance**, that **counting is
uniquely fragile in low-data regimes** (it peaks early and then degrades), and
that **compositional generalization collapses as more combinations are held out**.
The difficulty ordering for recombination is **attribution < counting < spatial
relations**.

### Why the name

**MOSAIC** = **M**ulti-**O**bject **S**patial relations, **A**ttr**I**bution,
**C**ounting. It is both a dataset generator and a diagnostic framework.

### The dataset

MOSAIC is built on a Blender-based simulator (COMFORT) so every scene has exact
ground-truth annotations. It has three task subsets:

| Subset | What is varied | What is fixed | Number of classes |
|---|---|---|---|
| **Attribution** | colours of a sphere and a cube | object identities (sphere/cube), positions randomized | 10 colours × 10 colours = 100 pairs |
| **Spatial Relations** | angular position of a second sphere around a brown reference sphere | colour, shape, size | 10 angular bins of 18°, with 18° gaps |
| **Counting** | number of spheres, 1–10 | colour, shape, size, camera | 10 counts |

Dataset variants:

- **Base** — the controlled setting above.
- **Complex** — more objects (2–10), duplicating objects for attribution or
  adding distractor spheres for relations. This raises scene complexity to match
  counting.
- **Grid** — objects constrained to radial cells with small jitter. This adds a
  spatial prior and lowers positional variability, mainly used for counting.
- **Composition** — adds a second conditioning factor (colour × relation, or
  colour × count) so compositional generalization can be measured.

Dataset sizes: **2k, 10k, 50k, 100k** (the abstract says different sizes; the
experiments use these). Distributions: **uniform** or **skewed**. The skewed
distribution is modelled on LAION-2B count frequencies: low counts are more
frequent; the same skew shape is applied to angle bins and colour pairs.

### The model and training setup

- Latent diffusion with a frozen pretrained **Stable Diffusion 2.0 VAE**.
- Two backbones:
  - **U-Net** (~90M parameters), trained with a score-matching objective —
    comparable to SD 2.0.
  - **DiT** (Diffusion Transformer), trained with **flow matching** — comparable
    to SD3.
- Conditioning: each concept is a **one-hot vector**, passed through a small
  condition encoder and injected through attention layers. For two concepts,
  two tokens are concatenated.
- Condition encoder and diffusion model are trained jointly.
- Evaluation: task-specific discriminative classifiers (a CNN for counting,
  ImageNet-pretrained ResNets for attribution and spatial relations). The paper
  verifies classifier accuracy on VAE reconstructions is near 100%.
- Memorization is measured with pixel-wise nearest-neighbour distances between
  generated and training images.

### Research questions

- **RQ1 — Concept generalization:** if every concept is seen at least once, can
  the model learn it reliably? How does concept imbalance affect this?
- **RQ2 — Compositional generalization:** if all individual concepts are seen,
  can the model recombine them into **unseen pairs**? How does this degrade as
  more pairs are held out?

For RQ2, the paper holds out **diagonals** of the concept × concept matrix:
0, 1, 3, 5, or 8 diagonals removed. Every individual concept remains observed;
only specific combinations are absent. The remaining cells are resampled so the
total dataset size stays comparable.

### Findings in detail

#### 1. Concept generalization: complexity dominates imbalance

- Attribution and spatial relations reach **>90% accuracy** at every dataset
  size and under both uniform and skewed distributions, for both U-Net and DiT.
- Counting behaves differently:
  - at **2k** examples it is near-perfect (memorization),
  - at **10k–50k** it **drops sharply**,
  - at **100k** it partially recovers.
- This is largely **independent of skewness**. The failure is not "rare count
  labels"; it is the transition from memorization to generalization.
- Increasing scene complexity (the **Complex** setting) also hurts attribution
  and spatial relations at 10k, but less than counting. So scene complexity is a
  major driver.
- The **Grid** layout, which constrains object positions, substantially improves
  counting at all sizes. A spatial prior helps.

Learning dynamics: counting accuracy **peaks early and then deteriorates**
while training loss keeps decreasing. This is an optimization/performance
misalignment: the model still fits the objective but loses the ability to keep
multiple objects distinct. Higher counts collapse first.

#### 2. Compositional generalization collapses with held-out combinations

- On **seen** compositions, attribution and spatial relations stay strong;
  counting is lower and more variable.
- On **unseen** compositions, accuracy falls as:
  - dataset size decreases, and
  - the number of held-out diagonals increases.
- Difficulty ordering: **Attribution < Counting < Spatial Relations**.
- Confusion matrices:
  - Attribution errors are local in colour space (purple ↔ red/blue) — the
    concept is partially learned.
  - Counting errors are off by one or two instances — near-misses.
  - Spatial relations show broad, unstructured confusion — the relation is not
    cleanly represented.
- Data scaling helps only partially; it does not remove the held-out-combination
  penalty.

#### 3. The trends survive more realistic settings

- Fine-tuning **SD3-medium with LoRA** on SPEC (realistic text–image pairs):
  spatial-relation accuracy improves with training, but counting accuracy
  **degrades** even as training loss falls. The controlled trend persists.
- A less-controlled **MOSAIC OBJECTS** variant with real object categories,
  varying camera distance, and occlusions: unseen object pairs still degrade
  rapidly, and the model often collapses to a single object.

### The three takeaways (paper's own framing)

1. Concept learning depends on **scene complexity and dataset size**, not
   primarily concept imbalance. Counting peaks early and degrades; attribution
   and relations converge more stably. Reducing scene complexity (fewer objects,
   grid layout) helps.
2. As more compositions are held out, compositional generalization degrades,
   especially for **spatial relations**. This looks like a fundamental limitation
   of current diffusion models, not just a data-frequency issue.
3. The behaviours persist under realistic appearance, viewpoint, and occlusion
   changes.

### What MOSAIC does *not* claim

- It is not a study of **marginal-preserving exposure manipulation**. It varies
  dataset size, skew, complexity, and held-out combinations; it does not hold
  predicate marginals fixed while varying joint exposure/lift.
- It does not study **rule-level exposure thresholds (N50)**.
- It does not provide a **composition-gap** estimand against a product-of-
  marginals baseline.
- It does not test **exposure allocation as constrained optimization**.
- It is a controlled diagnostic benchmark, not a claim about natural-image
  training at scale.

These gaps are exactly where "Which Exposure Matters?" positions itself — but
the proposal explicitly says MOSAIC overlap risk is highest and must be checked
against the full paper and released code before any novelty claim.

### How this project uses MOSAIC

1. **Closest prior work.** The proposal's Stage 0 includes "a small replication
   of MOSAIC's complexity-versus-imbalance contrast on this renderer". The
   expected go condition is that the replication matches MOSAIC's direction.
2. **Design inspiration.** The project's `mosaic_*` modules build a
   MOSAIC-inspired 2×2 contrast: `{low, high} scene complexity × {balanced,
   imbalanced} concept distribution`. It is explicitly **not** a replication:
   the renderer (2D shapes), architecture (`PilotUNet`), objective, and
   evaluator all differ.
3. **Hypothesis support.** MOSAIC's finding that scene complexity dominates
   imbalance motivates controlling complexity (`K_attr`, `K_rel`) in this
   project's benchmark and statistical model.
4. **Caveat for talks.** Do not say "we replicate MOSAIC". Say "we built a
   MOSAIC-inspired contrast; a formal adapted replication is still outstanding."

### Current status in this repo

- `docs/mosaic-replication.md` — source review; no replication run.
- `docs/mosaic-data-feasibility.md` — toy dataset feasibility.
- `docs/mosaic-matched-81570.md` and `docs/mosaic-distributed-81570.md` —
  bounded four-cell matched experiment on two GPUs (low/high × balanced/
  imbalanced), exploratory only.
- `docs/stage0-status.md` — MOSAIC gate listed as **not run / no result**.

### Questions to ask when reading the full paper

- How exactly is "concept imbalance" operationalised, and is it comparable
  across attribution, relations, and counting?
- Are the classifier-based evaluations sensitive to the same failures a human
  would see?
- Does the "counting peaks then degrades" pattern depend on the fixed 100k
  budget, the architecture, or the VAE?
- What happens under a **marginal-preserving** manipulation, where skew is held
  fixed but joint co-occurrence changes? (This project's question.)
- How much of the compositional collapse is explained by **memorization** rather
  than a compositional deficit?

---

## Other adjacent work (to add)

The proposal also positions against:

- **Okawa, Lubana, Dick, Tanaka (2023).** *Compositional Abilities Emerge
  Multiplicatively.* NeurIPS 2023. arXiv:2310.09336. Source of the
  product-of-constituents independence baseline.
- **Nützel, Dombrowski, Kainz (2025).** *GRASP: Guided Residual Adapters with
  Sample-wise Partitioning.* arXiv:2512.01675. Gradient conflict / long-tail
  fine-tuning.
- **Bradley (2026).** *Local Mechanisms of Compositional Generalization in
  Conditional Diffusion.* ICML 2026, PMLR 306:9531–9565. Local conditional
  scores and compositional generalization.
- **Qin et al. (2023).** *Class-Balancing Diffusion Models.* CVPR 2023.
- **Hang et al. (2023).** *Efficient Diffusion Training via Min-SNR Weighting.*
  ICCV 2023.
- **Go et al. (2023).** *Addressing Negative Transfer in Diffusion Models.*
  NeurIPS 2023.

Add PDFs here as they are read, with the same three-question note format.
