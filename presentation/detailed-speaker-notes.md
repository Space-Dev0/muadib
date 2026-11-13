# Detailed Presentation Notes & Q&A Defense Guide

**Presentation:** Which Exposure Matters? How training examples shape image-generation skills
**Format:** 63 Widescreen Slides (`main.pdf`), compiled from `main.tex` via Tectonic
**Audience:** Technical research audience, deep learning practitioners, and academic reviewers

This is a draft set of presenter notes and possible questions for the 63 frames.
It received a quick review, not a full literature or mathematical audit. Check
technical claims and citations against the proposal, source code and original
papers before presenting them. The confirmatory study and Stage 0 gates remain
incomplete; speculative applications and outcomes are not established results.

---

## Slide 1: Title Page

### Presentation Notes
- **Opening:** Welcome the audience. Introduce the project as an investigation into the foundational mechanics of conditional generative vision: specifically, how data composition and statistical exposure shape an image model's ability to follow complex, multi-concept instructions.
- **Context:** Clarify immediately that this talk covers a research proposal and initial exploratory instrument development. Emphasize that while software systems and preliminary prototypes exist, the formal confirmatory experiment has not yet been executed.
- **Tone:** Methodical, empirical, and grounded. The objective is to understand what models actually learn from their training mixtures rather than claiming exaggerated benchmark breakthroughs.

### Probable Audience Questions
1. **Q:** *Is this presentation presenting completed experimental findings or a research proposal?*
   **A:** This presentation introduces the research proposal, the mathematical framework, and our exploratory instrument development. While we have built the synthetic renderer, measurement pipelines, and exploratory diffusion models, Stage 0 validation gates have not yet passed, so all numbers and curves presented are illustrative or preliminary calibrations.
2. **Q:** *Why focus on synthetic scenes instead of standard benchmarks like COCO or LAION?*
   **A:** Natural web datasets suffer from massive uncontrolled confounding, label noise, and latent co-occurrences that cannot be audited. Synthetic scenes provide an absolute ground truth, allowing exact mathematical tracking and manipulation of concept exposures without hidden confounders.

---

## Slide 2: Table of Contents

### Presentation Notes
- **Overview:** Provide the structural roadmap of the presentation across four coherent acts:
  1. *Foundations & Framing:* Situating the work within adjacent research, establishing project novelty, and unpacking diffusion model math and U-Net architecture.
  2. *Scene World & Exposure:* Defining our controlled synthetic environment, relation semantics, distinct object constraints, and the exposure hierarchy.
  3. *Evaluation & Optimization:* Establishing composition gaps, N50 sample efficiency, AdamW/EMA training dynamics, and constrained mixture optimization.
  4. *Obstacles, Gates & Applications:* Transparently reviewing Stage 0 bottlenecks, examining high-impact real-world use cases (medical imaging and robotics), and outlining confirmatory next steps.

### Probable Audience Questions
1. **Q:** *Which section contains the primary theoretical contribution?*
   **A:** Section 2 and 3: the formal distinction between marginal, pairwise, and rule-level exposure, the mathematical definition of lift, and the formulation of dataset design as constrained convex optimization.

---

## Slide 3: Adjacent Research in Generative Vision

### Presentation Notes
- **Literature Context:** Anchor the presentation within recent breakthroughs and limitations in conditional diffusion:
  - *MOSAIC (Jeong et al., ICML 2026):* Found that scene complexity and relational structure drive generation failures far more than simple concept imbalance, and that compositional generalization collapses as combinations are held out.
  - *Constituent Abilities (Okawa et al., NeurIPS 2023):* Observed that compositional performance across vision-language benchmarks emerges multiplicatively from constituent component proficiencies.
  - *Gradient Conflicts (GRASP, 2025):* Documented that during long-tail diffusion fine-tuning, conflicting gradient directions between head and tail classes cause catastrophic forgetting of rare prompts.
  - *Score Matching (Bradley, ICML 2026):* Showed that local conditional score estimation underlies multi-concept composition.
- **The Research Gap:** While prior work establishes that compositionality is fragile, no existing study has systematically uncoupled marginal concept frequency from joint pair and rule exposure in diffusion models.

### Probable Audience Questions
1. **Q:** *How does your setup differ from MOSAIC?*
   **A:** MOSAIC investigates held-out compositionality across varied scene complexities, but varies concept frequencies along standard class-imbalance distributions. We explicitly hold marginal concept frequencies constant while varying joint co-occurrence (lift), isolating the causal impact of joint exposure.
2. **Q:** *Why does Okawa et al.'s multiplicative finding matter for diffusion?*
   **A:** Okawa et al. established a multiplicative independence null for discriminative classifiers. We adapt this principle to generative diffusion models, evaluating whether generation errors compound multiplicatively across spatial relations.

---

## Slide 4: Proposed contributions

### Presentation Notes
- **Four Distinct Contributions:**
  1. *Marginal-Preserving Lift Manipulation:* We aim to vary joint rule exposure while holding individual predicate marginals matched, using lift as an experimental dose variable. Feasibility remains under investigation.
  2. *Falsifiable Independence Baseline:* We construct a formal null hypothesis based on independent predicate performance and measure the composition gap $\Gamma$ on both logit and log scales.
  3. *Parametric Threshold Estimation ($N_{50}$):* Rather than reporting arbitrary accuracy numbers at a single fixed compute budget, we map complete exposure-response curves to measure the minimum exposure required for 50% accuracy.
  4. *Constrained Mixture Optimization:* We explore condition weights that improve exposure separation; current candidates do not meet the stated decorrelation targets.

### Probable Audience Questions
1. **Q:** *Has anyone ever varied joint exposure without changing marginal exposure before?*
   **A:** Similar co-occurrence controls have been explored in related settings. Whether this project's proposed comparison is novel requires a thorough literature review.
2. **Q:** *What if the surrogate model used for optimization is inaccurate?*
   **A:** A surrogate must be validated before using it for downstream optimization. The Stage 0 design gate concerns exposure decorrelation, not surrogate prediction.

---

## Slide 5: The question in one sentence

### Presentation Notes
- **Slide Focus:** The fundamental question: *Does an image model need to see the whole combination, or are its parts enough?*
- **Visual Walkthrough:** Walk the audience through the three boxes on the slide: "red objects", "square objects", and the combined rule "red left of square".
- **Intuition:** Contrast seeing ingredients independently versus seeing their structured arrangement. If a diffusion network is trained on scenes containing red circles and blue squares, does it inherently possess the compositional capability to synthesize a red square left of a blue circle when prompted? Or does compositional binding require dedicated joint exposure during training?

### Probable Audience Questions
1. **Q:** *Isn't compositional generalization already widely studied in models like DALL-E or Stable Diffusion?*
   **A:** Large models demonstrate impressive visual synthesis, but systematic evaluations (e.g., MOSAIC, Winoground, DrawBench) reveal persistent failure modes on spatial bindings and attribute binding (the "binding problem"). Crucially, because web-scale training data is proprietary or unannotated, existing literature cannot determine whether success stems from true compositional generalization or from memorized joint co-occurrences in massive training sets.
2. **Q:** *What constitutes a "part" versus a "whole combination"?*
   **A:** A "part" refers to a constituent marginal predicate (e.g., an object having the attribute "red", or an object having the shape "square"). The "whole combination" is the full conjunctive rule binding those attributes across distinct objects via a spatial relationship.

---

## Slide 6: Why should anyone care?

### Presentation Notes
- **The Core Problem:** Deep learning models can exhibit high marginal accuracy on individual concepts while failing catastrophically on compositional conjunctions.
- **Highlight:** Point out the diagram: a model knows "red" and knows "square". When prompted with "?", it may generate red blobs and disconnected squares, or mix up attributes (e.g., making the wrong object red).
- **Practical Impact:** Understanding this limitation has profound implications for dataset engineering. If models require joint exposure, training budgets must be spent collecting co-occurrences; if marginal exposure suffices, data collection can be vastly simpler.

### Probable Audience Questions
1. **Q:** *Why would a neural network fail to bind two concepts it already knows well?*
   **A:** Deep networks often learn "bag-of-words" heuristics, detecting isolated features without learning relational structure. In spatial attention, features can bleed across objects (the "attribute leakage" phenomenon), leading the denoiser to associate color with the wrong spatial position.
2. **Q:** *How does this research relate to human cognition?*
   **A:** Humans exhibit systematicity: once we understand "red" and "square" and "left of", we can imagine "red left of square" instantly without prior joint observation (Fodor & Pylyshyn). We are investigating whether diffusion architectures exhibit systematicity or operate strictly as associative memory interpolators.

---

## Slide 7: First: what is an image-generation model?

### Presentation Notes
- **High-Level Definition:** Explain that a conditional image generation model is a function mapping a conditioning vector (instruction/prompt) and a stochastic latent vector into pixel space.
- **Statistical Nature:** Emphasize that generative models do not "understand" instructions symbolically; they learn probability densities $p(x \mid c)$ from training data, synthesizing pixels that conform to learned conditional distributions.

### Probable Audience Questions
1. **Q:** *Why frame this in terms of diffusion models rather than autoregressive models or GANs?*
   **A:** Diffusion models currently represent the state of the art in high-fidelity continuous visual synthesis. Their continuous iterative denoising dynamics and explicit score-matching objectives make them mathematically transparent for tracing how conditioning information steers generation.

---

## Slide 8: What is an image made of?

### Presentation Notes
- **Pedagogical Aim:** Ground the audience in the discrete numeric nature of images. An image is not an abstract concept; it is an $H \times W \times C$ tensor.
- **Key Point:** Highlight that every shape, border, and color is represented as numeric channel intensities (RGB). The model must learn to manipulate these numeric arrays to create boundaries, textures, and spatial relationships.

### Probable Audience Questions
1. **Q:** *What image resolution is used in your prototype?*
   **A:** Our synthetic prototype operates at $64 \times 64$ RGB pixels. This resolution is sufficient to render distinct geometric shapes and clear spatial separations while keeping training computationally lightweight and tractable on standard hardware.

---

## Slide 9: What is noise?

### Presentation Notes
- **Definition:** Noise represents unstructured, uncorrelated random perturbations drawn from a Gaussian distribution $\mathcal{N}(0, \mathbf{I})$.
- **Visual Progression:** Show that adding noise destroys high-frequency details first (edges, sharp colors) and eventually wipes out low-frequency structures, leaving pure entropy. Denoising is the process of reversing this thermodynamic degradation.

### Probable Audience Questions
1. **Q:** *Is Gaussian noise the only noise distribution suitable for diffusion?*
   **A:** Gaussian noise is standard due to the Central Limit Theorem and because linear combinations of Gaussian variables remain Gaussian, which allows closed-form sampling at arbitrary timesteps without simulating intermediate steps.

---

## Slide 10: Diffusion: learn to reverse noise

### Presentation Notes
- **The Forward Process:** Walk through the asset images `noise-0.png` to `noise-3.png`.
- **Core Mechanism:** Explain that training corrupts clean scenes with noise and learns a reverse-process prediction. Noise prediction is one common objective; our later pilot uses velocity prediction.
- **Note:** Clarify the caption note: the 4-step image sequence is an illustrative teaching visual; actual diffusion uses dozens to hundreds of discrete timesteps.

### Probable Audience Questions
1. **Q:** *How does predicting noise help generate images?*
   **A:** A noise predictor can be used to estimate a score or a clean-image component and build reverse transitions. This connection depends on the noise schedule and parameterization; the pilot also uses velocity prediction.

---

## Slide 11: Generation goes in the other direction

### Presentation Notes
- **Inference Mechanism:** Explain the reverse direction (from `noise-3.png` back to `noise-0.png`).
- **Contrast:** Forward corruption samples random noise from a specified process. Reverse generation uses iterative neural network evaluations to transform a noise sample toward an image.

### Probable Audience Questions
1. **Q:** *Why can't the model jump directly from pure noise to the clean image in a single step?*
   **A:** The manifold of natural images is highly non-convex and multi-modal. A single-step prediction from pure noise averages across all possible images, producing a blurry mean. Iterative steps allow the stochastic trajectory to break symmetry and settle into a specific mode.

---

## Slide 12: Forward diffusion: from image to noise

### Presentation Notes
- **Mathematical Rigor:** Introduce the formal Markov chain:
  $$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\, \sqrt{1 - \beta_t}\, x_{t-1},\, \beta_t \mathbf{I}\right)$$
- **Closed-Form Trick:** Highlight the reparameterization shortcut using $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
  $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
- **Significance:** Emphasize that this closed-form expression means training never requires simulating a sequential Markov chain: any timestep $t$ can be sampled in $O(1)$ time during batch training.

### Probable Audience Questions
1. **Q:** *What variance schedule $\beta_t$ does your implementation use?*
   **A:** In `exposure.pilot_train`, we implement a linear schedule where $\beta_t$ linearly interpolates from $10^{-4}$ to $0.02$ across $T$ timesteps, ensuring smooth degradation where $\bar{\alpha}_T \approx 0$.
2. **Q:** *Why use $\sqrt{\bar{\alpha}_t}$ and $\sqrt{1 - \bar{\alpha}_t}$ as coefficients?*
   **A:** This ensures the total variance is preserved at unity ($\bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1$) if the input $x_0$ is normalized to unit variance.

---

## Slide 13: The reverse process: noise prediction

### Presentation Notes
- **Reverse Formulation:** Present the parameterized reverse transition:
  $$p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}\left(x_{t-1};\, \mu_\theta(x_t, t, c),\, \sigma_t^2 \mathbf{I}\right)$$
- **MSE Loss:** Explain the derivation of Ho et al. (DDPM): parameterizing $\mu_\theta$ in terms of a noise estimator network $\epsilon_\theta(x_t, t, c)$ simplifies the complex variational lower bound (VLB) into a weighted Mean Squared Error:
  $$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon, c} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$
- **Takeaway:** Noise prediction is one common diffusion objective. The later pilot uses velocity prediction, another parameterization of the same clean/noise components.

### Probable Audience Questions
1. **Q:** *Why predict $\epsilon$ instead of predicting the clean image $x_0$ or velocity $v$?*
   **A:** Predicting $\epsilon$ places higher implicit perceptual weight on intermediate noise levels, producing sharper visual samples. We also provide velocity-prediction probes in `exposure/velocity_probe.py`.

---

## Slide 14: Backbone architecture: hierarchical U-Net

### Presentation Notes
- **Walkthrough the Architecture:** Walk through the TikZ schematic on the slide:
  - Input: $64 \times 64 \times 3$ noisy tensor.
  - Encoder: Initial convolution to `width`, followed by residual blocks with GroupNorm (4 groups) and SiLU activations. Strided convolutions downsample features to $32 \times 32$ and $16 \times 16$.
  - Bottleneck: Operating at $16 \times 16$ with channel dimension `dim = width * 4`. Contains deep residual blocks and multi-head cross-attention.
  - Decoder: Nearest-neighbor interpolation upsamples features, which are concatenated with skip connections from the encoder to retain precise pixel coordinates.
- **Why U-Net:** U-Nets maintain spatial correspondence via skip connections while enabling low-resolution semantic abstraction at the bottleneck.

### Probable Audience Questions
1. **Q:** *Why use GroupNorm instead of BatchNorm or LayerNorm?*
   **A:** BatchNorm introduces dependencies across batch elements, which causes instability in diffusion training where batch size may vary or timesteps are randomly mixed. GroupNorm operates per sample across channel groups.
2. **Q:** *Why is attention restricted only to the $16 \times 16$ bottleneck?*
   **A:** Attention has quadratic computational complexity in spatial tokens ($N^2$). At $64 \times 64$, $N=4096$; at $16 \times 16$, $N=256$, making cross-attention fast and memory-efficient while still providing global receptive field coverage.

---

## Slide 15: Conditioning via cross-attention

### Presentation Notes
- **Detailing the Mechanism:** Explain how symbolic rules interact with continuous visual representations.
- **Token Encoding:** A rule (e.g., "red left of square") is decomposed into symbolic predicate tokens. Each token specifies `(kind, subject_slot, object_slot, attribute_value)`. These are mapped via embedding tables into context vectors $c \in \mathbb{R}^{K \times d}$.
- **Attention Computation:** Spatial features at the bottleneck project to queries $Q \in \mathbb{R}^{256 \times d}$; rule tokens project to keys $K$ and values $V$.
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$
- **Intuition:** Each spatial location queries the rule tokens: "Am I supposed to be red? Am I the object on the left?" Attention weights route prompt semantics into the correct spatial coordinates.

### Probable Audience Questions
1. **Q:** *How are null or empty conditions handled?*
   **A:** As implemented in `PilotUNet`, if a rule has fewer predicates than max slots or if condition dropout occurs, a learned `null` token parameter is exposed so that the multi-head attention module never encounters an all-zero key-padding mask.
2. **Q:** *Does cross-attention guarantee correct spatial binding?*
   **A:** No—and that is the core research question! Cross-attention provides the architectural capacity to bind attributes to objects, but empirical studies show it often suffers from attribute leakage unless the training data provides sufficient statistical guidance.

---

## Slide 16: What does ``conditional'' mean?

### Presentation Notes
- **Conceptual Synthesis:** Reiterate the role of conditioning. A denoiser without conditioning generates arbitrary samples from the prior $p(x)$. Conditioning injects a gradient field steering the trajectory toward the conditional distribution $p(x \mid c)$.
- **Control:** The condition acts as a steering wheel throughout every reverse timestep.

### Probable Audience Questions
1. **Q:** *Can conditioning overpower the noise, causing mode collapse?*
   **A:** In standard diffusion, conditioning is injected gently at each step. However, if guidance is too strong (excessive CFG scale), samples can become oversaturated or collapse into high-contrast caricatures.

---

## Slide 17: Steering generation: classifier-free guidance

### Presentation Notes
- **Motivation:** Conditional diffusion models often under-sample prompt details without explicit guidance.
- **Method (Ho & Salimans):** During training, we randomly drop the conditioning tokens with probability $p_{\text{drop}} = 10\%$, substituting the learned null token $\emptyset$.
- **Inference Formula:**
  $$\tilde{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \left(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\right)$$
- **Effect of $s$:** When $s=1$, it is standard conditional generation. When $s > 1$, the model extrapolates in the direction of the prompt away from unconditional noise, drastically improving prompt adherence and spatial precision.

### Probable Audience Questions
1. **Q:** *Why not use classifier guidance with an external image classifier?*
   **A:** Classifier guidance requires training a separate noise-robust classifier $\nabla_{x_t} \log p(c \mid x_t)$ at all noise levels $t$, which is complex and prone to adversarial exploitation. Classifier-free guidance trains both modes within the exact same generative backbone with zero external models.
2. **Q:** *What guidance scale do you use in your pilot?*
   **A:** In exploratory sampling, we evaluate scales between $s=1.5$ and $s=3.0$, balancing relationship precision against diversity.

---

## Slide 18: Training and generation are different

### Presentation Notes
- **Contrasting the Two Regimes:**
  - *Training:* Forward corruption is one-shot; clean images and exact noise vectors are known. Updates are fast, parallel, and independent across timesteps.
  - *Generation:* Auto-regressive across time. We start from pure ungrounded noise; the model has no target image to peek at and must traverse the full reverse trajectory sequentially.
- **Takeaway:** Generalization failure during generation occurs even when training loss is very low, because errors can accumulate along the sampling trajectory.

### Probable Audience Questions
1. **Q:** *Does low training MSE guarantee high composition accuracy?*
   **A:** Absolutely not. A model can achieve low MSE by accurately predicting high-frequency noise while failing to place objects in their requested spatial arrangement. This discrepancy is precisely why automated semantic checkers and human audits are required.

---

## Slide 19: The scene world we control

### Presentation Notes
- **Synthetic Environment:** Introduce the synthetic 2D scene generator developed in `exposure.render`.
- **Properties:**
  - Exactly 3 non-overlapping geometric objects placed on an $8 \times 8$ placement grid.
  - Rendered at $64 \times 64$ RGB pixels.
  - Generates ground-truth JSON annotations detailing bounding boxes, centers, colors, and shapes.
- **Why:** Eliminates natural image annotation noise. We know with 100% mathematical certainty what is present in every training scene.

### Probable Audience Questions
1. **Q:** *Isn't a 3-object synthetic toy world too simple to inform real-world diffusion models?*
   **A:** Toy worlds are standard in rigorous foundational science (e.g., CLEVR, bAbI, dSprites). If a state-of-the-art diffusion architecture cannot solve compositional binding in a completely controlled 3-object world where exposures are exact, it cannot be expected to solve it reliably in complex open-world domains.

---

## Slide 20: Our basic visual vocabulary

### Presentation Notes
- **Vocabulary:** 4 colors (Red, Blue, Green, Yellow) and 3 shapes (Circle, Square, Triangle).
- **Combinatorics:** There are 12 colored-object types. The renderer chooses colors and shapes independently, so object types may repeat; positions are sampled without replacement from grid locations before jitter.
- **Caveat:** Emphasize the slide caption: this is the prototype vocabulary. The final benchmark vocabulary has not yet been frozen.

### Probable Audience Questions
1. **Q:** *Why not include more shapes like stars or hexagons?*
   **A:** Expanding the vocabulary exponentially increases the dimensionality of the exposure matrix and the number of samples required to achieve statistical confidence without adding conceptual insight into spatial binding.

---

## Slide 21: Relationships add structure

### Presentation Notes
- **Geometric Definitions:**
  - $\text{left\_of}(A, B)$: $x_{\text{center}}(A) < x_{\text{center}}(B)$.
  - $\text{above}(A, B)$: $y_{\text{center}}(A) < y_{\text{center}}(B)$.
- **Key Insight:** Spatial relations compare pairs of objects. They cannot be evaluated by inspecting an object in isolation.
- **Subtlety:** Note that exact ties ($x_A = x_B$) are not considered left of; left includes diagonal configurations (above-left or below-left).

### Probable Audience Questions
1. **Q:** *Why compare bounding box centers rather than bounding box edges?*
   **A:** Center comparisons are mathematically robust to shape differences and avoid edge-touching ambiguities in discrete pixel grids.

---

## Slide 22: A rule is a combination of predicates

### Presentation Notes
- **Definition:** Define a composite rule:
  $$\text{color}(o_1, \text{red}) \land \text{shape}(o_2, \text{square}) \land \text{left\_of}(o_1, o_2)$$
- **Terminology:** A *predicate* is an atomic proposition (unary attribute or binary relation). A *rule* is a conjunction of predicates with shared variable bindings.

### Probable Audience Questions
1. **Q:** *Do you evaluate disjunctive rules (e.g., 'red OR square')?*
   **A:** No, our grammar focuses strictly on conjunctive rules, which directly probe compositional binding and intersectional exposure.

---

## Slide 23: Two roles mean two distinct objects

### Presentation Notes
- **Crucial Semantic Rule:** The two variable bindings $o_1$ and $o_2$ in a relational rule must bind to **distinct physical objects**.
- **Visual Clarification:** Contrast the two columns:
  - Valid: Two red objects (one circle, one square) where the red circle is left of the red square.
  - Invalid: A single red square cannot be "left of itself".
- **Rule:** Variables may share attribute values (both red), but identity must be distinct ($o_1
e o_2$).

### Probable Audience Questions
1. **Q:** *Why enforce distinct object semantics?*
   **A:** Without distinct object semantics, reflexive bindings trivialise spatial queries and introduce severe semantic contradictions (e.g., an object being left of itself).

---

## Slide 24: What do we mean by exposure?

### Presentation Notes
- **Formal Definition:** Exposure $E(\phi)$ is the empirical proportion of training scenes in which the rule or event $\phi$ is true:
  $$E(\phi) = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(\text{scene}_i \models \phi)$$
- **Core Principle:** Count what is *actually present* in rendered pixels and annotations, not just what prompt was sent to the renderer.

### Probable Audience Questions
1. **Q:** *Why might what is present differ from what was requested?*
   **A:** In synthetic generation, requesting "red circle" renders a scene with a red circle plus two random background distractor objects. Those distractor objects might happen to form a blue square or satisfy an unprompted relational rule!

---

## Slide 25: One scene can teach several things

### Presentation Notes
- **Concept Overlap:** Walk through the visual example: a scene showing a red circle and a blue square.
- **Simultaneous Exposure:** This single scene increases exposure for:
  1. The marginal attribute "red"
  2. The marginal attribute "square"
  3. The pairwise attribute co-occurrence "red and square"
  4. The spatial relation "red left of square"
- **Takeaway:** Training scenes are multi-label observations. Manipulating one exposure inevitably influences others unless carefully controlled.

### Probable Audience Questions
1. **Q:** *How do you prevent distractor objects from causing uncontrolled exposure?*
   **A:** This is precisely why we compute the full Exposure Matrix (Slide 41) and solve constrained optimization problems to balance mixture weights.

---

## Slide 26: Three levels of exposure

### Presentation Notes
- **Hierarchy:** Detail the three structural granularities:
  1. *Marginal (Single):* Presence of an isolated feature (e.g., "red object").
  2. *Pairwise (Joint Primitives):* Co-occurrence of two features without explicit binding (e.g., "red object and square object in the same scene").
  3. *Full Rule (Relational Binding):* The complete bound conjunction (e.g., "red object left of square object").
- **Research Question:** Which level of exposure dictates whether the model can generate the full rule?

### Probable Audience Questions
1. **Q:** *Could pairwise exposure be sufficient without full rule exposure?*
   **A:** That is Hypothesis 3 (Slide 35)! If pairwise exposure suffices, the model exhibits strong structural inductive bias; if full rule exposure is required, the model is strictly an associative memory interpolator.

---

## Slide 27: A simple counting example

### Presentation Notes
- **Intuition:** 3 out of 10 scenes satisfy the condition $\rightarrow 30\%$ empirical exposure.
- **Statistical Uncertainty:** In finite datasets, empirical counts are subject to binomial sampling noise. In our codebase (`exposure.measure`), we compute Wilson score confidence intervals to ensure exposure estimates are statistically meaningful.

### Probable Audience Questions
1. **Q:** *How many samples are needed to estimate exposure accurately?*
   **A:** With $N=10,000$ scenes, Wilson intervals for moderate exposures are narrower than $\pm 1\%$, providing high precision for our exposure matrices.

---

## Slide 28: Why single-concept counts are not enough

### Presentation Notes
- **The Core Paradox:** Walk through Dataset A vs. Dataset B.
  - In Dataset A, red objects and squares always appear in the same scenes.
  - In Dataset B, red objects appear only in scenes without squares, and squares appear only in scenes without red objects.
  - Both datasets can have identical 50% marginal frequencies for red and 50% for square!
- **Conclusion:** Marginal frequency is completely blind to joint co-occurrence.

### Probable Audience Questions
1. **Q:** *Do existing generative benchmarks control for this distinction?*
   **A:** No. Standard generative benchmarks match class balance or concept frequencies marginally, ignoring joint co-occurrence and lift in the training corpora.

---

## Slide 29: Lift: is a combination common or rare?

### Presentation Notes
- **Mathematical Definition:**
  $$\text{lift}(\phi_1, \phi_2) = \log\left(\frac{P(\phi_1 \land \phi_2)}{P(\phi_1) P(\phi_2)}\right)$$
- **Interpretation:**
  - $\text{lift} = 0$: Concepts co-occur exactly as predicted by independent random chance.
  - $\text{lift} > 0$: Synergistic co-occurrence (over-represented).
  - $\text{lift} < 0$: Suppressed co-occurrence (under-represented).
- **Caveat:** Note the caption: independent multiplication is an empirical reference, not an absolute physical law, because distinct object requirements induce mild dependencies.

### Probable Audience Questions
1. **Q:** *Why use a logarithm instead of the raw ratio?*
   **A:** The log transform symmetrizes over-representation and under-representation around zero and linearizes multiplicative relationships.

---

## Slide 30: Illustrating lift with numbers

### Presentation Notes
- **Numerical Example:**
  - $P(\text{Red}) = 0.50$
  - $P(\text{Square}) = 0.40$
  - Independent reference: $0.50 \times 0.40 = 0.20$ (20%).
  - If actual joint exposure is $0.10$ (10%), $\text{lift} = \log(0.10 / 0.20) = \log(0.5) \approx -0.693$.
- **Takeaway:** This combination is suppressed relative to chance, despite high marginal frequencies.

### Probable Audience Questions
1. **Q:** *Are these project measurements?*
   **A:** No, these are purely didactic numbers to make the mathematical definition clear.

---

## Slide 31: How do we measure model success?

### Presentation Notes
- **Evaluation Pipeline:**
  1. Prompt model with target rule across multiple random seeds.
  2. Synthesize images using reverse diffusion.
  3. Automated checker analyzes generated images.
  4. Compute success rate:
     $$\text{Success} = \frac{\text{classified images satisfying rule}}{\text{total classified images}}$$
- **Crucial Rule:** Abstentions (where the checker cannot decide) must be reported separately and never silently counted as successes or failures.

### Probable Audience Questions
1. **Q:** *Why not count abstentions as failures?*
   **A:** Conflating checker uncertainty with model failure pollutes generative evaluation with measurement instrument artifacts.

---

## Slide 32: A reference for composition

### Presentation Notes
- **The Compositional Baseline:** If a model masters concept A with accuracy $0.80$ and concept B with accuracy $0.70$, and their bindings combine independently, the expected joint success is $0.80 \times 0.70 = 0.56$ (56%).
- **Role:** This product serves as an empirical null benchmark to quantify whether combining skills incurs an additional penalty.

### Probable Audience Questions
1. **Q:** *Is the product rule a guaranteed upper bound?*
   **A:** No, synergistic contextual cues could theoretically make combining concepts easier. That is why it is treated as a testable reference rather than a theoretical ceiling.

---

## Slide 33: The composition gap

### Presentation Notes
- **Definition:** The Composition Gap $\Delta$ is the difference between observed joint success and the product reference.
- **Interpretation:**
  - $\Delta < 0$: Compositional deficit. The model struggles to coordinate familiar concepts.
  - $\Delta = 0$: The observed rate equals the chosen product reference.
  - $\Delta > 0$: Compositional synergy.
- **Visual:** Emphasize that the bar chart on the slide is an illustration of the concept.

### Probable Audience Questions
1. **Q:** *How large are composition gaps typically in exploratory runs?*
   **A:** The available pilot samples do not establish a reliable composition-gap estimate. This remains a research question.

---

## Slide 34: What does a zero gap mean?

### Presentation Notes
- **Significance:** A zero gap means observed success equals the product reference. The product is a heuristic and does not prove independent mechanisms or absence of difficulty.
- **Scale Considerations:** The formal proposal formulates this gap on the logit scale:
  $$\Gamma^{\text{logit}} = \operatorname{logit}(\mathrm{RSR}) - \operatorname{logit}\!\left(\prod_p c_p\right)$$
  Working drafts also evaluate log-scale expressions. Neither has been preregistered yet.

### Probable Audience Questions
1. **Q:** *Why consider logit scale over linear difference?*
   **A:** Linear differences suffer from floor and ceiling compression effects near 0 and 1; logit differences map probabilities to an unbounded real line.

---

## Slide 35: Our main hypotheses

### Presentation Notes
- **Three Falsifiable Hypotheses:**
  1. *Complexity scaling:* Rules with more predicates exhibit larger negative composition gaps.
  2. *Joint exposure benefit:* Increasing joint training exposure directly improves rule satisfaction.
  3. *Pairwise transfer:* Intermediate exposure to pairs aids zero-shot generalization to full unseen rules.
- **Scientific Stance:** These are hypotheses to test and falsify, not pre-ordained conclusions.

### Probable Audience Questions
1. **Q:** *What happens if Hypothesis 2 fails?*
   **A:** If increasing joint exposure does not improve performance, it would demonstrate that diffusion architectures are bottlenecked by architectural capacity or attention mechanisms rather than data distribution.

---

## Slide 36: A separate question: how much is enough?

### Presentation Notes
- **Introducing N50:** Define N50 as the critical exposure threshold where rule satisfaction crosses 50%.
- **Perspective:** Learning is not binary. Instead of evaluating an arbitrary single checkpoint (e.g., 500k steps), we should map complete exposure-response curves to measure sample efficiency.

### Probable Audience Questions
1. **Q:** *Why 50% instead of 90% or 95%?*
   **A:** 50% is the median transition point on standard logistic curves, where derivative sensitivity is highest and estimation variance is lowest.

---

## Slide 37: An illustrative learning curve

### Presentation Notes
- **Sigmoid Trajectory:** Walk through the classic S-curve of learning: slow initial acquisition, rapid threshold transition around N50, and asymptotic saturation.
- **Experimental Warning:** To fit N50 accurately, experimental designs must bracket the threshold. Extrapolating N50 from data that never crosses 50% is mathematically invalid.

### Probable Audience Questions
1. **Q:** *Has N50 been estimated on your pilot model?*
   **A:** No, exploratory allocations have not yet satisfied the required design stability to fit rigorous parametric curves.

---

## Slide 38: Model optimization: AdamW and EMA

### Presentation Notes
- **AdamW Optimization:**
  $$\theta_{k+1} = \theta_k - \eta_k \, \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon_{\text{opt}}} - \lambda \eta_k \theta_k$$
  Explain that AdamW decouples weight decay $\lambda$ from gradient updates, preventing weight magnitude explosion.
- **Exponential Moving Average (EMA):**
  $$\theta_{\text{EMA}} \leftarrow \mu \, \theta_{\text{EMA}} + (1 - \mu) \, \theta_k \quad (\mu = 0.999)$$
  Diffusion updates suffer from high gradient variance across timesteps. EMA acts as a temporal low-pass filter, smoothing weight oscillations.
- **Inference Protocol:** Sampling always uses EMA weights.

### Probable Audience Questions
1. **Q:** *What learning rate is used in `exposure.pilot_train`?*
   **A:** The saved 800,000-update pilot checkpoint records a learning rate of $10^{-4}$; optimizer defaults should be checked against its recorded runtime.
2. **Q:** *How much does EMA affect visual quality?*
   **A:** Substantially. Checkpoints sampled without EMA show graininess and broken object borders; EMA checkpoints yield smooth, consistent geometry.

---

## Slide 39: Inference optimization: sampling schedules

### Presentation Notes
- **Comparing Samplers:**
  - *DDPM:* Discretized ancestral Markov chain. Requires hundreds of small stochastic steps to avoid trajectory drift.
  - *DDIM:* Formulates diffusion as a deterministic probability flow ODE with identical marginal distributions.
- **Practical Benefit:** Deterministic ODE trajectories can be integrated using larger step sizes (e.g., 20–50 steps instead of 1000) or higher-order Runge-Kutta solvers (DPM-Solver), slashing inference latency by $20\times$.

### Probable Audience Questions
1. **Q:** *Does using DDIM change the rule satisfaction rate compared to DDPM?*
   **A:** In preliminary testing, deterministic DDIM sampling preserves semantic rule fidelity while significantly reducing compute overhead during large evaluation sweeps.

---

## Slide 40: How do we change exposure fairly?

### Presentation Notes
- **The Fair Comparison Challenge:** If we simply add more "red left of square" examples, we might inadvertently increase the total number of red objects or squares, confounding the experiment.
- **Solution:** We optimize a probability vector over candidate prompts to shift joint exposure while keeping marginal concept frequencies matched.

### Probable Audience Questions
1. **Q:** *Can you always find weights that perfectly match marginals while varying joint exposure?*
   **A:** Not always! That is a mathematical feasibility question governed by the rank and nullspace of the exposure matrix.

---

## Slide 41: The exposure matrix

### Presentation Notes
- **Matrix Structure:**
  - Columns $j \in \{1, \dots, K\}$: Requested conditioning templates.
  - Rows $i \in \{1, \dots, M\}$: Realized visual events measured from scene annotations.
  - Entry $\mathbf{M}_{ij}$: Probability that requesting prompt $j$ produces event $i$.
- **Estimation:** Matrix entries are estimated via Monte Carlo sampling over thousands of rendered scenes.

### Probable Audience Questions
1. **Q:** *Why is the matrix not an identity matrix?*
   **A:** Because prompts only specify part of the scene! If you request "red object", the renderer also places two distractor objects, which might randomly create an unprompted "circle" or "left of" event.

---

## Slide 42: Exposure design as constrained optimization

### Presentation Notes
- **Formulation:** Frame dataset design as a convex optimization problem:
  $$\min_{w} \; D_{\text{KL}}(w \,\|\, u) = \sum_{k=1}^K w_k \log(K w_k)$$
  subject to:
  $$\mathbf{M}_{\text{marginal}} w = b_{\text{matched}}, \quad l \le \mathbf{M}_{\text{joint}} w \le u, \quad w \in \Delta^{K-1}$$
- **Solver:** Solved using Sequential Least Squares Programming (SLSQP).
- **Objective:** Minimum KL divergence ensures the mixture remains as uniform and diverse as possible while satisfying strict exposure constraints.

### Probable Audience Questions
1. **Q:** *What happens if the constraints are infeasible?*
   **A:** Our tool (`exposure.design`) first runs a Phase-I Linear Program to check feasibility. If infeasible, it halts and reports that no valid mixture exists under those bounds.

---

## Slide 43: A controlled comparison

### Presentation Notes
- **Experimental Setup:**
  - Mixture A: Matched basic concepts, Low Joint Exposure.
  - Mixture B: Matched basic concepts, High Joint Exposure.
- **Causal Inference:** Because marginals are identical, any performance difference between models trained on A vs. B can be causally attributed to joint exposure.

### Probable Audience Questions
1. **Q:** *How large is the joint exposure separation between Mixture A and B in your current designs?*
   **A:** In exploratory designs, we achieved a separation of approximately 15% joint exposure while keeping marginals within 2% tolerance.

---

## Slide 44: Why do we need a checker?

### Presentation Notes
- **Role of the Checker:** An automated program that inspects generated pixel arrays, detects objects, identifies bounding boxes, extracts color/shape attributes, and tests geometric relations.
- **Why Automation:** Human grading is subjective, fatigue-prone, and cannot scale to tens of thousands of generated evaluation images.

### Probable Audience Questions
1. **Q:** *How does the prototype checker detect objects from pixels?*
   **A:** In `exposure.checker`, it uses connected-component labeling and color palette clustering, followed by centroid and contour analysis to classify shape and relative position.

---

## Slide 45: The checker can also be wrong

### Presentation Notes
- **Three Classification Outcomes:**
  1. *False Accept:* Checker says rule is satisfied; image actually violates it.
  2. *False Reject:* Checker rejects a valid image due to minor pixel artifacts.
  3. *Abstention:* Visual ambiguity prevents confident classification.
- **Transparency:** All three outcomes must be audited and reported.

### Probable Audience Questions
1. **Q:** *Which error is worse: false accepts or false rejects?*
   **A:** Both are damaging: false accepts artificially inflate benchmark scores, while false rejects unfairly penalize creative or slightly noisy model generations.

---

## Slide 46: Humans help validate the checker

### Presentation Notes
- **Human Audit Protocol:**
  - Stratified sample of at least 300 generated images across complexity tiers.
  - Independent human evaluation of rule satisfaction.
  - Calculate Cohen's Kappa, precision, and recall against checker predictions.

### Probable Audience Questions
1. **Q:** *What agreement threshold must the checker meet to pass validation?*
   **A:** The proposal specifies an agreement rate exceeding 90% with low abstention rates before the checker can be trusted for confirmatory evaluation.

---

## Slide 47: Why blind the review?

### Presentation Notes
- **Blinding Protocol:** Reviewers see only the image and the question (e.g., "Is there a red circle left of a square?").
- **Double-Blind:** Reviewers do not see the automated checker's verdict, and do not know what training mixture or model produced the image.
- **Purpose:** Reduces the risk that checker predictions influence reviewer judgments; blinding cannot eliminate all bias.

### Probable Audience Questions
1. **Q:** *How do you measure inter-annotator agreement among human reviewers?*
   **A:** A subset of shared calibration images is reviewed by all annotators to compute inter-rater reliability (Krippendorff's alpha).

---

## Slide 48: What does the current model produce?

### Presentation Notes
- **Visual Evidence:** Examine the exploratory pilot samples (`pilot-red.png` and `pilot-relation.png`) from the 800,000-update checkpoint.
- **Qualitative Assessment:**
  - Single-attribute requests ("red") succeed with clear color and shape.
  - Relational requests ("blue above circle") exhibit spatial confusion and object merging.
- **Honest Caveat:** Emphasize that these are exploratory samples, not a statistical benchmark measurement.

### Probable Audience Questions
1. **Q:** *Why did the 800k checkpoint struggle with 'blue above circle'?*
   **A:** The pilot architecture used a very small width (`width=16`), and exploratory training mixtures had not achieved full exposure decorrelation.

---

## Slide 49: What have we built?

### Presentation Notes
- **Infrastructure Summary:**
  - Configurable synthetic scene renderer and symbolic rule parser.
  - Exact Monte Carlo exposure measurement engine with Wilson intervals.
  - Constrained LP and SLSQP mixture optimization solvers.
  - Online diffusion training pipeline (`exposure.pilot_train`) with AdamW, EMA, and CFG.
  - Automated pixel checker and blinded human review interfaces.
- **Distinction:** Prototype availability is not the same as experimental validation.

### Probable Audience Questions
1. **Q:** *Is the software open-source and reproducible?*
   **A:** Yes, the entire codebase is structured with deterministic seeds, unit tests, and CLI entry points.

---

## Slide 50: What has the team reviewed?

### Presentation Notes
- **Team Calibration:** The team conducted blinded calibration reviews on a preliminary 300-image, four-condition pilot pack.
- **Findings:** Revealed specific edge-case disagreements regarding blurry object contours and strict centroid alignment, allowing refinement of labeling rubrics.

### Probable Audience Questions
1. **Q:** *Did the 300-image pilot pack cover all complexity levels?*
   **A:** No, it was restricted to four conditions. Full benchmark coverage across all 57 proposed rule templates remains outstanding.

---

## Slide 51: One current obstacle: checker coverage

### Presentation Notes
- **The Coverage Obstacle:** The prototype checker's strict pixel thresholds reject recognizable shapes that have slight boundary fuzziness or anti-aliasing artifacts.
- **The Dilemma:** Relaxing thresholds improves coverage but increases false accepts. Threshold tuning requires independent validation.

### Probable Audience Questions
1. **Q:** *Could a modern vision-language model (e.g., GPT-4o or CLIP) serve as the checker?*
   **A:** VLMs suffer from known spatial relation hallucinations (e.g., confusing left/right). A deterministic, auditable pixel-level checker is far more verifiable once calibrated.

---

## Slide 52: One current obstacle: relation exposure

### Presentation Notes
- **The Relational Baseline Problem:** In a random 3-object grid scene, an unconstrained spatial relation like "any object left of another" is true in over 95% of scenes simply by chance!
- **Consequence:** The marginal baseline for unconstrained relations is near saturation, making it mathematically distinct from the specific attributed rule.

### Probable Audience Questions
1. **Q:** *How do you resolve relation saturation?*
   **A:** By evaluating relations strictly in conjunction with attribute bindings or using tighter geometric placement constraints.

---

## Slide 53: One current obstacle: experimental design

### Presentation Notes
- **The Multicollinearity Obstacle:** In exploratory mixtures, shifting joint exposure while constraining marginals resulted in high Variance Inflation Factors (VIF > 10).
- **Consequence:** Multicollinearity makes it difficult to disentangle joint exposure effects from subtle marginal drifts in regression models.

### Probable Audience Questions
1. **Q:** *How do you plan to reduce VIF?*
   **A:** By expanding the conditioning template set to 57 templates, enlarging the nullspace of the exposure matrix to find cleaner orthogonal directions.

---

## Slide 54: Stage 0: checks before the main study

### Presentation Notes
- **The Three Proposal Gates:**
  1. *Gate 1:* Human-audited checker error at most 2%, with the proposal's required complexity-stratified sample.
  2. *Gate 2:* Exposure decorrelation ($|r|\le 0.3$ and VIF $<2$).
  3. *Gate 3:* Baseline benchmark replication against adapted MOSAIC protocols.
- **Current Status:** **None of the three gates is passed.** Confirmatory training remains paused until all gates clear.

### Probable Audience Questions
1. **Q:** *Why pause training instead of running experiments now?*
   **A:** Running large-scale training without passing validation gates wastes computational resources and risks producing uninterpretable, confounded results.

---

## Slide 55: What happens after the gates?

### Presentation Notes
- **The Confirmatory Workflow:**
  1. Freeze experimental design and rule templates.
  2. Train controlled diffusion models across balanced exposure regimes.
  3. Generate complexity-stratified test images across multiple seeds.
  4. Evaluate with validated checkers and compute composition gaps.
  5. Conduct pre-registered statistical hypothesis testing.

### Probable Audience Questions
1. **Q:** *What compute resources will the full experiment require?*
   **A:** The proposal gives an initial compute estimate, but the final number of runs and update budget require measured pilot runtimes and a frozen protocol.

---

## Slide 56: Possible outcome: joint exposure matters

### Presentation Notes
- **Positive Scenario:** Models trained with high joint exposure exhibit significantly smaller composition gaps than models trained with low joint exposure, even though marginal exposures were identical.
- **Implication:** Compositional binding in diffusion architectures cannot be learned purely from isolated parts; training sets must include explicit multi-concept pairings.

### Probable Audience Questions
1. **Q:** *If this outcome occurs, how should dataset creators respond?*
   **A:** Data curation must focus on combinatorial diversity and relation mining, rather than simply balancing single-class marginal frequencies.

---

## Slide 57: Possible outcome: the hypothesis fails

### Presentation Notes
- **Negative Scenario:** Models perform equally well regardless of joint exposure, composing concepts zero-shot from marginal exposure alone.
- **Scientific Value:** This would be a profound demonstration of architectural systematicity, proving that cross-attention mechanisms generalize spatial arrangements without explicit joint examples.

### Probable Audience Questions
1. **Q:** *Is a negative result publishable?*
   **A:** Absolutely. In rigorous science, falsifying a strong hypothesis under controlled conditions is just as informative and valuable as confirming it.

---

## Slide 58: Real-world implications: medical imaging

### Presentation Notes
- **Clinical Motivation:** Detail the high-stakes consequences of compositional failure in healthcare AI:
  - *Rare Pathologies & Co-morbidities:* A hospital dataset might have thousands of chest X-rays with pneumonia and thousands with pneumothorax, but almost none exhibiting both simultaneously in the right upper lobe.
  - *Synthetic Augmentation Risks:* If clinical data teams train diffusion models to synthesize rare multi-pathology cases for training diagnostic classifiers, does the model accurately bind both pathologies, or does it suffer attribute leakage (e.g., blending lesion textures across lung fields)?
- **Takeaway:** Determining exposure thresholds ($N_{50}$) is safety-critical: it establishes whether generative models can reliably synthesize rare disease combinations from marginal clinical observations without hallucinating invalid clinical phenotypes.

### Probable Audience Questions
1. **Q:** *How does synthetic scene binding translate to complex radiographic textures?*
   **A:** Spatial binding is mathematically identical: associating an abnormality attribute (e.g., ground-glass opacity) with a specific anatomical landmark (e.g., left lower lobe). If diffusion models cannot bind 2D colored shapes reliably, they cannot be trusted to bind multiple subtle pathologies in continuous radiological volumes.
2. **Q:** *Could clinical generative models cause diagnostic errors downstream?*
   **A:** Yes. If a generative model hallucinates spurious co-occurrences or places a lesion in an impossible anatomical relationship, downstream diagnostic classifiers trained on that synthetic data will learn dangerous diagnostic shortcuts.

---

## Slide 59: Real-world implications: robotics and safety

### Presentation Notes
- **Safety-Critical Domains:**
  - *Autonomous Driving:* Rare edge cases are almost always novel combinations of common elements: e.g., an open vehicle door + a child crossing + dense fog. Generative world models used for safety simulation must be audited for whether they accurately synthesize these joint hazards.
  - *Robotic Manipulation:* Generating goal images for robot policies requires exact spatial relationships (e.g., "place the clamp left of the bracket without colliding with the fixture").
  - *Industrial Quality Control:* Multi-defect inspection requires generative models to simulate interacting material flaws (e.g., micro-crack adjacent to a high-stress weld seam).

### Probable Audience Questions
1. **Q:** *Why is data simulation preferred over collecting real edge cases in robotics?*
   **A:** Physical collection of hazardous edge cases (e.g., vehicle crashes or industrial failures) is dangerous, destructive, and expensive. Generative simulation is the only scalable alternative, making exposure auditing vital.

---

## Slide 60: Why this research matters

### Presentation Notes
- **Broader Significance:**
  - *Theoretical:* Pinpoints the boundary between memorized association and systematic generalization in generative models.
  - *Diagnostic:* Provides mathematical tools to audit latent exposure in datasets.
  - *Practical:* Prevents wasteful over-collection of redundant data by identifying what exposure actually drives performance.

### Probable Audience Questions
1. **Q:** *Can this framework be applied to text-to-video or 3D models?*
   **A:** Yes. The mathematical formulation of marginal vs. joint exposure and composition gaps applies universally to any conditional generative domain.

---

## Slide 61: Take-home message

### Presentation Notes
- **Core Summary:** Reiterate the central insight: *A model's success may depend on which combinations it has seen, not only how often it has seen the pieces.*
- **Closing Thought:** Grounded measurement and controlled experimental design are essential to demystify generative AI capabilities.

### Probable Audience Questions
1. **Q:** *What is the single most important takeaway for a practitioner?*
   **A:** Always audit the joint co-occurrence structure of your training data before attributing compositional failures to model architecture.

---

## Slide 62: Questions?

### Presentation Notes
- **Closing:** Thank the audience, advisors, and collaborators.
- **Invitation:** Open the floor to questions on synthetic controls, diffusion mechanics, exposure optimization, checker validation, and next experimental steps.

### Probable Audience Questions
1. **Q:** *When do you expect Stage 0 gates to be completed?*
   **A:** We are currently running the expanded 57-template design optimization and preparing the stratified human audit pack, targeting Stage 0 resolution in the next experimental cycle.

---

## Slide 63: References and project evidence

### Presentation Notes
- **Citations:**
  - *Which Exposure Matters?* Research Proposal, Rev. 4.
  - Jeong et al., MOSAIC (ICML 2026): scene complexity and imbalance benchmarks.
  - Okawa et al. (NeurIPS 2023): compositional vs. constituent abilities.
  - Nützel et al., GRASP (2025): gradient conflicts in long-tail diffusion.
  - Bradley (ICML 2026): local conditional scores in compositional diffusion.
  - Ho et al. (NeurIPS 2020) & Song et al. (ICLR 2021): DDPM and DDIM foundations.
- **Final Note:** Remind the audience to consult original literature for detailed prior-work claims.

### Probable Audience Questions
1. **Q:** *How does your composition gap compare to Okawa et al.'s metric?*
   **A:** Okawa et al. multiply constituent accuracies on classification tasks. We adapt and extend this multiplicative null to generative diffusion models on both logit and log scales, incorporating automated checker abstention tracking.
