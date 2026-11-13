# Presentation: Which Exposure Matters?

This directory contains the Beamer slide deck and accompanying presenter materials for the research project **"Which Exposure Matters? How training examples shape image-generation skills"** (Proposal and exploratory work).

## Directory Contents

- **`main.pdf` (Compiled Presentation PDF)**:
  A 63-slide, 16:9 widescreen Beamer presentation document. It provides a comprehensive, rigorous overview covering:
  - Table of Contents and presentation roadmap
  - Adjacent research (MOSAIC, Okawa et al., GRASP, Bradley) and project novelty
  - Core research question and compositional binding motivation
  - In-depth diffusion model mathematics (Gaussian forward process, closed-form sampling, reverse transitions, MSE noise prediction)
  - Detailed model architecture (hierarchical U-Net with GroupNorm, multi-head cross-attention at the $16 \times 16$ bottleneck, token embeddings, classifier-free guidance)
  - Optimization dynamics (AdamW weight decay, Exponential Moving Average shadow weights, and DDPM vs. DDIM sampling schedules)
  - Synthetic scene environment, vocabulary, spatial relations, distinct object semantics, and exposure levels
  - Formal metrics (empirical exposure, log lift, multiplicative composition reference, logit/log composition gaps, N50 sample efficiency)
  - Dataset mixture optimization (formulating exposure design as constrained convex optimization $\min D_{\text{KL}}(w \| u)$ via SLSQP)
  - Automated pixel checkers, false accept/reject/abstention dynamics, and blinded human review protocols
  - Current obstacles (checker boundary sensitivity, relational baseline saturation, and design multicollinearity)
  - Stage 0 validation gates, future workflow, and theoretical implications
  - Real-world implications and safety-critical applications (medical imaging, rare pathology co-morbidity synthesis, and robotics / autonomous vehicle edge cases)

- **`main.tex` (LaTeX Source)**:
  The LaTeX source file formatted for Beamer (`aspectratio=169`, 11pt) using the default LaTeX font theme (`\usefonttheme{serif}`, Computer Modern Roman and Math). Includes self-contained vector TikZ schematics for the U-Net architecture, cross-attention dataflow, medical scan binding, diffusion processes, and optimization pipelines. Compiles cleanly with [Tectonic](https://tectonic-typesetting.github.io/).

- **`detailed-speaker-notes.md` (Detailed Notes & Q&A Defense Guide)**:
  A draft 63-slide guide with technical narration and possible audience questions. It has had a quick review; check technical claims and citations before use.

- **`speaking-guide.md` (Quick Speaking Guide)**:
  A quick-reference, frame-by-frame (1–63) bullet-point summary designed for handy reference during live presentation delivery.

- **`assets/` & `generate_noise.py` (Exploratory Image Assets)**:
  Image figures referenced in `main.tex`:
  - `assets/synthetic-example.png`: An example synthetic scene rendered from the project toy world demonstrating shapes, colors, and spatial arrangements.
  - `assets/noise-0.png` through `assets/noise-3.png`: Progression demonstrating forward diffusion corruption and reverse denoising steps.
  - `generate_noise.py`: A Python script using NumPy and Pillow that generates `noise-0.png` through `noise-3.png` deterministically from `synthetic-example.png` (random seed 801).
  - `assets/pilot-red.png` & `assets/pilot-relation.png`: Actual exploratory samples produced by the preliminary 800,000-update prototype model checkpoint (prompted for "red" and "blue above circle"), illustrating current model capabilities and visual fidelity.

## Building the Presentation

The presentation is built using `tectonic`. A `Makefile` is provided in this directory with a default build target.

From inside the `presentation/` directory:

```bash
# Build main.pdf (default target)
make

# Or run tectonic directly
tectonic main.tex
```

### Additional Makefile Targets

```bash
# Regenerate the synthetic noise image assets
make assets

# Remove compiled PDF
make clean

# Display available targets
make help
```
