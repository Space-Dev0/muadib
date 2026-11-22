# Slide 8 — Experimental setup, diffusion training and EMA sampling

This guide explains the eighth slide of the 12-slide submission presentation. It connects the data, the denoising task, the displayed optimization equations, and the evaluation protocol. A short audience script appears at the end.

## 1. What this slide is about

This slide describes a **four-condition diffusion-model feasibility pilot**. It asks whether our small conditional U-Net can be trained and evaluated on simple scenes when we vary object count and the balance of target colors.

It is not the direct joint-versus-marginal exposure experiment from slide 4. The pilot uses a color request and a target circle; it does not train on the proposed controlled mixtures of full compositional rules.

## 2. Understand the 2 × 2 dataset design

We vary two factors:

1. **Scene complexity:** scenes have either two or three objects.
2. **Target-color balance:** target colors are either balanced 50:50 or imbalanced 90:10.

Crossing these factors creates four conditions:

| Condition | Objects per scene | Red/blue target balance | Fixed training scenes |
|---|---:|---:|---:|
| Low complexity, balanced | 2 | 50:50 | 200 |
| Low complexity, imbalanced | 2 | 90:10 | 200 |
| High complexity, balanced | 3 | 50:50 | 200 |
| High complexity, imbalanced | 3 | 90:10 | 200 |

Each condition has its own dataset and its own model in the reported block. An image is 64 × 64 pixels. The intended target is a red or blue circle. In these pilot data, target circles are red or blue and distractors have other colors and shapes, so target role and attributes are confounded. Both target combinations occur in training; this is not a held-out red-circle-versus-blue-circle composition test.

**Balanced 50:50** means the two requested target colors are equally represented in the training condition. **Imbalanced 90:10** means one target color is much more common. “Two objects versus three objects” changes the object count, but the four pilot datasets were drawn separately rather than constructed as exact matched-geometry pairs. Therefore a contrast cannot be interpreted as isolating object count from every possible layout difference.

## 3. Model and training budget

Each reported condition uses a conditional PilotUNet with base width 16. It takes a noisy image, a diffusion timestep, and a requested-color condition, then predicts a velocity target. The model uses timestep embeddings, cross-attention for condition information, and U-Net skip connections; slide 6 gives the architecture overview.

For the main block on this slide, each condition was trained for **500,000 optimizer updates** using batch size **4**. The learning rate reported in the paper is **0.0001**, and EMA decay is **0.999**. A training update is one optimizer step using a batch; it is not a complete pass over the dataset and is not a diffusion timestep.

A useful scale calculation is:

\[
500{,}000\;\text{updates}\times4\;\text{images/update}
=2{,}000{,}000\;\text{image presentations}.
\]

Relative to 200 fixed training images, this is 2,000,000 / 200 = **10,000 dataset-equivalent passes**. Because batches are sampled, that does not mean every image was necessarily selected exactly 10,000 times. The small fixed dataset makes memorization a real question; the update count alone does not establish generalization.

A second training-seed block on the same datasets is also documented in the paper. The 64-image numbers on this slide and the following results slide refer to the first fixed evaluation block, not a combination of all blocks.

## 4. What happens in one diffusion-training step?

It helps to distinguish two different indices:

- `k`: an **optimizer update**, counting how many times model parameters have been updated.
- `t`: a **diffusion timestep**, describing a noise level in the 1,000-step diffusion schedule. A timestep is sampled for training examples.

One training step conceptually proceeds as follows:

1. Draw a batch of four clean training images `x₀` and their color conditions.
2. Draw diffusion timesteps and Gaussian noise `ε`.
3. Use the forward-noising equation to construct each noisy image `x_t`.
4. Compute the correct velocity target from the clean image and sampled noise.
5. Feed `x_t`, `t` and the condition into PilotUNet to predict that target.
6. Calculate a mean-squared error loss between the prediction and target.
7. Use backpropagation and AdamW to update the network's parameters.
8. Update the EMA copy of the parameters.

Training does not repeatedly run the complete 1,000-step reverse sampler for every optimizer update. It learns from examples at sampled noise levels; the full reverse procedure is used when generating images.

## 5. The forward-noising and velocity equations

Slide 6 shows the equations underlying the training statement on slide 8:

\[
 x_t=\sqrt{\bar\alpha_t}\,x_0
     +\sqrt{1-\bar\alpha_t}\,\epsilon,
 \qquad \epsilon\sim\mathcal N(0,I).
\]

- `x₀` is the original clean training image.
- `ε` is sampled Gaussian noise, with the same dimensions as the image.
- `t` indexes the noise level.
- `ᾱ_t` is determined by the chosen noise schedule.
- `x_t` is the corresponding noisy image.

At a low noise level, `x_t` retains more image information. At a high noise level, it contains a larger noise contribution. The model is trained at many `t` values so it can support generation across the schedule.

The pilot uses **velocity prediction**. Its target is

\[
 v_t=\sqrt{\bar\alpha_t}\,\epsilon
     -\sqrt{1-\bar\alpha_t}\,x_0.
\]

The neural network predicts `v_t` from `x_t`, `t`, and the condition. The loss is the mean-squared difference between predicted and true velocity:

\[
 \mathcal L(\theta)
 =\mathbb E\!\left[\|v_\theta(x_t,t,\tilde c)-v_t\|_2^2\right].
\]

`θ` denotes the network parameters. `c̃` is the requested condition or a null condition when condition dropout is applied. The paper reports dropout probability 0.1. In a minibatch, the expectation is approximated with the sampled images, timesteps and noise.

**Why use a velocity target?** It is a valid denoising parameterization combining clean image and noise information. It can be converted algebraically to a noise prediction for sampling. A low velocity loss is useful but does not by itself prove that a generated image satisfies the requested rule.

## 6. Read the first optimization line on slide 8

The slide displays the gradient and an ordinary gradient-descent update:

\[
 g_k=\nabla_\theta\mathcal L_k(\theta_k),
 \qquad
 \theta_{k+1}^{\mathrm{SGD}}=\theta_k-\eta g_k.
\]

- `g_k` is the gradient: the direction in parameter space in which the current loss increases most rapidly locally.
- `∇_θ` means differentiate with respect to the model parameters.
- `η` is the learning rate.
- Subtracting `ηg_k` moves parameters in a direction intended to reduce the current minibatch loss.

The slide uses `t` as the update index in its SGD/AdamW lines; this guide calls it `k` to avoid confusing it with diffusion timestep `t`. The displayed SGD expression is a basic comparison for understanding optimization. The reported pilot was trained with **AdamW**, not ordinary SGD.

## 7. Read the AdamW line

A common notation for the displayed AdamW update is

\[
 \theta_{k+1}=(1-\eta\lambda)\theta_k
 -\eta\frac{\widehat m_k}{\sqrt{\widehat v_k}+\epsilon_{\rm opt}},
 \qquad
 \widehat m_k=\frac{m_k}{1-\beta_1^k},\quad
 \widehat v_k=\frac{v_k}{1-\beta_2^k}.
\]

The terms have distinct jobs:

- `m_k` is a moving average of past gradients: a momentum-like first moment.
- `v_k` is a moving average of squared gradients: a second moment. It is **not** the diffusion velocity target `v_t`; the same letter is used for different quantities in these two formulas.
- The hats denote bias-corrected moment estimates.
- `β₁` and `β₂` govern how quickly the two moving averages change.
- `ε_opt` is a small numerical stabilizer in the denominator. It is **not** the Gaussian image noise `ε` in the diffusion equation.
- `λ` here is the AdamW weight-decay coefficient. It is **not** joint-exposure lift from slide 3.
- `(1 − ηλ)θ_k` expresses decoupled weight decay in this form of the update.

AdamW adapts the scale of parameter updates using the gradient histories. Weight decay is applied separately from the adaptive gradient term. It is an optimizer choice; it does not guarantee generalization or determine whether the generated image follows the condition.

## 8. Explain the EMA update on the slide

The third line displays the exponential moving average (EMA):

\[
 \bar\theta_{k+1}=\rho\bar\theta_k+(1-\rho)\theta_{k+1},
 \qquad \rho=0.999.
\]

`θ_{k+1}` is the newly updated training model. `θ̄_k` is a separate running average of previous model weights. The EMA update retains 99.9% of the previous average and adds 0.1% of the new weights at each update.

For a simple numerical example, if one EMA weight was 2.0 and the new training weight was 3.0, the next EMA value would be `0.999 × 2.0 + 0.001 × 3.0 = 2.001`. In reality this happens for the model parameters, not just one number.

The goal is to provide a smoothed parameter trajectory for evaluation. EMA is **not** a second independently trained model, not a training dataset average, and not an average of final generated images. It can be helpful in practice, but the presence of EMA alone does not establish an improvement without an appropriate comparison.

## 9. Explain how the EMA model is used during sampling

After training, the sampler starts with a noise image and applies reverse diffusion steps using the EMA-weighted denoiser. The slide shows the velocity-to-noise conversion:

\[
 \widehat\epsilon_t
 =\sqrt{\bar\alpha_t}\,v_{\bar\theta}(x_t,t,c)
 +\sqrt{1-\bar\alpha_t}\,x_t.
\]

The denoiser `v_{θ̄}` uses the EMA weights. It predicts velocity at the current noisy image and diffusion timestep. The formula converts that prediction into a noise estimate for the sampler's reverse update. If the velocity prediction equals the exact target, the conversion recovers the sampled noise algebraically.

The reported protocol uses **1,000 reverse steps**, guidance scale **1**, and EMA weights. Under the code's guidance convention, scale 1 selects the conditional prediction; it does not amplify the conditional direction above that prediction. Eight fixed noise seeds are used for each of two requested colors, giving **16 generated images per cell** and **64 images across the four cells**.

A noise seed controls the starting randomness of generation. A training seed controls aspects of model initialization and training randomness. Many generated images from one trained model are **not** many independently trained model replicates.

## 10. What the pilot does and does not establish

The pilot demonstrates a concrete training and generation workflow and produces the images analyzed on slides 9 and 10. It gives preliminary descriptive evidence about the four pilot conditions.

It does not establish the effect of controlled joint exposure: the required exposure mixtures were not the intervention here. The fixed small datasets, confounded target attributes, one training-seed block for the 64-image table, separately generated layouts, and limited evaluation also restrict what can be inferred. The later seed block shares the data and does not by itself provide independent dataset replication.

## 11. Questions you may be asked

**Is 500,000 the number of diffusion steps?** It is the number of optimizer updates per model. The diffusion schedule has 1,000 timesteps, and generation uses 1,000 reverse steps in this protocol.

**Why are there four models?** The experiment crosses two object-count levels and two color-balance levels. One separate model is trained for each cell of the reported block.

**What is the difference between AdamW and EMA?** AdamW changes the trainable weights to minimize loss. EMA tracks a smoothed copy of those changing weights, which is then used for sampling.

**Is 64 a dataset size?** Here 64 is the number of generated evaluation images: four cells × 16 images. Each training condition has 200 fixed training scenes.

**Does 59/64 prove high compositional generalization?** No. That count concerns the requested-color circle in this small, confounded pilot, not a held-out multi-predicate exposure intervention.

## What to tell the audience

> “We trained four conditional diffusion models, crossing two versus three objects with balanced versus imbalanced target colors. Each model used 200 fixed scenes, batches of four, and 500,000 AdamW updates. At each update we add noise at a sampled diffusion timestep and train the U-Net to predict a velocity target. We maintain an exponential moving average of its weights and use that EMA model for the 1,000-step reverse sampling process. Eight paired noise seeds and two colors give 16 samples per condition. This is a model-feasibility pilot; the direct joint-exposure intervention is a separate next step.”
