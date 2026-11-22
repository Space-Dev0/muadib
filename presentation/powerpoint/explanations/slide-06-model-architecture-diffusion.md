# Slide 6 — Model architecture and diffusion

This guide explains the sixth slide from the input image through the U-Net, the forward-noising equation, velocity prediction, conditioning and reverse sampling. The final section is a short script for the audience. The architecture details are checked against the project's `exposure/pilot_train.py` and the reported pilot configuration.

## 1. What is this slide showing?

Our pilot model is a compact **conditional U-Net** trained as a diffusion denoiser. It receives:

1. A noisy 64 × 64 RGB image `x_t`.
2. A diffusion timestep `t` telling it which noise level it is processing.
3. A condition telling it what target to generate; in the four-cell pilot this is a requested color.

It outputs a tensor of the same image dimensions that predicts a **velocity target**. During training the prediction is compared with a target derived from a clean training image and sampled noise. During generation the network's prediction helps a sampler repeatedly transform noise into an image.

The diagram on the slide is schematic. It summarizes the spatial resolutions and information flow; it is not a complete layer-by-layer computational graph.

## 2. Why a diffusion model?

A diffusion model learns a sequence of denoising-related transformations rather than producing a complete image in one ordinary forward pass. During training, we can manufacture noisy inputs by combining known clean images with known random noise. That gives us a target for learning.

At sampling time, a new random noise image is progressively transformed by a reverse process. Because the model is conditional, the requested color is provided as it predicts at each step. Generation is not guaranteed to satisfy the condition: that must be evaluated separately.

The general DDPM approach is described by [Ho, Jain and Abbeel (2020)](https://arxiv.org/abs/2006.11239). Our velocity parameterization is associated with work such as [Salimans and Ho (2022)](https://arxiv.org/abs/2202.00512).

## 3. What is a U-Net?

A U-Net is an encoder–decoder neural network with connections between corresponding resolution levels. The usual picture resembles a “U”:

```text
64 × 64 input --> encoder --> 32 × 32 --> encoder --> 16 × 16 bottleneck
       |                        |                         |
       +---- skip connection ---|--------------> decoder |
                                +---- skip ----> decoder |
                                       64 × 64 output <--+
```

This sketch is conceptual, not the exact tensor graph.

- The **encoder** converts pixels into feature maps and reduces spatial resolution.
- The **bottleneck** processes a more compact spatial representation and, in this model, incorporates condition information through cross-attention.
- The **decoder** increases spatial resolution toward the output size.
- **Skip connections** carry features from encoder stages to matching decoder stages so that fine spatial information can be used again.

A **feature map** is an array of learned intermediate values, not necessarily a directly interpretable image. Channels provide multiple learned features at each spatial location.

## 4. Read “width 16” and the 64 → 32 → 16 path

**Base width 16** refers to the number of channels at the first feature level in the reported pilot, not a 16 × 16 input image. The input image is still 64 × 64 pixels and has three RGB channels.

In the current implementation:

| Stage | Spatial resolution | Feature channels when width = 16 |
|---|---:|---:|
| Input image | 64 × 64 | 3 RGB channels |
| First encoder level | 64 × 64 | 16 |
| Second encoder level | 32 × 32 | 32 |
| Bottleneck | 16 × 16 | 64 |
| Final output | 64 × 64 | 3 predicted values per pixel |

The two downsampling operations are stride-2 convolutions. The decoder upsamples to the corresponding skip-connection sizes, concatenates the saved encoder features, and applies residual blocks. The output has three channels because it predicts an image-shaped velocity tensor; the three numbers at a pixel should not be read directly as a finished RGB image.

The width is an architecture setting. It is distinct from the number of objects, the batch size, and the 1,000 diffusion timesteps.

## 5. What do the residual blocks do?

A residual block computes a learned transformation and adds a skip or shortcut path from its input. A simple conceptual form is

\[
 y=\operatorname{skip}(x)+F(x,\operatorname{emb}(t)).
\]

The current `Residual` blocks use normalization, convolutions, a nonlinear activation, and a projected timestep embedding. The shortcut is an identity when channel counts match, or a 1 × 1 convolution when they do not. The equation is a simplified description; the exact order of operations is specified by the code.

Residual connections within blocks and U-Net skip connections across encoder and decoder are related ideas but occur at different places. They do not prove that every spatial detail is retained perfectly.

## 6. What is the timestep embedding?

`t` identifies the noise level in the diffusion schedule. A network asked to process a nearly clean image should not necessarily behave identically to one asked to process an almost completely noisy image.

The current model uses a learned timestep embedding. The embedding is projected and added within residual blocks. It supplies information about the current diffusion step; it is not an optimizer-update counter, not an object position and not a generation seed.

For the reported pilot, the schedule contains 1,000 diffusion timesteps. During training, the model sees sampled timesteps rather than performing a full 1,000-step generation for every optimizer update.

## 7. What is the condition and how does cross-attention use it?

A condition specifies what kind of output is requested. In the reported four-cell pilot, the model is asked for a target color. The general project architecture can encode structured predicate tokens, but the particular pilot condition is a single requested-color predicate; it is not a full natural-language sentence or a multi-object rule prompt.

The model turns condition-token fields into learned vector embeddings. At the 16 × 16 bottleneck, the spatial features interact with the condition representation through a **four-head cross-attention** module:

- The bottleneck image features supply the **queries**.
- The condition embeddings supply the **keys** and **values**.
- Four attention heads provide parallel learned projections.
- The resulting conditioned features are added back into the bottleneck representation.

Cross-attention is a mechanism for incorporating condition information, not proof that the model interprets a request correctly. With a single valid condition token, there are no multiple valid token choices to distinguish with a softmax; the model is still conditioned through the projected value and surrounding network. Do not claim that the four-cell pilot learns attention over a full multi-word prompt.

During training, the implementation can replace a condition with a learned null token when condition dropout is used. The reported dropout probability is 0.1. A null token also prevents the attention operation from receiving an entirely masked context.

## 8. Forward noising: explain every symbol

The slide shows

\[
 x_t=\sqrt{\bar\alpha_t}\,x_0
     +\sqrt{1-\bar\alpha_t}\,\epsilon,
 \qquad \epsilon\sim\mathcal N(0,I).
\]

- $x_0$: a clean training image, represented as an image-shaped numerical tensor.
- $\epsilon$: a sample of standard Gaussian noise with the same shape.
- `t`: the chosen diffusion timestep.
- $\bar\alpha_t$: the cumulative signal coefficient defined by the noise schedule.
- `x_t`: the clean image and noise combined at that timestep.

If $\bar\alpha_t$ is close to 1, the clean-image term is stronger; if it is close to 0, the noise term is stronger. For a purely illustrative example, if $\bar\alpha_t = 0.64$, the coefficients are `sqrt(0.64) = 0.8` and `sqrt(0.36) = 0.6`, so $x_t = 0.8x_0 + 0.6\epsilon$.

The implementation defines a linear beta schedule from `0.0001` to `0.02` over the configured timesteps and takes the cumulative product of $1-\beta_t$ to obtain $\bar\alpha_t$. The illustration is just arithmetic; it is not a claim about a specific pilot timestep.

**Forward noising is not the generation process.** It is the known corruption process used to construct training examples. Generation uses a learned reverse sampler.

## 9. Velocity target: what is the network learning?

The second displayed equation is

\[
 v_t=\sqrt{\bar\alpha_t}\,\epsilon
     -\sqrt{1-\bar\alpha_t}\,x_0.
\]

This **velocity** is a particular combination of the known clean image and the sampled noise. It is not the physical speed of an object moving across the picture.

The model predicts `v_t` from the noisy image, timestep and condition. Its training objective is the mean-squared error between the prediction and the target:

\[
 \mathcal L(\theta)
 =\mathbb E\left[\left\|v_\theta(x_t,t,\tilde c)-v_t\right\|_2^2\right].
\]

$\theta$ represents model parameters; $\tilde c$ is the provided condition or the null condition after condition dropout. In practice the expectation is approximated using a minibatch. The AdamW optimizer updates the model to reduce this loss.

The symbol `v_t` here means a diffusion target. It is different from the second-moment variable often called `v` in AdamW. A small training loss does not automatically establish that a sampled image contains the requested object or satisfies a relational rule.

## 10. How does the sampler use a velocity prediction?

The sampler can convert the velocity output into an estimated noise tensor:

\[
 \widehat\epsilon_t
 =\sqrt{\bar\alpha_t}\,v_\theta(x_t,t,c)
 +\sqrt{1-\bar\alpha_t}\,x_t.
\]

When the predicted velocity equals the exact target, substitution shows that this expression returns the sampled noise. Similarly, the implied clean-image estimate is

\[
 \widehat x_0
 =\sqrt{\bar\alpha_t}\,x_t
 -\sqrt{1-\bar\alpha_t}\,v_\theta(x_t,t,c).
\]

These equations describe parameterization conversion. The actual reverse step also uses the diffusion schedule and sampler; conversion alone is not the entire sampling algorithm.

The reported pilot samples with an **exponential moving average** (EMA) of the trained weights, 1,000 reverse steps and guidance scale 1. EMA is a smoothed copy of parameters maintained during training, not an average over generated images. Under the code's guidance convention, scale 1 selects the conditional prediction; it does not amplify it beyond that conditional prediction. Slide 8 explains EMA and optimizer updates in more detail.

## 11. From training to a generated image

1. Take a clean annotated training image and a target-color condition.
2. Sample a timestep and random Gaussian noise.
3. Construct `x_t` and the corresponding velocity target using the known equations.
4. Predict velocity with PilotUNet and update its parameters using AdamW.
5. Maintain EMA parameters as training proceeds.
6. For sampling, draw starting noise and use the EMA model through the reverse schedule, conditioned on the requested color.
7. Evaluate the finished generated image using the defined human review or checker protocol.

A **sampling noise seed** controls generation randomness. It is not a separate trained model, and changing it is not equivalent to running an independent training replicate.

## 12. Likely evaluator questions

**Why is it called a U-Net?** Its encoder reduces resolution and its decoder increases resolution, with skip connections between matching stages, giving a U-shaped conceptual layout.

**Why give the model `t`?** The amount of noise changes with the diffusion step. The timestep embedding tells the network which denoising problem it is solving.

**What does attention attend to?** Bottleneck image features query the condition-token embeddings. In this pilot the condition is a requested color, not a long text prompt.

**Is `v_t` an object's speed?** No. It is a mathematical target constructed from the clean image and sampled Gaussian noise.

**Does the output directly equal the finished image?** No. The network predicts a velocity tensor at the current step; the reverse sampler uses its prediction over many steps to produce a final image.

**Does this prove the model understands complex relations?** No. The slide describes architecture and training mechanics. The four-cell pilot condition is a color request, and the proposed multi-predicate exposure experiment is separate.

## What to tell the audience

> “Our model is a compact conditional U-Net. It takes a noisy 64-by-64 image, the noise-level timestep and a requested color. The encoder reduces the image to 32-by-32 and then 16-by-16 features; cross-attention injects the condition, and the decoder uses skip connections to return to image resolution. During training we construct noisy images from clean scenes and train the network to predict a velocity target. During generation the sampler uses the trained EMA model repeatedly to turn noise into an image.”
