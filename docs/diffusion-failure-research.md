# Research on merged and blurry diffusion samples

These papers suggest testable explanations for our previews. None establishes
the cause of our failure or proves that a proposed intervention will work.

- [Lin et al., Common Diffusion Noise Schedules and Sample Steps Are Flawed](https://arxiv.org/abs/2305.08891)
  identify a mismatch when the noisiest training sample retains signal but
  inference starts from pure Gaussian noise. Our multiscale probe ends at
  alpha=0.01 and starts from Gaussian noise; this is a real mismatch, but its
  contribution to the observed shapes has not been measured. A valid test
  trains and samples consistently under each schedule rather than silently
  changing only inference on the existing checkpoint.
- [Compositional Abilities Emerge Multiplicatively](https://arxiv.org/abs/2310.09336)
  studies synthetic conditional generation and finds that rare concept
  combinations can require more optimization steps. This motivates matched
  learning curves and per-condition measurements; it does not imply any
  particular update count will solve our task.
- [Favero et al., How Compositional Generalization and Creativity Improve as
  Diffusion Models are Trained](https://proceedings.mlr.press/v267/favero25a.html)
  report increasing coherence with training time or dataset size in studied
  settings. Our 20k, 50k and 80k exploratory previews are not a controlled
  estimate of such a curve because optimizer states and seeds differ.
- [Jeong et al., MOSAIC](https://proceedings.mlr.press/v306/jeong26b.html)
  investigate scene complexity, imbalance and counting in a different,
  controlled benchmark. Our two-dimensional toy experiment is an adaptation,
  not their benchmark or evidence that their conclusions hold here.

In our own matched sampling comparison, 50, 200 and 500 denoising steps gave
visually similar blotchy examples. That narrows the step-count hypothesis for
those four examples; it does not rule out a different sampler or schedule.
Our proposed next diagnostic is a bounded stronger-denoiser run on the same
dataset, with model, schedule, training seed and samples recorded. Different
architecture and training budgets cannot isolate a causal factor by themselves.
