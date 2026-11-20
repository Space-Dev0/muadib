# Presenter notes: Which Exposure Matters?

These notes follow the 12-slide PowerPoint. Treat quoted passages as a guide rather than a script. The joint-exposure hypothesis remains untested and Stage 0 has not passed.

## Slide 1 — Introduction

“Good morning. Our project is called *Which Exposure Matters?* We are studying how the composition of a model’s training data might affect its ability to generate images that combine familiar concepts.”

Give a simple example: a model might have seen many red objects and many squares, but relatively few red squares. Would seeing more red squares help it draw a red square? The question becomes harder when a request also contains a relation, such as a red object to the left of a square. Introduce the team and explain that the project connects digital image processing, computational optimization and computer networks.

**Transition:** Before discussing the experiment, explain what a diffusion model does.

## Slide 2 — How diffusion generates an image

“A diffusion model learns through a corruption-and-recovery process. During training, we add Gaussian noise to a clean image. The model sees the noisy image, a timestep and a condition, and learns a denoising target. At generation time, it begins with noise and repeatedly applies the learned model.”

The first equation is

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon`.

Here `x_0` is the clean image, `epsilon` is standard Gaussian noise, and `alpha_bar_t` comes from a noise schedule. At a small timestep the noisy image retains more signal; later in the schedule it contains more noise. The images shown on the slide illustrate forward corruption in reverse order; they are not snapshots of a generated sample.

Our pilot uses the velocity target

`v_t = sqrt(alpha_bar_t) epsilon - sqrt(1 - alpha_bar_t) x_0`.

The U-Net learns to predict this target from the noisy image, timestep and condition. This is one parameterization of the denoising task. During sampling its prediction can be converted into the quantities needed for reverse steps. Cite Ho et al. (2020) for DDPM and Salimans and Ho (2022) for velocity prediction.

**Transition:** We use this model to study whether knowing parts guarantees a correct combination.

## Slide 3 — Research question: parts and combinations

“Suppose red objects appear often and squares appear often. That does not tell us how frequently the same object is both red and square.” Explain that *marginal exposure* counts an individual concept and *joint exposure* counts the combination. The diagram illustrates that the joint frequency can change while the margins remain fixed.

Our proposed question is whether the model’s success changes with joint training exposure after accounting for the individual concepts. Emphasize that this is the hypothesis and planned intervention, not a result obtained by the pilot. For relations, “left of” compares horizontal center coordinates, and the two object variables refer to distinct objects.

## Slide 4 — Related research

DDPM by Ho et al. establishes the foundational denoising framework. Liu et al. study combining concepts using composable diffusion models. MOSAIC by Jeong et al. studies multi-object generation in a controlled synthetic setting.

Okawa et al. (NeurIPS 2023) study compositional generalization of conditional diffusion models on synthetic data. They investigate how learning constituent capabilities relates to generating new combinations. This is close prior work: our proposed distinction is to construct training mixtures that vary joint exposure while controlling marginal exposure. Be clear that this is a proposed extension, and that our current study has not yet demonstrated the intervention or established a causal effect.

Reference: Maya Okawa, Ekdeep Singh Lubana, Robert Dick and Hidenori Tanaka, *Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task*, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html

## Slide 5 — A controlled synthetic world

“Our renderer creates small images of colored shapes and stores the properties and positions of the objects.” Because scene metadata is known, a program can evaluate whether a particular rule is satisfied in a rendered scene. This supports controlled measurements that would be much harder with a large web-image collection.

The 57-condition exposure matrix uses 5.7 million accepted scenes to estimate how often predicates and rules occur. This measurement dataset is separate from the four-condition diffusion pilot. Each pilot condition used 200 fixed training scenes. The picture is a clean rendered example rather than an image produced by the diffusion model.

Acknowledge that synthetic scenes make control easier but do not establish performance on natural photographs. Earlier relation predicates also proved near-tautological, and revised designs require validation.

## Slide 6 — Exposure and experimental design

Let `m_p` denote how frequently predicate `p` appears under a training mixture, and `n_r` denote how frequently the entire rule `r` appears. A proposed lift is `log(n_r) - sum_p log(m_p)`. This expresses joint exposure relative to a product of marginal exposures; it should not automatically be interpreted as a causal effect.

We generate candidate scene distributions and adjust their mixture weights. The objective is to vary exposure while controlling nuisance variables. Optimization matters because marginal constraints, rule complexity and co-occurrence can be correlated. A mathematically feasible mixture must also have useful variation and hold up on independently measured scenes.

The existing design has not passed its identifiability and validation gates. Exact fixed marginals can make a marginal-exposure coefficient unidentifiable alongside unrestricted rule intercepts. Explain that finding and diagnosing this problem is part of the work; do not call it a solved experimental design.

## Slide 7 — Pilot model and experimental setup

The pilot asks a narrower question than the main project: how do object count and target-color balance affect a model’s ability to produce a requested-color circle? It crosses two object counts (two and three) with two training color ratios (50:50 and 90:10). Each of the four models used 200 fixed images and trained for 500,000 updates.

At evaluation, eight noise seeds and two target colors produce 16 images per condition. Reusing sampling seeds makes comparisons easier, but these seeds are not independent model-training replicates. The reported 64-image block has one model-training seed per condition. The fixed, small datasets can encourage memorization, and target role is confounded with attributes. Therefore do not present this pilot as a direct test of the joint-exposure hypothesis.

## Slide 8 — Pilot results: 64 reviewed samples

Walk the audience through the counts: low/balanced has 16 yes, 0 no and 0 unresolved; low/imbalanced has 15, 0 and 1; high/balanced has 14, 2 and 0; high/imbalanced has 14, 1 and 1. The total is 59 yes, 3 no and 2 unresolved.

A complexity contrast compares the average of the two low-complexity success rates with the average of the two high-complexity rates. An imbalance contrast compares balanced with imbalanced cells. Assigning the two unresolved labels in all possible ways gives the displayed ranges. These are finite-image sensitivity bounds, not statistical confidence intervals.

The four returned review files for this original block were byte-identical, so those files alone do not establish independent reviewer authorship. The small number of images, one model-seed block and fixed training datasets prevent a broad conclusion about which factor generally matters more.

## Slide 9 — Qualitative samples

Point to each image and explain the question the reviewers were answering: “Is a circle of the requested color present?” Sample 0003 received a yes; sample 0001 was uncertain. These are selected examples for explanation and should not be presented as a random or representative gallery.

A diffusion model samples from a distribution. There is no single unique reference image that every successful output must match. A useful evaluation therefore checks whether the requested properties are present, while also recording unclear or unrecognizable outputs.

## Slide 10 — Checker and human audit

The checker attempts to turn an image into object descriptions and rule decisions. On the 64-image block the experimental checker answered 46 and abstained on 18. Its answered cases agreed with the recorded labels, but it abstained on all three human-negative examples and both unresolved examples.

This is selected agreement: the checker’s accuracy on cases where it chose to answer does not tell us its false-accept rate on difficult failures. A reliable evaluation requires deliberately sampled positive and negative examples, reporting abstentions, and a separate validation set. Local vision-model comparisons have been exploratory and do not by themselves establish a validated relation checker.

## Slide 11 — Computer networking prototype

Explain the problem: a scene renderer and a model trainer may run on different machines. Our prototype sends scenes and JSON annotations over TCP so the producer and consumer can operate separately. TCP supplies ordered reliable byte delivery, while the application must still handle framing, reconnects and validation.

A separate checkpoint-transfer prototype can resume an interrupted transfer and checks the received bytes against a SHA-256 hash. A hash checks integrity; it does not itself authenticate the sender or encrypt traffic. The experiment used secure tunneling where relevant.

The localhost throughput numbers come from short CPU streaming trials and include the tested application pipeline. They are not measurements of GPU utilization, maximum network capacity or a distributed training speedup. The two-device inference demonstration produced one image per device and is a functional demonstration, not a performance benchmark.

## Slide 12 — Current status and next steps

“We have built the controlled renderer, exposure-measurement tools, mixture-design diagnostics, diffusion pilot, review workflow and networking prototypes.” The main scientific question still needs an identifiable exposure intervention and sufficiently reliable outcome measurement.

State the next steps in order: validate checker performance on positive, negative and relational cases; establish a feasible design with the intended exposure variation and independent validation; then train independently replicated models on the selected mixtures and evaluate the hypothesis. Stage 0 has not passed. End by returning to the main question: do models learn combinations from knowing their parts, or does exposure to the combination itself matter?

## References for the presenters

- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020: https://arxiv.org/abs/2006.11239
- Salimans and Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*, 2022: https://arxiv.org/abs/2202.00512
- Liu et al., *Compositional Visual Generation with Composable Diffusion Models*, 2022: https://arxiv.org/abs/2206.01714
- Okawa et al., *Compositional Abilities Emerge Multiplicatively*, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html
- Jeong et al., *When Do Diffusion Models Learn to Generate Multiple Objects?*, 2026: https://proceedings.mlr.press/v306/jeong26b.html
