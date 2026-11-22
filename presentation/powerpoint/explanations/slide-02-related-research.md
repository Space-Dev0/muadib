# Slide 2 — Related research

This guide explains the five research entries on slide 2, why they matter to our project, and how to discuss our proposed contribution accurately. The final section is a short script for the audience. Paper links are included so you can inspect the original sources.

## 1. Why do we need a related-research slide?

A research project does not start in isolation. The literature review tells the audience which ideas already exist, how they connect to our work, and what question we propose to investigate. It also prevents us from claiming that our team invented diffusion models, synthetic composition tests, or the idea that training frequency can matter.

The slide moves from a general image-generation method to increasingly specific composition studies:

1. **DDPM:** how a diffusion model learns to generate images.
2. **Composable diffusion:** how generation can combine conditions at inference time.
3. **MOSAIC:** how to investigate multiple objects under controlled data conditions.
4. **Okawa et al.:** how compositional ability relates to atomic capabilities and training data in a synthetic task.
5. **Liang et al.:** how diffusion models can learn factorized representations and composition in controlled experiments.

Our proposed question is related to these studies: can we vary exposure to complete combinations while accounting for exposure to the individual ingredients, and then measure the effect on generation? That planned intervention has not yet been completed.

## 2. First define the basic words

A **generative model** learns a distribution over data and can produce new samples. An image-generation model produces an image rather than a single class label.

A **diffusion model** learns to reverse a noise-corruption process. During training, a clean image is mixed with noise at a selected noise level. The model learns a target that supports denoising. During generation, the sampler begins with noise and repeatedly uses the learned model to construct an image. Slide 6 explains our particular implementation.

A model is **conditional** when it receives additional information specifying the desired output, such as a requested color, a class, or a text prompt. Our four-cell pilot uses a target-color condition; that is much narrower than free-form natural-language prompting.

**Compositionality** means combining components to satisfy a structured request. A model that can generate a red object and a circle separately has not automatically shown that it can generate a **red circle**, much less a red circle to the left of a blue square. Correct binding matters: a red square beside a blue circle does not satisfy “red circle.”

**Marginal exposure** is how frequently an individual event occurs in training. **Joint exposure** is how frequently the specified combination occurs. Slide 4 gives numerical tables showing that the two can vary separately.

## 3. Ho et al. (2020): Denoising Diffusion Probabilistic Models

**Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Jonathan Ho, Ajay Jain and Pieter Abbeel, NeurIPS 2020.

### What problem does it address?

How can we learn a probabilistic model that generates realistic samples? DDPM provides an influential framework in which a model learns to reverse a gradual noising process.

### The central idea

1. Begin with clean training examples.
2. Define a forward process that progressively adds noise.
3. Train a neural network on denoising-related predictions at different noise levels.
4. Generate by starting from noise and applying learned reverse steps.

The original paper connects the model's training objective to denoising and diffusion probabilistic modeling. The popular “learn to predict noise” explanation is a useful entry point, but diffusion models can use different mathematically related prediction targets.

### Why is it relevant to us?

Our pilot follows the general diffusion pattern: noise is added at a sampled timestep, a conditional U-Net predicts a target, and a reverse sampler generates images. **Our pilot uses velocity prediction**, so do not claim it uses only the original paper's noise-prediction parameterization. Velocity and noise estimates can be converted as shown on slides 6 and 8.

### What does this paper not prove for our study?

A capable denoising method does not by itself show that a model correctly binds attributes or that increased joint exposure causes a composition improvement. Those are separate experimental questions.

## 4. Liu et al. (2022): Composable diffusion models

**Paper:** [Compositional Visual Generation with Composable Diffusion Models](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6940_ECCV_2022_paper.php), Nan Liu and colleagues, ECCV 2022.

### What problem does it address?

A requested image can combine several concepts, objects or attributes. Liu and colleagues investigate structured composition of diffusion-model components to generate outputs satisfying combinations of conditions.

### What is the approach at a high level?

The paper formulates a way to combine diffusion-model information associated with different conditions at generation time. It studies compositional generation and illustrates how conditions can be composed, including attribute-binding examples. It is best described as an **inference-time composition** approach: it changes how available model information is combined during sampling.

### How is that different from our proposed question?

Our intended intervention concerns the **training-data distribution**: how often individual ingredients and complete combinations are encountered while learning. That is not the same as combining conditional predictions at inference. Both approaches address composition, and they could eventually be compared, but our current pilot has not demonstrated that changing training exposure outperforms inference-time composition.

The slide phrase “independently learned concepts” is a simplified explanation; the paper's technical formulation uses composable diffusion models. Do not imply that our team replicated all of its methods or results.

## 5. Jeong et al. (2026): MOSAIC and multiple objects

**Paper:** [When Do Diffusion Models Learn to Generate Multiple Objects?](https://proceedings.mlr.press/v306/jeong26b.html), Yujin Jeong and colleagues, ICML 2026.

### What problem does it address?

Diffusion systems can produce visually compelling images but may be unreliable when they need to generate several objects with the requested properties and relations. The paper investigates multi-object generation with controlled synthetic data and broader experiments.

### Why does a controlled multi-object setting matter?

When the dataset explicitly controls object count, attribute balance, combinations and relations, researchers can investigate which conditions make a generation problem harder. This makes the experimental question more precise than looking at a few unrelated example images.

### How does it relate to our pilot?

Our four-cell pilot adapts the broad idea of comparing **object count** and **target-color balance** in a controlled synthetic setting. It uses two or three objects and 50:50 or 90:10 target-color balance. Each pilot cell contains 200 fixed training images, and the reported first block uses one trained model per cell.

This is a much smaller adaptation, not a full replication of MOSAIC. Our target role is confounded with color and shape, and the four pilot layouts are not exactly matched across datasets. The first 64-image block cannot establish the wider paper's claims or prove our main joint-exposure hypothesis.

## 6. Okawa et al. (2023): compositional abilities and data frequency

**Paper:** [Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html), Maya Okawa and colleagues, NeurIPS 2023.

### What is the question?

How does a conditional diffusion model's ability to generate a new combination relate to its ability to handle the component concepts, and how does training data influence that process?

### What is the main idea?

The authors conduct a controlled synthetic study of compositional generalization. Their work connects the emergence of composite abilities with atomic capabilities and studies the effects of training-data attributes and frequency. Their reported behavior in that setting motivates thinking about combinations and component competence together.

**Out-of-distribution composition** here means a target combination is not represented as an ordinary training combination under the study's split. It does not necessarily mean every individual concept is entirely new.

### How does it inform our project?

A product of primitive success rates provides one proposed reference for our composition-gap diagnostic. If the full-rule satisfaction rate equals that reference, the stated logit difference is zero. This is a comparison, not a claim that every predicate outcome is independent or that the product is universally the correct baseline.

Our planned extension emphasizes **nonzero joint exposure**, mixture constraints and explicit measurement of complete-rule versus individual-event frequencies. This is closely related work; we should not claim that the general importance of data frequency or multiplicative structure is new to us. We also have not yet completed the direct mixture-intervention experiment.

## 7. Liang et al. (2024): factorization and composition

**Paper:** [How Diffusion Models Learn to Factorize and Compose](https://papers.nips.cc/paper_files/paper/2024/hash/1b60893887328a6a50dd00ae0d5ed51a-Abstract-Conference.html), Qiyao Liang, Ziming Liu, Mitchell Ostrow and Ila Fiete, NeurIPS 2024.

### What does “factorize” mean here?

Informally, a model's internal representation is **factorized** when different underlying factors of variation have separable structure. For a teaching example, imagine an image changes in horizontal position and vertical position. A factorized representation would capture useful distinctions between those varying features rather than treating every `(x, y)` pair as an unrelated category. The actual paper studies this in a carefully specified, highly reduced setting.

### What did the paper study?

The authors trained conditional diffusion models on controlled 2D Gaussian data and examined learned representations, composition and interpolation. They report factorized, partially continuous representational structure and compositionality with a small amount of compositional data in their experiments. The paper also connects its findings to a percolation-theory account.

### Why does it matter for us?

It shows that **marginal factors, joint examples and the representations a diffusion model learns are already serious research topics**. Our proposed controlled-mixture question must be framed relative to this close work, not presented as the first study of composition or factor learning.

Our current scenes use colored shapes and rules rather than the paper's 2D Gaussian experimental setting. We have not reproduced its results or established that the same internal representations arise in PilotUNet.

## 8. Compare the papers without confusing their contributions

| Work | Main focus | Connection to our project |
|---|---|---|
| Ho et al. | Diffusion-based image generation | Foundation for the denoising workflow |
| Liu et al. | Composing diffusion-model information during generation | Alternative inference-time approach to composition |
| Jeong et al. | Controlled multi-object generation | Motivation for an object-count and imbalance pilot |
| Okawa et al. | Composition, atomic capabilities and synthetic data frequency | Close motivation for composition references and exposure questions |
| Liang et al. | Factorized representations and compositional learning | Close prior work on factors and joint examples |

These papers use different models, datasets, interventions and metrics. Avoid comparing their numbers directly to the small 64-image pilot.

## 9. How should we state our intended contribution?

A careful statement is:

> We propose combining explicit rule semantics, estimates of individual and complete-rule exposure, constrained mixture design, identifiability checks and validated evaluation to study whether joint exposure is informative beyond marginal exposure.

The contribution is the **specific combination and proposed intervention**, not the claim that our group invented the ingredients. Our completed work includes the synthetic renderer, exposure measurement, optimization diagnostics, a small four-cell diffusion pilot and evaluation infrastructure. The main joint-exposure model experiment remains future work.

## 10. Likely evaluator questions

**Why cite both Okawa and Liang?** They study closely related aspects of compositional learning: atomic capabilities and data frequency in one case, factorized representations and composition in another. They make the motivation stronger and set a higher bar for a novelty claim.

**Is MOSAIC our dataset?** Our four-cell pilot is an adaptation of a controlled multi-object question. It is not the full MOSAIC study or a direct replication of all its datasets, models and conclusions.

**Is Liu's approach the same as changing exposure?** No. The cited work composes diffusion-model information at generation time; our proposed intervention changes aspects of the training distribution.

**Has our method outperformed these papers?** No such comparison has been established. Different tasks and metrics also make an informal numerical comparison inappropriate.

## What to tell the audience

> “Our project builds on several strands of work. DDPM gives us the diffusion framework, while composable diffusion studies combining conditions during generation. MOSAIC motivates controlled multi-object experiments. Okawa and Liang show that concept frequency, combination examples and factorized representations already matter in synthetic diffusion research. Our proposed contribution is to measure and vary complete-rule exposure while accounting for individual exposures; the direct model experiment is still to be run.”
