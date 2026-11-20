# Presenter notes — 12-slide submission

This deck follows the supplied presentation format. Distinguish the proposed joint-exposure study from the separate completed pilot. The main hypothesis remains untested and Stage 0 has not passed.

## Slide 1 — Introduction

A diffusion model learns to generate images by reversing a gradual noise-corruption process. At generation time it starts from random noise and repeatedly applies learned denoising steps, guided by a condition. The pictures show illustrative linear blends of a clean render and fixed noise, not model outputs or the actual diffusion schedule. Our project asks whether it can combine familiar concepts and how seeing those combinations in training affects generation. A model might have seen many red objects and many squares but few red squares. We construct a controlled synthetic environment to measure the difference between seeing parts and seeing their combination. The project connects digital image processing, computational optimization and computer networks.

## Slide 2 — Literature survey

DDPM (Ho et al., 2020) provides the denoising foundation. Liu et al. study composable diffusion models. MOSAIC (Jeong et al., 2026) studies multi-object generation in controlled synthetic scenes. Okawa et al. (NeurIPS 2023) find that composing lower-frequency concepts into out-of-distribution samples required considerably more optimization steps than generating in-distribution samples. Our proposed extension is to intervene on nonzero joint exposure while controlling individual exposures. This is a research design, not yet an established empirical result; the larger literature already studies related questions.

## Slide 3 — Research hypotheses

H1 predicts negative composition gaps that grow with relational complexity. H2 predicts that, at fixed marginals, the composition gap increases with joint-exposure lift, especially for relational rules. H3 compares a held-out rule with its pairs retained versus also held out. H4 predicts that broader partner contexts lower the exposure needed for 50% success (N50) at comparable frequency. H5 predicts that update-transfer features improve held-out learning-speed predictions and predict ablation directions. H6 predicts excess loss in particular log-SNR bands and, for relations, in inter-object gaps and object boundaries. These are proposal hypotheses, not results of the four-cell pilot.

## Slide 4 — Methodology and exposure optimization

Render labeled scenes, estimate the exposure matrix, choose mixtures, train models, sample and evaluate. The tables demonstrate the same P(A)=P(B)=0.5 but joint P(A,B)=0.1 or 0.4. They are an illustration, not measured model results.

The convex baseline minimizes KL(w || u) over nonnegative mixture weights summing to one subject to linear exposure bounds. If uniform u is feasible, w=u uniquely minimizes KL at zero; this baseline alone creates no contrast. If uniform is excluded, a nonempty closed feasible set has a positive optimum. The separate four-mixture max–min approach seeks contrast but is exploratory and not the same convex optimization problem. Its six-rule fresh draw had 47 of 72 margins certified and 25 unresolved under simultaneous bounds. Stage 0 has not passed.

## Slide 5 — Model architecture

PilotUNet is a time- and condition-aware image U-Net. For the pilot, its base width is 16. The encoder maps a 64×64 noisy image through 32×32 and 16×16 spatial features. Timestep embeddings enter residual blocks, and the 16×16 bottleneck uses four-head cross-attention over the condition tokens. The decoder predicts the velocity target while skip connections retain spatial information. During training x_t combines x_0 and Gaussian noise through a schedule. Velocity v_t is another parameterization of the denoising target. During sampling, start from noise and perform reverse steps using EMA weights. The diagram is schematic; the implementation uses cross-attention at 16×16.

## Slide 6 — Dataset

The project renderer produces colored shapes with known object attributes and coordinates. The separate exposure matrix contains 57 conditions and 5.7 million accepted synthetic scenes. The four-cell pilot uses four datasets of 200 fixed scenes each. Target circles are red or blue, whereas distractors have other attributes, so target role is confounded with attributes. No independent held-out image dataset is claimed for this pilot. The image shown is an illustrative clean render.

## Slide 7 — Experimental setup and training

Cross two vs. three objects with 50:50 vs. 90:10 target color balance. For each cell train a separate width-16 PilotUNet on 200 fixed 64×64 images, batch size 4, for 500,000 updates. Each step samples image, timestep and Gaussian noise, constructs a noised image, predicts velocity and minimizes mean-squared error using AdamW at learning rate 1e-4. An EMA model is used for evaluation. Generate red and blue targets with eight paired noise seeds, producing 16 images per cell. One model-training seed block does not provide independent dataset replication; noise seeds are not training seeds.

## Slide 8 — Quantitative results

Across the 64 images, the recorded review labels are 59 yes, three no and two unresolved. Cell counts are 16/0/0, 15/0/1, 14/2/0, 14/1/1. Resolving the two uncertain labels in all possible ways yields complexity-drop bounds of 6.25–12.5 percentage points and imbalance-drop bounds of −3.125–3.125 points. These are finite-sample sensitivity bounds, not confidence intervals. Four returned CSVs were byte-identical; these files alone do not establish independent authorship. The pilot does not test the joint-exposure hypothesis.

## Slide 9 — Qualitative results

Show the two selected generated samples on the left: sample 0003 received a yes for target circle presence, while sample 0001 was uncertain. These are selected examples, not a random gallery or paired pixel reconstructions. On the right, explain the evaluation: four 500,000-update pilot models produced 64 images. The returned labels were 59 yes, three no and two unresolved. The experimental checker answered 46 of 64 and abstained on every one of the three human-negative examples. These exploratory pilot results do not test the central joint-exposure hypotheses.

## Slide 10 — Ablation and failure cases

The current four-cell experiment is a data-factor comparison, not an architecture ablation. No measured attention or U-Net-component effect should be claimed. The experimental checker answered 46 of 64 questions and abstained on all three human-negative cases as well as both unresolved cases. Agreement among accepted cases cannot identify the false-accept rate; negative cases and abstentions matter. Broader generated-image checker validation remains open.

## Slide 11 — Discussion: novelty and networking

The proposed research novelty is controlled variation of full-rule exposure while accounting for constituent exposure; the outcome experiment has not yet been run. The networking prototype is an engineering contribution combining established methods, not a new networking algorithm. A CPU producer streams rendered PNG scenes and JSON annotations over TCP through an SSH tunnel. A separate gzip checkpoint protocol resumes after an intentional 50,000-byte drop and verifies the bytes with SHA-256. Two GPU laptops each generated one inference image; this is task-parallel inference, not synchronized distributed training or proof of speedup. The localhost 1,060–1,082 scenes/s numbers include rendering and encoding; GPU idle time and raw link capacity were not established.

## Slide 12 — Conclusion and future work

We built the renderer, measurement and design tools, diffusion pilots, checker audit and networking prototypes. Results currently support feasibility and diagnostics, not the central exposure claim. Next validate the checker on challenging positive, negative and relational images; establish an identifiable exposure design; then run independent training and data seeds and test the proposed hypotheses. Stage 0 has not passed.

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020: https://arxiv.org/abs/2006.11239
- Salimans and Ho, *Progressive Distillation for Fast Sampling of Diffusion Models*, 2022: https://arxiv.org/abs/2202.00512
- Liu et al., *Compositional Visual Generation with Composable Diffusion Models*, 2022: https://arxiv.org/abs/2206.01714
- Okawa et al., *Compositional Abilities Emerge Multiplicatively*, NeurIPS 2023: https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html
- Jeong et al., *When Do Diffusion Models Learn to Generate Multiple Objects?*, 2026: https://proceedings.mlr.press/v306/jeong26b.html
