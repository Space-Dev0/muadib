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

H1 predicts that full-rule success falls below the product reference, increasingly so for relational complexity. H2 predicts that, at fixed predicate marginals, the composition gap increases with joint lift and that the association is stronger for relational rules. The two tables use 0 for absent and 1 for present; their individual marginals remain at 0.5 while the joint probability changes from 0.1 to 0.4. These numbers illustrate the idea and are not experimental results.

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

## Slide 9 — Six research hypotheses

These are the six hypotheses in the research proposal. None has been established by the current exploratory pilot. Explain that the composition gap Γ compares rule success with a product reference for the individual predicate successes; it is zero when the two values are equal. This reference is a diagnostic and does not by itself establish independence or causality.

**H1 — Composition costs.** Full-rule success is expected to fall below the product reference, with a more negative gap as relational complexity rises. This asks whether combining concepts is harder than the reference predicts.

**H2 — Joint exposure.** At fixed individual predicate marginals, the proposal predicts that Γ increases with joint-exposure lift and that the slope is greater for relational rules. This needs controlled exposure contrasts and an identified model.

**H3 — Pairwise support.** Hold out a full rule while retaining exposure to its component pairs, then compare this with a condition where those pairs are also held out. The prediction is that retaining pairs improves zero-shot success on the full rule.

**H4 — Context breadth.** At comparable predicate frequency, a concept seen with more diverse partners is predicted to require less exposure to reach N50, the point associated with 50% success. This requires a defined breadth measure and an observed learning curve that brackets the threshold.

**H5 — Update transfer.** Features measuring how updates on one rule affect others are predicted to improve out-of-run predictions of rule learning speed beyond exposure and complexity, and their signs are predicted to track multi-step ablation effects. This requires independent held-out evaluation.

**H6 — Where errors occur.** Excess composition loss is predicted to concentrate in particular log-SNR noise bands, with band patterns linked to the data’s per-frequency SNR. For relational rules it is also expected around object boundaries and the gaps between objects.

The proposal makes the allocation optimization contribution conditional on H2 or H5. The four-cell pilot tests a different question and does not resolve H1–H6.

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

## Additional notes for the expanded CoMA and CN slides

### Slide 6 — Computational optimization (CoMA)

The exposure matrix A records estimated probabilities for each event in each candidate condition. A training mixture is a nonnegative vector w that sums to one. Thus m_p = Σ_j A_pj w_j is the marginal exposure to predicate p and n_r = Σ_j A_rj w_j is the joint exposure to rule r. The lift ℓ_r = log n_r − Σ_{p in r} log m_p expresses joint exposure relative to the product of individual marginal exposures.

The proposed factorial design searches for four shared mixtures. It varies marginal exposure and joint lift across the mixtures and centers predictors within each rule to support an identified within-rule analysis. Our max–min-margin solver introduces z and maximizes it subject to the selected design constraint margins being at least z and the mixture weights satisfying their bounds and sum-to-one constraints. SLSQP found a local candidate; this does not establish a global optimum.

For the exploratory six-rule candidate, a new 82924 draw gave 72/72 nonnegative point margins and a within-rule information condition number of 3.33. Simultaneous 95% bounds certified 47 of the 72 constraints, while 25 remained unresolved; none were certified violated. A six-rule exploratory result does not cover the full benchmark, and Stage 0 has not passed. The current rule and mixture design needs broader validation.

### Slide 11 — Computer networks (CN)

The data plane uses TCP to stream rendered scenes plus JSON annotations. A CPU renderer, serialization, transport and receiver are part of the measured path. On localhost, 300,000 scenes per batch size gave 1,082, 1,061 and 1,060 scenes/s for batch sizes 2, 8 and 16, respectively. Separate first-laptop SSH-tunneled runs with 300 scenes each yielded 582, 801 and 926 scenes/s. These are single short trials and not measures of raw link capacity or integrated training throughput.

For fault tolerance, a compressed checkpoint was sent with a deliberate connection drop after 50,000 compressed bytes; the receiver resumed from that offset and verified a SHA-256 hash. The hash checks content integrity, not authentication. SSH tunnels provide the protected transport used in this test.

For task-parallel inference, the coordinator sent separate image jobs to two GPU laptops, each producing one image. This verifies a small end-to-end path; it is not synchronous distributed gradient training and does not establish a speedup. GPU idle latency was not measured.

## Additional notes: convex baseline and model training

### Slide 6 — What does minimum KL solve?

Let w be the vector of mixture probabilities across J training conditions and u the uniform reference, u_j = 1/J. The convex problem minimizes KL(w || u) = Σ_j w_j log(w_j/u_j) subject to weights summing to one, a lower weight floor, and linear lower and upper exposure bounds L_e ≤ A_e: w ≤ U_e. A contains estimated event probabilities, so these constraints are empirical rather than guarantees about unknown probabilities.

Gibbs' inequality says KL(w || u) ≥ 0, with equality exactly when w = u. Therefore, if uniform is feasible, w = u is the unique constrained solution and the minimum is exactly zero. A tiny numerical residual near zero is expected, not an optimization defect. If uniform is excluded while the feasible set is nonempty, the optimum is positive. Minimum KL chooses the feasible mixture closest to uniform; it does not itself create a low/high exposure contrast.

The separate max–min exploratory procedure searches over four shared mixtures and maximizes the smallest specified constraint margin. Because the relevant exposure and lift constraints involve nonlinear functions, do not label the whole four-mixture procedure as the same convex KL problem. In the six-rule 82924 draw, 72 point margins passed, while simultaneous 95% bounds certified 47 and left 25 unresolved. A local solver success is not a global certificate and Stage 0 remains open.

### Slide 7 — Gradient descent and AdamW

A training loss measures the difference between the diffusion model's predicted velocity and the target velocity. The gradient g_t = ∇_θ L_t points locally in the direction of increasing minibatch loss. Plain gradient descent updates θ by subtracting learning rate η times g_t; minibatch stochastic gradient descent uses an estimate of the full-data gradient.

AdamW keeps running first and second moments m_t = β₁m_{t−1} + (1−β₁)g_t and v_t = β₂v_{t−1} + (1−β₂)g_t², then forms bias-corrected m̂_t and v̂_t. Its update θ_{t+1} = (1−ηλ)θ_t − η m̂_t/(√v̂_t + ε) separates weight decay from the adaptive gradient update. Here λ is weight decay, ε improves numerical stability, and division and squares are elementwise. This optimizer is for neural network training; the convex KL mixture design is a separate optimization task. The reported pilot uses AdamW with learning rate 10⁻⁴ and an EMA model for sampling. These formulas describe the method and do not establish that extra training would answer the exposure hypothesis.
