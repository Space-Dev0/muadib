# Slide 12 — Project status, conclusion and next steps

This guide explains the final slide of the 12-slide submission presentation. Its job is to connect what the team has built, what the existing experiments show, and what remains before the central research question can be answered. The final section gives a concise audience script.

## 1. The conclusion in one sentence

We have built and tested major components of a controlled synthetic diffusion-research pipeline, and have pilot results to guide development; the direct experiment asking whether joint exposure matters beyond marginal exposure has not yet been completed.

This is a meaningful engineering and research-methodology contribution in progress. The conclusion should distinguish **implemented components**, **measured pilot results**, and **future hypotheses**.

## 2. Return to the central research question

Suppose a training collection contains many red objects and many circles. A diffusion model might learn both concepts individually. The research question is whether seeing more examples of the **red-circle combination** improves its ability to generate a red circle, once we account for how often the ingredients were seen individually.

- **Marginal exposure:** how often an individual event occurs, such as a red object appearing in a scene.
- **Joint exposure:** how often the specified complete rule or combination holds, such as a red object that is itself a circle.
- **Composition:** correctly combining the required attributes or relations, not merely producing the ingredients somewhere in the image.

In multi-object scenes, “red appears” plus “a circle appears” can be true even if the red object is a square and the circle is blue. The bound rule “a red circle exists” is a separate event. For relations, a rule must also identify the relevant objects and check the specified direction.

Slide 4's two tables show that two distributions can have the same P(A) = P(B) = 0.5 while their joint P(A, B) differs, 0.1 versus 0.4. Those tables illustrate why controlling individual frequencies does not automatically control combination frequency. They do not report an achieved model effect.

## 3. Line 1: “Built: a scene renderer…”

The project has several concrete components:

### A. Controlled scene renderer

The renderer creates simple synthetic scenes of colored shapes and records object-level metadata such as colors, shapes and coordinates. Because the images and annotations are generated together, the team can define and check explicit visual rules. This makes it easier to investigate small, controlled questions before moving to natural images.

A scene description is the renderer's record of what it placed. That record is useful for synthetic-data measurement; it is not automatically available as ground truth for a newly generated diffusion-model image.

### B. Exposure measurement

The measurement pipeline estimates how often individual predicates and complete rules hold under candidate generating conditions. These estimated frequencies form an **exposure matrix**. The documented large measurement used 57 rule conditions and 100,000 accepted scenes per condition: **5.7 million accepted synthetic scenes** in total.

This is evidence that the renderer and measurement pipeline were run at scale. It does not mean that 5.7 million images were used in each pilot model's training or that 5.7 million diffusion outputs were evaluated. The slide 8 pilot uses 200 fixed training scenes per condition.

### C. Mixture-design optimization and diagnostics

The code can calculate mixture weights and evaluate exposure constraints and design diagnostics. A mixture says how frequently to draw from each generating condition; its estimated event exposure is a weighted combination of the condition-specific event probabilities.

The minimum-KL baseline has an important property: if the uniform mixture is feasible, it uniquely minimizes KL divergence to uniform with value zero. That baseline is useful as a reference, but it does not itself produce an exposure contrast. The separately explored max–min formulation seeks design margins; its nonlinear constraints and finite-sample uncertainty mean its candidate still needs validation.

A statistical design must produce the right contrasts **and** make the intended effects identifiable. As one example, if a marginal-exposure variable is constant within each rule, a regression with an unrestricted intercept for every rule cannot separately estimate that marginal's effect. This is a property of the proposed design and model specification, not merely a solver setting.

### D. Diffusion training and sampling

The team implemented a conditional image U-Net and a diffusion training/sampling workflow. In the reported first pilot block, four models correspond to two object counts crossed with two target-color balances. Each condition uses 200 fixed scenes, batch size 4 and 500,000 training updates. EMA weights are used for generation.

The first block evaluates 16 generated images per condition, or 64 total. A second model-training seed block is also documented, using the same dataset realization. These are pilot experiments about object count and class balance; they are not the proposed joint-exposure intervention.

### E. Review and checker workflow

Generated images were reviewed for the requested-color circle. An automated checker can also attempt a decision or abstain when it cannot make one reliably. A useful checker needs to recognize correct examples **and** incorrect examples, while reporting how often it abstains. Agreement on the subset it answers does not automatically establish accuracy on the full image set.

For the first 64-image pilot, the experimental checker answered 46 images and abstained on 18, including all three human-negative examples and both unresolved images. Consequently, those data do not establish its false-accept rate. Four returned review files were byte-identical; those files alone do not establish independent reviewer authorship. The team separately confirmed independent reviews for the second block.

### F. Networking engineering

The project also contains prototype infrastructure for sending rendered scenes and annotations over TCP through SSH forwarding, resumable compressed checkpoint transfer with SHA-256 checks, and coordinated task-parallel inference across workers. A private VPN such as Tailscale can provide a network path for SSH.

The two-worker inference check assigned one image job to each GPU laptop. This supports a small functional demonstration of coordination, not a measured multi-GPU speedup or synchronized distributed training. The network graph uses measured application-level scene throughput, including rendering and encoding; it is not a measurement of raw network link capacity or GPU training throughput.

## 4. Line 2: what does the pilot actually tell us?

The first evaluation block produced 64 images: 16 for each of the four conditions. The recorded labels were **59 yes**, **3 no** and **2 unresolved** for whether the requested-color circle was present. Because two labels are unresolved, the report considers both possible binary assignments for them.

The resulting finite-image complexity-drop range is **6.25 to 12.5 percentage points**. The color-imbalance-drop range is **−3.125 to 3.125 percentage points**. These are bounds for this particular set of images under alternate assignments of unresolved labels; they are **not confidence intervals** and do not establish a population effect.

The second seed block reported in the paper had 58 yes and 6 no labels, with a descriptive complexity drop of zero. This is one reason not to present the first block as a definitive, general result. The two blocks share a dataset realization, so they are not independent replications at the dataset level.

Most importantly, neither block manipulated full-rule joint exposure while holding the relevant marginal exposures controlled. The pilot's object-count and color-balance comparison therefore cannot answer the main exposure question, even though it exercises useful training and evaluation machinery.

## 5. Line 3: why do the exposure design and checker need validation?

### Exposure design

To attribute an outcome difference to a proposed exposure contrast, we need distributions that actually differ in the intended joint exposure, have appropriately controlled individual exposures, and are measured precisely enough to verify those properties. We should also specify which rules and which target population the conclusion concerns.

The project has identified challenges with overly common existential relations, binding-template confounds, collinearity, and uncertainty in estimated exposure matrices. Some candidate designs satisfy parts of a search objective but not all of the required checks. A solver returning a candidate is a starting point for validation, not proof that the study design is ready.

### Checker and human audit

A checker should distinguish positives from hard negatives, including wrong color, wrong shape, missing target, wrong relation and incorrect attribute binding. It should have an explicit abstention policy and be compared with appropriately independent, blinded human judgments on representative generated images.

A checker that answers only easy positive images can appear accurate on answered cases while failing to evaluate difficult or negative examples. Therefore report a confusion matrix **and coverage**, including abstentions. Synthetic renderer labels can help with development, but performance on clean synthetic images does not automatically transfer to blurry or unusual model outputs.

## 6. Line 4: the next experiment, step by step

1. **Define the target estimand and rules.** Choose a fixed set of clear, nontrivial bound rules and state exactly what effect or contrast is to be measured.
2. **Improve and validate scene/rule construction.** Counterbalance attributes and relations, include difficult negative cases, and check that a rule is not almost automatically true.
3. **Estimate event exposures with uncertainty.** Measure both singleton predicates and correctly bound full rules, using independent or held-out scene draws where appropriate.
4. **Choose and validate a tractable set of training mixtures.** Seek meaningful joint-exposure differences at controlled marginals, check feasibility and coverage across the target rules, and assess identifiability under the intended statistical model. Do not assume that per-rule optimal mixtures automatically produce one practical global intervention.
5. **Validate the evaluator.** Include independently reviewed positive, negative, relational and ambiguous generated images; report both agreement and abstention. Keep development and final evaluation separate.
6. **Train comparable models under the frozen mixtures.** Use independent dataset and model-training seeds, consistent model architecture and documented compute. Paired choices can reduce nuisance variation when appropriate, but paired noise seeds do not replace independent trained models.
7. **Generate and score outputs.** For each rule and model, generate a prespecified number of samples and evaluate the complete rule with the validated procedure and/or blinded human review.
8. **Analyze the predefined contrast.** Compare rule satisfaction with the actual measured exposures and uncertainty; distinguish training-seed, dataset-seed, and image-sampling variation.
9. **Report outcomes honestly.** A null, negative, uncertain, or positive result is informative if the intervention and measurement are valid. Present confidence intervals or other uncertainty measures appropriate to the actual replicated design, not the finite-image ambiguity bounds as if they were confidence intervals.

## 7. What is rule satisfaction rate?

For a given rule $r$, **rule satisfaction rate** (RSR) is the proportion of evaluated generated samples that satisfy the whole rule:

$$
 \widehat{\mathrm{RSR}}_r
 =\frac{\text{number of evaluated outputs satisfying }r}
        {\text{number of evaluated outputs with a defined decision}}.
$$

For example, if 40 out of 100 fully evaluated images satisfy “a red circle is left of a blue square,” the observed RSR is 0.40. The rule requires the correct attributes on the correct distinct objects and the specified relation. If an evaluator abstains on some outputs, the denominator and abstention count must be reported; RSR among accepted cases alone may describe a selected subset rather than all generated images.

RSR is an **outcome** measured on generated images. Marginal and joint **exposures** describe the training distribution. Do not use these terms interchangeably: exposure is what the model is shown; satisfaction is how a generated output is evaluated.

## 8. Likely questions from evaluators

**What is the project's novelty?** The proposed contribution is a combined controlled study of complete-rule exposure versus constituent exposure, with explicit rule semantics, mixture design and identifiability diagnostics. Controlled synthetic composition, diffusion, frequency effects, and mixture optimization already exist in prior work. The novel claim is a proposed integration and specific intervention, not that nobody has studied compositional diffusion or concept frequency before.

**Is the central hypothesis proven?** No. We have tools, design diagnostics and feasibility pilots, but the proposed joint-exposure training intervention remains to be performed and evaluated.

**Why use synthetic images?** They offer known object annotations and controllable combinations, allowing us to isolate and audit proposed exposure events. The trade-off is that findings do not automatically generalize to natural images or large pretrained models.

**Do the 5.7 million scenes mean that the four diffusion models trained on 5.7 million images each?** No. Those scenes belong to exposure measurement across 57 conditions. Each of the four pilot datasets contains 200 fixed training scenes.

**Did Tailscale or two GPUs accelerate training?** The project uses network and worker prototypes and performed a small task-parallel inference check. It has not established synchronized multi-GPU training or a measured GPU speedup.

**What result should be on the final slide?** State the scale of the measurement and what the pilot evaluated, then state the next controlled test. Avoid calling the proposed relationship proven or treating reviewer counts as a general accuracy guarantee.

## What to tell the audience

> “We have built a controlled scene renderer, measured predicate and rule exposure, developed mixture-design and diffusion-training tools, and evaluated a four-condition pilot. The large measurement covered 5.7 million synthetic scenes across 57 conditions; the separate pilot trained four models on 200 scenes per condition. Our main question is whether showing a particular combination more often matters beyond showing its individual parts. The next step is to validate the exposure contrast and checker, then train independently replicated models on controlled mixtures and measure complete-rule satisfaction.”
