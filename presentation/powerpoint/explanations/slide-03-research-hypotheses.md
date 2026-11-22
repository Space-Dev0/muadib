# Slide 3 — Research hypotheses H1–H6

This guide explains the third slide of the 12-slide submission presentation, including the notation, the reasoning behind each hypothesis, and how each would be tested. The final section is a short script for an audience that has not studied diffusion. **All six are proposed hypotheses, not results established by the four-condition pilot.**

## 1. What is a research hypothesis?

A hypothesis is a specific, testable prediction. It is not a conclusion simply because it appears on a slide. Our broad question is: **when the individual ingredients are familiar to a model, does additional exposure to a particular combination predict whether the model can generate that combination?**

For example, the model may have seen red objects and circles often, but relatively few red circles. We want to distinguish familiarity with the parts from familiarity with the complete combination. Later hypotheses extend this question to relations, context, training dynamics and the location of errors.

## 2. Vocabulary used throughout the slide

A **predicate** is a basic true-or-false statement, such as “object A is red,” “object A is a circle,” or “object A is left of object B.” A **rule** is a set of predicates that must hold together with the intended object bindings. For “a red circle is left of a blue square,” the red and circle properties must apply to the same first object; the blue and square properties to the same second object; and the relation must hold between those distinct objects.

**Marginal exposure** is the frequency of an individual event in the training distribution. **Joint exposure** is the frequency of the specified full combination. Seeing red objects and circles separately does not establish how often the model saw a red circle. The two tables on slide 4 demonstrate this distinction for two simple binary events.

**Rule satisfaction rate** (RSR) is an output measure: the fraction of evaluated generated images that satisfy the complete requested rule. It is different from exposure, which concerns what the model was shown during training. If a checker abstains, its coverage and the evaluation denominator must be made explicit.

**Relational complexity** refers to the number or nature of relationships in a rule. “A red circle” is an attribute-binding rule; “a red circle left of a blue square” also specifies a relation. The project must define and counterbalance complexity rather than assume that every longer rule is automatically comparable.

## 3. H1 — Composition cost

### What the slide says

> Full-rule success falls below a product reference; the gap may become more negative as relational complexity rises.

The idea is to compare success on a complete rule with a reference constructed from success rates for its constituent predicates. The reference is a diagnostic comparison, not a guarantee that the constituent outcomes are independent.

A proposed index is

\[
 \Gamma_r=\operatorname{logit}(\mathrm{RSR}_r)
 -\operatorname{logit}\!\left(\prod_{p\in r}c_p\right),
 \qquad \operatorname{logit}(u)=\log\!\left(\frac{u}{1-u}\right).
\]

Here `r` is a rule, `p` indexes its predicates, `c_p` is a corresponding primitive success rate, and `RSR_r` measures complete-rule success. The exact prompt and sample population used to estimate each `c_p` must be specified; primitive-only prompts and measurements within composite prompts answer different questions.

For an easy numerical example, suppose two primitive success rates are each 0.8, giving a product reference of 0.64. If the complete-rule RSR is 0.50, the complete rule falls below that reference, and the logit difference is negative. If complete-rule RSR equals 0.64, **the index is exactly zero**. The logit of a product is not the sum of individual logits. Empirical rates of 0 or 1 require a prespecified treatment because their logits are infinite.

The hypothesis predicts a negative composition gap, potentially becoming more negative for relationally complex rules. To test it, we would need a validated evaluator, a clearly defined reference, comparable rule groups and replicated trained models. A negative gap would be evidence relative to that chosen reference, not proof of a particular internal cause.

## 4. H2 — Joint exposure beyond marginal exposure

### What the slide says

> At fixed predicate marginals, the composition gap increases with joint-exposure lift; compare relations with attribute-only rules.

The intuition is that the model may benefit from seeing a complete combination, even when it already has similar amounts of experience with each part. The key phrase is **at fixed or appropriately controlled marginal exposure**: otherwise, the model might improve merely because it saw an individual ingredient more often.

One descriptive exposure-lift quantity is

\[
 \ell_r=\log n_r-\sum_{p\in r}\log m_p
       =\log\!\left(\frac{n_r}{\prod_{p\in r}m_p}\right),
\]

where `n_r` is full-rule exposure and `m_p` are individual-event exposures, with all terms positive. The slide denotes joint lift by $\lambda$; this guide uses $\ell$ to avoid confusing it with an optimizer's weight-decay coefficient. If the marginals are genuinely fixed, increasing `n_r` increases this lift. With changing marginals, the ratio can change for several reasons.

**Example:** in slide 4, both individual marginals equal 0.5 in each illustrative table, while the joint changes from 0.1 to 0.4. That shows a possible exposure contrast. It does **not** show that `Γ` or RSR improves: a trained-model experiment is needed for that.

For real multi-object scenes, an existential scene-level predicate and a same-object bound rule are different events. Their probability product is not automatically a valid independent-object baseline. We must also ensure that relation types are not confounded with binding structure.

A test would predefine a target rule set, design feasible mixtures, measure their achieved marginals and joint exposures on held-out scenes, train comparable replicated models and evaluate the complete rules. Identifiability matters: if a covariate is fixed within each rule and the analysis includes an unrestricted intercept per rule, its independent effect cannot be separately estimated in that specification.

## 5. H3 — Pairwise support

### What the slide says

> A held-out complete rule may succeed better when its constituent predicate pairs are retained in training.

Suppose a full rule contains three ingredients A, B and C. Consider two intended training designs, both withholding the complete A-and-B-and-C rule:

- **Rule-only holdout:** retain selected pairwise combinations such as A+B, A+C and B+C.
- **Rule-and-pairs holdout:** also remove the relevant pairwise combinations.

H3 predicts better performance on the held-out full rule when pairwise support remains. The reasoning is that pairwise exposure may provide partial evidence for combining the full rule. That is a hypothesis, not an observed mechanism in this project.

A careful experiment must define “pair” with the same variable bindings as the full rule. It must also check what else changes: removing pairs can inadvertently alter single-predicate frequencies, scene composition or difficulty. Those differences require control or explicit analysis. A held-out rule means it is excluded from the relevant training condition, not merely absent from a particular minibatch.

## 6. H4 — Context breadth and N50

### What the slide says

> More varied partner concepts may reduce N50 at matched frequency.

**Context breadth** describes how many distinct, relevant partners or contexts a concept occurs with. For instance, a red object might appear with a variety of shapes or neighboring objects instead of repeatedly appearing in the same narrow combination. The exact breadth measure must be fixed before analysis.

**N50** is the exposure level associated with 50% rule success, under the chosen exposure scale and fitted learning curve. It is not “50 training examples” and it is not necessarily a raw count unless the study defines its exposure axis that way.

A hypothetical comparison: if a broad-context condition reaches 50% rule success around exposure 100, while a narrow-context condition reaches it around exposure 200, the broad condition has the lower N50 under that specified scale. H4 predicts that direction at matched or appropriately controlled frequency. Those values are invented for explanation, not project results.

N50 may be unestimable if a learning curve never crosses 50% in the observed range, or uncertain if measurements are sparse. Matching overall frequency while changing context can itself be difficult, so both must be measured.

## 7. H5 — Update transfer

### What the slide says

> Transfer features may add predictive value beyond exposure and complexity; check held-out learning speed and multi-step ablations.

An update to model parameters while training on one example could potentially help or hinder performance on another rule. A **transfer feature** summarizes a specified relationship between such updates and an outcome. For example, a researcher might evaluate how a controlled update on one rule changes a loss associated with another rule. The precise feature, update size and sign convention must be defined by the experiment; the hypothesis does not make every possible transfer metric valid.

“Adds predictive value beyond exposure and complexity” means asking whether the transfer feature improves predictions after the baseline information is already included. A valid check separates feature construction and model fitting from the held-out evaluation; otherwise, evaluation data can leak into the predictor.

A multi-step ablation would compare a prespecified training intervention with a matched reference and ask whether observed effects agree with the proposed transfer prediction. A single gradient or an in-sample correlation alone cannot establish a causal multi-step benefit.

## 8. H6 — Where and at which noise levels do errors occur?

### What the slide says

> Excess loss may concentrate in log-SNR bands linked to data frequency; for relations, inspect object boundaries and gaps.

A diffusion model is trained at many noise levels. **Signal-to-noise ratio** (SNR) compares the strength of the retained image signal with that of noise. A log-SNR scale takes the logarithm of that ratio and provides a convenient way to group noise levels. Different schedule conventions and exact band boundaries should be stated when reporting results.

H6 asks whether a difference between chosen conditions is concentrated in particular noise-level bands, instead of being uniform everywhere. It also asks whether spatially localized errors appear around object boundaries or in the space between objects, which can be relevant to relational scenes.

This requires a specified comparison, a loss and a defensible region-mask construction. Regions may differ in area, so summarize per-pixel error appropriately rather than comparing unnormalized totals. Noise-level loss is a diagnostic; it does not automatically prove which visual feature caused a generation error.

## 9. How the hypotheses fit together

| Hypothesis | Main question |
|---|---|
| H1 | Is complete-rule success below the chosen primitive-based reference? |
| H2 | Does controlled joint exposure help explain the composition gap? |
| H3 | Does retaining pairwise support help a held-out full combination? |
| H4 | Does greater context breadth change the exposure level needed for 50% success? |
| H5 | Do update-transfer features add predictive value or predict interventions? |
| H6 | At which noise levels and image regions are errors concentrated? |

These ideas need different interventions and measurements. The existing four-cell pilot varies object count and color balance; it does not directly execute all six tests.

## 10. Likely evaluator questions

**What is the difference between exposure and RSR?** Exposure concerns events in training data. RSR is a measured outcome on generated images.

**Is Γ necessarily negative?** No. Under the stated product reference it is exactly zero if RSR equals the product, and it can be positive or negative when the quantities are strictly between zero and one.

**Does high lift prove causation?** No. Lift is a descriptive exposure statistic. A causal claim requires a well-defined intervention, appropriate controls, replication and trustworthy measurement.

**Have the six hypotheses passed?** No. They describe the proposed research; the current pilot does not establish them.

## What to tell the audience

> “We have six hypotheses. First, a model may find a complete rule harder than its individual ingredients. Second, seeing the full combination more often may matter even when the ingredients are equally familiar. We also plan to test pairwise support, the diversity of contexts, whether training on one rule helps another, and where errors occur across noise levels and image regions. These are planned, testable predictions; the current four-condition pilot is a feasibility study rather than a direct test of all six.”
