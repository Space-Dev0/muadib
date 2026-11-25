# Slide 4 — Methodology, marginal exposure and joint exposure

This guide explains the fourth slide of the 12-slide submission presentation. Read it to understand the ideas; the final section is a shorter script you can say to the audience.

## 1. What question does this slide answer?

The project asks whether a diffusion model's experience with a **particular combination** affects its ability to generate that combination, even when it has seen the individual ingredients frequently.

For example, a training set may contain many red objects and many circles. Does that mean it contains many **red circles**? No. The color red and the shape circle could occur on different objects or in different images. This distinction motivates the study.

The slide introduces a proposed way to build training distributions that separate these two types of experience. The tables are a **constructed mathematical example**, not a report of model performance.

## 2. Start with the vocabulary

**Exposure** means how often an event occurs in the training distribution. A frequency such as 0.5 means the event occurs in about half of the relevant examples under that distribution. In a fixed dataset, one can estimate it by counting occurrences and dividing by the number of examples. In training with replacement, the number of times the model actually encounters an event also depends on the sampling procedure and training duration.

A **predicate** is a condition that can be true or false, such as “a red object is present,” “a circle is present,” or “one specified object is to the left of another.” An **annotation** records a scene's objects and their properties or positions; it lets our software check whether a predicate or a full rule is true.

**Marginal exposure** is the frequency of an individual event when we ignore whether the other event also holds. If event A means “red is present,” then P(A) measures how often A is true across all examples, including examples where B is true and examples where B is false.

**Joint exposure** is the frequency with which the specified events hold together. For simple binary events A and B, it is P(A ∩ B), also written P(A, B). For an object-bound rule such as “a red circle,” the project must check that the color and shape apply to the **same object**. For a spatial rule, the properties must apply to the specified distinct objects and the specified relation must hold.

> Important: “There is a red object somewhere and a circle somewhere” is not the same event as “there is a red circle.” A red square next to a blue circle satisfies both scene-level singleton predicates, but it does not satisfy the bound red-circle rule. The two-event tables on this slide are a probability illustration; actual rule exposure must use the project's binding semantics.

A **training mixture** specifies how frequently we sample from different candidate scene-generating conditions. A weight of 0.2 for a condition means it receives 20% of mixture probability, not that 20% of its images must be kept in a particular fixed dataset.

## 3. Read the two tables, cell by cell

Let A and B be two binary events. The labels `1` and `0` mean that the event is present or absent. For a simple concrete analogy, imagine each example contains one object, A means “the object is red,” and B means “the object is a circle.” This single-object analogy avoids the binding ambiguity above.

The table entries are proportions of all examples; each table sums to 1.

### Low co-occurrence

| | B absent | B present | Row total |
|---|---:|---:|---:|
| A absent | 0.1 | 0.4 | 0.5 |
| A present | 0.4 | 0.1 | 0.5 |
| Column total | 0.5 | 0.5 | 1.0 |

- The bottom row gives P(A) = 0.4 + 0.1 = **0.5**.
- The right column gives P(B) = 0.4 + 0.1 = **0.5**.
- The bottom-right cell gives P(A, B) = **0.1**.

Out of 100 illustrative examples, 50 satisfy A, 50 satisfy B, but only 10 satisfy both.

### High co-occurrence

| | B absent | B present | Row total |
|---|---:|---:|---:|
| A absent | 0.4 | 0.1 | 0.5 |
| A present | 0.1 | 0.4 | 0.5 |
| Column total | 0.5 | 0.5 | 1.0 |

- P(A) = 0.1 + 0.4 = **0.5**.
- P(B) = 0.1 + 0.4 = **0.5**.
- P(A, B) = **0.4**.

Out of 100 illustrative examples, 50 satisfy A, 50 satisfy B, and 40 satisfy both.

### What changed?

The separate event frequencies are unchanged: A is present 50% of the time and B is present 50% of the time in both cases. Their joint frequency changed from 10% to 40%: an increase of **30 percentage points**, or four times the original joint frequency. “Four times” describes this illustrative ratio; it is not a fourfold improvement in model performance.

If these were actual training datasets with one object per example, they would contain equally many red objects and circles, but different numbers of red circles. This is the kind of contrast the project wants to test.

For these particular binary marginals, the independent-events reference is P(A)P(B) = 0.25. The low table has joint probability 0.1; the high table has 0.4. Independence is a comparison point, **not an assumption that real predicates must be independent**.

## 4. Explain the workflow shown on the left

### Step 1: Render scenes

A synthetic renderer creates small images of colored shapes and records the objects, their attributes, and their positions. Because we have annotations, we can evaluate a rule against a scene description without guessing what objects the renderer intended.

### Step 2: Estimate exposure probabilities

For each candidate scene condition, render many scenes and count how often each predicate and each full rule holds. A row in the resulting exposure matrix corresponds to an event; a column corresponds to a candidate condition. An entry is an estimated event probability under that condition.

For example, suppose 70 out of 100 annotated scenes generated under a particular condition contain a red object. The estimated scene-level marginal for that event in that condition is 0.70. This is an estimate, not proof that its true probability is exactly 0.70.

### Step 3: Design mixtures

Choose nonnegative weights for the candidate conditions. The weights sum to one. Combining the estimated event rates with these weights estimates the event rates under the mixture:

$$
 m_p=\sum_j w_j A_{pj}, \qquad n_r=\sum_j w_j A_{rj}.
$$


Here `j` indexes candidate conditions; `w_j` is the sampling weight of condition `j`; `A_pj` is the probability that individual predicate `p` holds in condition `j`; and `A_rj` is the probability that complete rule `r` holds there. Thus `m_p` is a mixture-level marginal and `n_r` is a mixture-level full-rule exposure. We estimate the A values from generated scenes, so the resulting mixture values also have estimation uncertainty.

The intended design varies `n_r` while holding relevant marginals close to specified targets. Constraints set acceptable ranges and ensure valid mixture weights. Whether a proposed contrast is actually feasible and sufficiently well identified must be checked; it does not follow automatically from writing down the constraints.

### Step 4: Train and evaluate

The intended exposure study would train comparable models under the selected distributions, generate examples for specified rules, and compare rule satisfaction. Reliable interpretation requires adequate contrast, replication, and a validated evaluator. The current four-cell diffusion pilot described on slide 8 is a **separate feasibility experiment**; it did not train models on these proposed joint-exposure mixtures.

## 5. Where does optimization enter?

Optimization chooses mixture weights that satisfy explicit requirements. For example, we may bound individual-event frequencies while seeking a contrast in a full-rule frequency. The constraints, rather than an informal selection of examples, make the intended control visible and testable.

The next slide discusses two different objectives. The minimum-KL baseline finds a feasible mixture close to uniform; if uniform is feasible, it returns uniform with objective zero and does **not** by itself create a contrast. The separate exploratory max–min formulation seeks design margins. The latter is not simply the same convex problem, and its candidate has not established a completed exposure intervention.

**Exposure lift** is one descriptive way to compare joint exposure with a product of marginal exposures:

$$
 \ell_r=\log n_r-\sum_{p\in r}\log m_p
       =\log\!\left(\frac{n_r}{\prod_{p\in r}m_p}\right).
$$

It is defined only when the relevant probabilities are positive. A larger value means a larger joint-to-product ratio for the chosen events; it is not a measure of model accuracy. For scene-level existential predicates and object-bound rules, the product does not automatically represent an appropriate independence model, so the event definitions must be stated carefully.

## 6. Questions you may be asked

**Why not just increase all training data?** More data can change both individual and joint exposure at once. The research question requires distinguishing their contributions, so the proposed design tries to control them separately.

**Does equal marginal exposure guarantee equal joint exposure?** No. The slide is a constructive counterexample: both marginals are 0.5 in both tables, while the joint is 0.1 in one and 0.4 in the other.

**Does the high-joint table prove that models will perform better?** No. It shows that the data distributions can differ in this way. Model training and reliable evaluation are needed to test the performance claim.

**Is a scene with a red square and blue circle a red-circle example?** No. Both “red exists” and “circle exists” are true, but no single object is a red circle.

**Why are A-matrix values estimated?** They are calculated from a finite set of randomly generated scenes. A held-out measurement and uncertainty analysis help determine whether apparent constraints persist beyond the sample used to design the mixture.

## What to tell the audience

> “This slide explains our central distinction. Marginal exposure counts how often an individual concept occurs; joint exposure counts how often the complete combination occurs. In both tables A and B each occur 50% of the time, but they occur together 10% in one distribution and 40% in the other. We render and annotate scenes, estimate these event frequencies, and use constrained mixture weights to seek such contrasts. The tables are an illustration of the proposed intervention; the direct model experiment still needs to be completed.”
