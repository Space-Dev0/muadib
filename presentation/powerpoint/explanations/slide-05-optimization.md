# Slide 5 — Optimization: minimum KL and max–min

This guide explains the fifth slide line by line, including what is optimized, why the minimum-KL baseline can return zero, and how the separate exploratory max–min problem is interpreted. The final section is a short audience script.

## 1. What problem are we trying to solve?

Our proposed research asks whether full-rule or **joint exposure** matters beyond the exposure of individual predicates. To investigate that question, we would like to build training distributions with meaningful differences in full-rule exposure while controlling other relevant exposure quantities.

The data pipeline has multiple candidate scene-generating conditions. Instead of choosing examples informally, we assign each condition a **mixture weight**. Optimization helps us choose weights that obey explicit requirements.

This slide compares two different optimization problems:

1. A **convex minimum-KL baseline**: find a feasible mixture close to uniform.
2. A **separate exploratory max–min design**: seek the largest minimum constraint margin over four mixtures.

A good optimization objective must match the research goal. “Closest to uniform” and “creates a useful exposure contrast” are not the same goal.

## 2. A simple mixture example

Suppose there are four candidate scene conditions and their weights are

\[
 w=(w_1,w_2,w_3,w_4)=(0.10,0.20,0.30,0.40).
\]

Each weight is nonnegative and they add to 1. Sampling according to `w` means selecting condition 1 with probability 0.10, condition 2 with probability 0.20, and so on. A weight describes a distribution over conditions; it is not a model parameter or a gradient.

The **uniform** reference for four conditions is

\[
 u=(0.25,0.25,0.25,0.25).
\]

For `J` conditions, each uniform weight is `u_j = 1/J`.

The actual mixture pipeline estimates event probabilities under each condition. A mixture's event probability is a weighted average of those condition-specific estimates. Optimization selects weights; it does not directly change what an already-generated scene contains.

## 3. Review marginal and joint exposure

A **predicate** is a basic true-or-false event, such as a red object being present. A **rule** is a complete conjunction with the intended bindings, such as a red object that is itself a square.

- Marginal exposure `m_p`: the frequency of individual predicate `p` under the training mixture.
- Joint or full-rule exposure `n_r`: the frequency with which complete rule `r` holds under the mixture.

If a scene contains a red circle and a blue square, it satisfies “red exists” and “square exists,” but it need not satisfy the same-object rule “red square exists.” The annotations and event definitions must preserve the difference.

Let `A_pj` be the probability that predicate `p` holds in condition `j`, and let `A_rj` be the probability that full rule `r` holds there. Then

\[
 m_p=\sum_{j=1}^{J}w_j A_{pj},
 \qquad
 n_r=\sum_{j=1}^{J}w_j A_{rj}.
\]

In practice we use estimates $\widehat A$, because these probabilities are measured from finitely many generated scenes. Therefore, satisfying estimated constraints does not by itself guarantee exact population constraints.

## 4. Read the minimum-KL problem on the slide

The slide's first equation is

\[
 \min_w D_{\mathrm{KL}}(w\Vert u)
 =\min_w\sum_{j=1}^{J}w_j\log\!\left(\frac{w_j}{u_j}\right),
 \qquad u_j=\frac1J,
\]

subject to

\[
 \mathbf 1^\top w=1,\qquad w_j\ge w_{\min},
 \qquad L_e\le \widehat A_{e:}w\le U_e.
\]

Read each part as follows:

- `w` is the mixture we are choosing; `w_j` is condition `j`'s probability.
- `u` is the uniform reference distribution.
- `J` is the number of candidate conditions.
- `D_KL(w || u)` measures how different `w` is from that reference in the KL-divergence sense.
- $\mathbf 1^\top w=1$ makes the weights sum to one.
- `w_min` is a specified lower weight bound; with nonnegative weights, the vector is a probability mixture.
- `e` indexes a chosen event, such as an individual predicate or complete rule.
- $\widehat A_{e:}$ is the row of estimated probabilities for event `e` across the candidate conditions.
- `L_e` and `U_e` specify lower and upper bounds on that event's estimated mixture exposure.

The event bounds are linear in `w` because $\widehat A$ is held fixed during this optimization. The hat means the matrix is estimated, not known without error.

### What is KL divergence?

KL divergence is a measure of the difference between two probability distributions. It is not generally symmetric: in general `D_KL(w || u)` differs from `D_KL(u || w)`. Its conventional value is nonnegative for valid distributions, and it equals zero only when the distributions are equal, provided the reference gives positive probability wherever required.

The uniform reference has positive components `1/J`. The usual convention is `0 log 0 = 0` by continuity when a mixture weight is zero.

## 5. Why is minimum KL zero when uniform is feasible?

If `w = u`, then every ratio `w_j/u_j` equals 1. Since `log(1) = 0`, every term in the sum is zero:

\[
 D_{\mathrm{KL}}(u\Vert u)=\sum_j u_j\log(1)=0.
\]

Gibbs' inequality states that KL divergence is nonnegative and equals zero only when the two distributions coincide. Therefore, **if `u` satisfies every constraint**, no feasible solution can have a smaller objective, and `w = u` is the unique optimum.

A short proof intuition uses $-\log x\ge 1-x$. For positive `w_j`, put `x = u_j/w_j` and multiply by `w_j`:

\[
 w_j\log\!\left(\frac{w_j}{u_j}\right)\ge w_j-u_j.
\]

Summing gives $D_{\mathrm{KL}}(w\Vert u)\ge\sum_j(w_j-u_j)=0$, because both distributions sum to one. Equality requires `w = u`; zero weights can be handled by continuity. This proves a property of the objective, not that uniform is always feasible.

### If uniform is not feasible

If at least one constraint excludes `u`, the optimizer cannot select it. For a nonempty closed feasible set of probability vectors, the minimum is then positive. If the constraints are incompatible and the feasible set is empty, there is no feasible optimum; a solver should report infeasibility or failure rather than invent weights.

### Why this matters to the research question

If uniform is feasible, the solution is simply the reference distribution. It does not, by itself, generate separate low- and high-joint-exposure mixtures. A valid baseline can still be useful as a reference, but one must use a different objective or additional design requirements to seek a deliberate contrast.

## 6. A numerical KL example

For two conditions, let `u = (0.5, 0.5)`.

- At `w = (0.5, 0.5)`, KL is 0.
- At `w = (0.8, 0.2)`,

\[
 D_{\mathrm{KL}}(w\Vert u)
 =0.8\log(1.6)+0.2\log(0.4)
 \approx0.1927.
\]

The second mixture is further from the uniform reference in this KL sense. This is a constructed arithmetic example, not a measured project result. Also, KL is not a percentage and is not a model-accuracy score.

## 7. Why is the baseline described as convex?

A **convex set** contains the straight-line mixture between any two of its points. The nonnegative simplex together with linear lower and upper bounds on $\widehat A w$ is convex.

The function `D_KL(w || u)` is convex in `w` for a fixed positive reference `u`. Therefore, the minimum-KL baseline with these linear constraints is a convex optimization problem. With exact computation, a local optimum of a convex problem is global; numerical solver tolerances and input-estimation error still need checking.

Convexity does not mean that a feasible set is nonempty, that the estimated exposure matrix equals the true probabilities, or that the chosen objective creates the desired experimental contrast.

## 8. Read the separate max–min equation

The second formulation on the slide is

\[
 \max_{w^{(1)},\ldots,w^{(4)},z}z
 \quad\text{subject to}\quad
 g_i\!\left(w^{(1:4)};\widehat A^{(k)}\right)\ge z
 \quad\forall i,k,
\]

with

\[
 \mathbf1^\top w^{(q)}=1,\quad
 w_{\min}\le w_j^{(q)}\le w_{\max},\quad
 q=1,\ldots,4,\quad -1\le z\le1.
\]

Here:

- `w^(q)` is one of **four shared mixtures**. The same set of mixture weights is used across the selected rules, rather than defining a different model per rule.
- `q` indexes those four mixtures.
- $\widehat A^{(k)}$ is one of the supplied estimated fitting matrices; `k` indexes the matrix.
- `g_i` is a signed margin associated with a design condition. A nonnegative margin means the corresponding requirement is met under the declared sign convention.
- `z` is a lower bound on all selected margins, across the listed constraints and fitting matrices.
- The optimizer tries to make that common lower bound as large as possible.

The additional bounds ensure each mixture is a valid probability vector and prevent individual condition weights from falling below or rising above specified thresholds. The $-1\le z\le1$ bound is part of the displayed formulation; it should not be interpreted as a universal bound on every conceivable exposure metric.

### Why “max–min”?

Because `z` must be no greater than each margin, it is limited by the **smallest** margin. Maximizing `z` therefore aims to improve the weakest requirement rather than optimizing only a favorable average.

A toy example illustrates the idea. Suppose one number `w` must balance two margins:

\[
 g_1(w)=w-0.2,\qquad g_2(w)=0.7-w,\qquad 0\le w\le1.
\]

At `w = 0.45`, both margins are 0.25. Moving `w` in either direction makes one of the two margins smaller, so the best minimum in this simple example is 0.25. This is just an illustration of the max–min principle, not the project's actual four-mixture result.

## 9. Is the max–min problem also convex?

Not necessarily. The project's separate four-mixture margins include log-based exposure features and cross-factor constraints. Those forms do not automatically produce a convex feasible set. The project uses SLSQP for an exploratory local fit, so numerical success does not prove global optimality, population feasibility, statistical power or that the research gate has passed.

The goal of the four-mixture design is to obtain structured contrasts in marginal and joint-lift features across a fixed selected rule set. The estimated matrix, the rules, the thresholds, the initialization and the sampled scenes all affect the result. The information matrix and its rank also need to be checked; a positive-looking margin on a training draw is not enough by itself.

## 10. What does “47 of 72 margins certified” mean?

For one exploratory six-rule candidate, the team froze a design and evaluated it on a fresh renderer draw. Its **72 point-estimate margins** were nonnegative on that draw. A conservative simultaneous uncertainty calculation then classified:

- **47 of 72** design constraints as certified met;
- **0** as certified violated; and
- **25** as unresolved under the stated simultaneous 95% bounds.

“Unresolved” means that the available interval was not narrow enough to certify either direction. It does **not** mean that all 25 constraints failed, and it does not mean that they passed.

The uncertainty analysis uses simultaneous bounds for 72 mixture-probability quantities arising from four mixtures, six rules and three events per rule; interval arithmetic then propagates them through the design inequalities. The constraint audit also has 72 inequalities; these are related counts, not a reason to treat a single percentage as model accuracy.

The 95% statement is a family-wise coverage property under the stated independent-sampling assumptions for the frozen design and bound construction. It is not a 95% probability that “the whole research project is correct,” and 47/72 is not the fraction of images generated correctly.

The six-rule subset is exploratory. The fresh draw uses the same renderer after earlier design exploration, and other benchmark and checker requirements remain. These results do not establish that Stage 0 passed or that the joint-exposure model experiment was completed.

## 11. How does this connect to model training?

There are two distinct optimization tasks:

1. **Data-mixture optimization:** choose `w` to design which training conditions are sampled, subject to exposure requirements.
2. **Model optimization:** update neural-network parameters to reduce diffusion training loss, using AdamW in the reported pilot.

The first concerns the distribution of examples; the second concerns the model's learned parameters. The four-cell pilot reported on slide 8 did not train on the proposed four-mixture joint-exposure design.

## 12. Likely evaluator questions

**Is the KL objective always zero?** No. It is zero exactly when `w = u`. If uniform is feasible, that is the minimum. If uniform is excluded and a feasible solution exists, the minimum is positive.

**Why call the minimum-KL problem convex?** Fixed-reference KL is convex in the mixture weights, and the simplex and exposure bounds are linear constraints.

**Does minimizing KL automatically give a low-versus-high joint exposure experiment?** No. If uniform is feasible it returns uniform. A contrast requires a suitable different objective and constraints.

**Are the max–min margins the same as KL divergence?** No. The max–min objective concerns the weakest of several declared design margins over four mixtures.

**Does 47/72 mean 65% model accuracy?** No. These are uncertainty-certified design inequalities for a particular exploratory six-rule candidate, not generated-image success labels.

**Is 25 unresolved the same as 25 violations?** No. The intervals did not certify either outcome for those constraints.

## What to tell the audience

> “We formulate training-data design as an optimization problem over mixture weights. The first baseline minimizes KL divergence to a uniform mixture while satisfying linear exposure constraints. If uniform is feasible, it is the unique solution and the minimum is zero, so that objective alone does not create a low-versus-high exposure contrast. We therefore also explored a separate four-mixture max–min design that improves the weakest constraint margin. On one fresh six-rule check, 47 of 72 constraints were certified, while 25 remained unresolved; further validation is needed before training the direct intervention.”
