# Slide 5 — Optimization, explained with one small example

You do not need to know optimization before reading this. We will use the **same four kinds of shapes** all the way through, and build the equations one piece at a time.

The four-shape example is invented to teach the idea. It is **not** the actual list of 57 project conditions, and its numbers are **not** experimental results.

## 1. The story: making trays of shapes

Imagine we are making pictures for a little image model to learn from. Every picture in our pretend world contains **exactly one object**. There are four kinds of pictures:

| Box | What the picture contains |
|---|---|
| 1 | A red circle |
| 2 | A red square |
| 3 | A blue circle |
| 4 | A blue square |

We will make a tray of **100 pictures**. We get to decide how many pictures come from each box.

Our big question is:

> Can we show the model the same number of red objects and the same number of circles, but change how many **red circles** it sees?

A red circle is special here because it has **both** things together: it is red, and it is a circle.

---

## 2. First learn three little words

### Condition

A **condition** is one of the boxes we can choose from. In our pretend example there are four conditions: red circle, red square, blue circle, and blue square.

In the actual project, a condition is a candidate scene-generating condition. It can produce scenes containing several objects, and the scenes can satisfy other rules in addition to the condition used to generate them. The real conditions are not just these four boxes.

### Weight

A **weight** says how much of our tray comes from a box.

For example, if 25 of the 100 pictures come from box 1, its weight is

\[
 w_1=\frac{25}{100}=0.25.
\]

We can also use weights as probabilities when sampling: a weight of 0.25 means a 25% chance of choosing that condition on a draw. If we sample randomly, the realized counts need not be exactly 25; our trays below use exact counts to keep the arithmetic easy.

### Mixture

A **mixture** is the full list of weights for all the boxes.

For a tray with 25 pictures from each box, the mixture is

\[
 w=(0.25,\ 0.25,\ 0.25,\ 0.25).
\]

The weights add to 1, just as 100 pictures out of 100 make the whole tray. No weight can be negative.

---

## 3. Let's make our first tray: everything equal

| Box | Picture | Number in tray | Weight |
|---|---|---:|---:|
| 1 | Red circle | 25 | 0.25 |
| 2 | Red square | 25 | 0.25 |
| 3 | Blue circle | 25 | 0.25 |
| 4 | Blue square | 25 | 0.25 |
| | **Total** | **100** | **1.00** |

Now count:

- **Red pictures:** 25 red circles + 25 red squares = **50**.
- **Circle pictures:** 25 red circles + 25 blue circles = **50**.
- **Red-circle pictures:** **25**.

Red is one ingredient. Circle is another ingredient. A red circle is the two ingredients **together**.

### What does “marginal” mean?

A **marginal exposure** counts one ingredient at a time.

- Red marginal exposure = 50 out of 100 = **0.50**.
- Circle marginal exposure = 50 out of 100 = **0.50**.

### What does “joint” mean?

A **joint exposure** counts a specified combination.

- Red-circle joint exposure = 25 out of 100 = **0.25**.

In this toy world there is only one object in each picture, so “red and circle” clearly refers to the same object. In the real multi-object scenes, we must explicitly check that the red object **itself** is a circle. A red square next to a blue circle has a red object and a circle, but it has no red circle.

---

\newpage

## 4. Make a second tray: fewer red circles

Now we change the counts:

| Box | Picture | Number in tray | Weight |
|---|---|---:|---:|
| 1 | Red circle | 10 | 0.10 |
| 2 | Red square | 40 | 0.40 |
| 3 | Blue circle | 40 | 0.40 |
| 4 | Blue square | 10 | 0.10 |
| | **Total** | **100** | **1.00** |

Count again:

- **Red pictures:** 10 + 40 = **50**.
- **Circle pictures:** 10 + 40 = **50**.
- **Red-circle pictures:** **10**.

Red and circle still each appear in half the pictures. But red and circle appear **together** in only 10 pictures instead of 25.

We call this our **low-joint toy tray**.

---

## 5. Make a third tray: more red circles

Now use these counts:

| Box | Picture | Number in tray | Weight |
|---|---|---:|---:|
| 1 | Red circle | 40 | 0.40 |
| 2 | Red square | 10 | 0.10 |
| 3 | Blue circle | 10 | 0.10 |
| 4 | Blue square | 40 | 0.40 |
| | **Total** | **100** | **1.00** |

Count one more time:

- **Red pictures:** 40 + 10 = **50**.
- **Circle pictures:** 40 + 10 = **50**.
- **Red-circle pictures:** **40**.

We call this our **high-joint toy tray**.

| Tray | Red pictures | Circle pictures | Red circles |
|---|---:|---:|---:|
| Equal | 50 | 50 | 25 |
| Low joint | 50 | 50 | 10 |
| High joint | 50 | 50 | 40 |

This table is the big idea. All three trays have the same individual counts for red and circle. But the complete **red-circle** combination appears 10, 25 or 40 times.

The numbers are a teaching example. We have **not** shown that a model trained on these trays performs better or worse. That would require an actual controlled model experiment.

---

## 6. Where does optimization come in?

Imagine there are not four boxes, but many boxes. We may have lots of requirements:

- The weights must add to 1.
- No weight can be negative.
- Red exposure must be within a chosen range.
- Circle exposure must be within a chosen range.
- Some complete rules must have specified exposure properties.

Trying to guess every weight by hand gets hard. **Optimization** is a way to ask a computer to choose numbers that follow our rules and do well on a chosen goal.

There are always three questions:

1. **What can the computer choose?** Here: mixture weights.
2. **What rules must the choices follow?** Here: valid probabilities and exposure constraints.
3. **What is the goal?** On this slide, we compare two different goals: minimum KL and max–min margin.

The rules are called **constraints**. A choice that obeys all the rules is called **feasible**. If no choice obeys them all, the problem is infeasible. That is different from a solver simply failing to find an answer.

---

## 7. What does the letter “u” mean?

The letter `u` means the **uniform mixture**: every box gets the same weight.

In our four-box story,

\[
 u=(0.25,\ 0.25,\ 0.25,\ 0.25).
\]

That is the equal tray from section 3. It is only a **reference**: the optimizer compares its chosen mixture against this equal mixture.

If there are `J` boxes, the uniform weight of each box is

\[
 u_j=\frac{1}{J}.
\]

For four boxes, each weight is 1/4. The project's large measurement used 57 candidate conditions; a uniform reference over all 57 would have 1/57 per condition. Our four boxes are only a made-up teaching example.

### Important: which “four”?

There are three different uses of the number four in this project:

1. The **four pretend boxes** in this guide are a simple example.
2. The **four mixtures** in the exploratory max–min design are four complete weight lists; each can contain weights over many real candidate conditions.
3. The **four pilot conditions** on slide 8 cross object count with color balance. They are a separate diffusion feasibility pilot.

They are not the same four things.

---

## 8. Read the first line on the slide: “minimize KL”

The first equation says

\[
 \min_w D_{\mathrm{KL}}(w\Vert u)
 =\min_w\sum_{j=1}^{J}w_j\log\!\left(\frac{w_j}{u_j}\right),
 \qquad u_j=\frac{1}{J}.
\]

Let's translate it into ordinary words:

> “Among the mixtures that obey our rules, choose the one that is closest to the equal mixture according to KL divergence.”

Here is every piece:

| Symbol | Simple meaning in our story |
|---|---|
| `w` | The tray mixture we ask the computer to choose. |
| `w_j` | The weight of box `j` in that mixture. |
| `u` | The equal mixture: the same weight for every box. |
| `u_j` | The equal reference weight for box `j`. |
| `J` | How many boxes or candidate conditions there are. |
| `j` | Which box we are talking about. |
| KL divergence | A mathematical score comparing the chosen weights with the reference weights. |
| `min` | Make that score as small as possible. |

KL divergence is a comparison score, not a count of wrong images and not model accuracy. It is usually written as a sum of terms involving logarithms. You do not need to calculate logarithms in your head to understand the purpose: **the baseline looks for a feasible mixture close to the equal reference**.

The order of KL matters in general: KL from `w` to `u` is not necessarily the same as KL from `u` to `w`. Here the equation specifically uses the direction from the chosen mixture to the reference.

---

## 9. Why does the answer become zero?

Look at the equal tray again:

\[
 w=(0.25,0.25,0.25,0.25),\qquad
 u=(0.25,0.25,0.25,0.25).
\]

They are exactly the same tray. For every box, the ratio `w_j/u_j` is 1. The logarithm of 1 is 0. Every term in the sum is therefore 0:

\[
 D_{\mathrm{KL}}(u\Vert u)=0.
\]

KL divergence cannot be negative for valid probability distributions, and it is zero only when the two distributions are equal. This is known as Gibbs' inequality. Therefore:

> If the equal mixture obeys every constraint, it is the unique minimum-KL solution, and the minimum value is zero.

In our pretend story the equal tray has 50 red pictures and 50 circle pictures. So if our requirements are exactly those two counts, the equal tray is allowed. Minimum KL chooses it: 25 red circles.

### Is minimum KL always zero?

**No.** It is zero only if the equal mixture is allowed.

For example, if we add a requirement that at least 40 pictures must be red circles, the equal tray has only 25 and is no longer allowed. The high-joint tray has 40 and can satisfy that toy requirement. If a feasible tray exists but the equal tray is excluded, the minimum KL is positive.

If the constraints are contradictory and no tray is allowed, there is no feasible solution at all.

### Why is this a problem for our question?

We want to compare mixtures with different full-rule exposure. But **“stay as close as possible to equal”** is not the same request as **“create two meaningfully different red-circle exposures.”** When the equal mixture is allowed, minimum KL gives us the equal tray. It does not by itself produce both our low-joint tray and our high-joint tray.

That makes minimum KL a useful **baseline**, but not a complete answer to the exposure-design question.

---

## 10. What do the constraint symbols mean?

The slide writes the rules underneath the first equation:

\[
 \mathbf 1^\top w=1,\qquad
 w_j\ge w_{\min},\qquad
 L_e\le\widehat A_{e:}w\le U_e.
\]

Read them in this order.

### Rule 1: all weights add up to one

\[
 \mathbf 1^\top w=1.
\]

For the low-joint tray: `0.10 + 0.40 + 0.40 + 0.10 = 1.00`. We have accounted for the whole tray.

### Rule 2: each box must have enough weight

\[
 w_j\ge w_{\min}.
\]

`w_min` is the smallest allowed weight chosen for the design. If the minimum were 0.05, each box would need at least 5% of the tray. All three of our toy trays satisfy that particular lower bound.

The actual choice of minimum is a design setting; 0.05 is only a teaching example. The minimum-KL equation on the slide specifies a lower bound. The separate max–min formulation also displays an upper bound.

### Rule 3: the measured exposure must stay between limits

\[
 L_e\le\widehat A_{e:}w\le U_e.
\]

This line looks scary, but it means:

> “For event `e`, the estimated exposure from the chosen tray must be no smaller than the lower limit and no larger than the upper limit.”

- `e` is an event we care about, such as “red” or “circle.”
- `L_e` is its lower allowed exposure.
- `U_e` is its upper allowed exposure.
- The row with a hat, `Â_e:`, records estimated event frequencies for that event across boxes.
- Multiplying that row by `w` gives the estimated exposure of event `e` in the mixture.

### Let's calculate one row, using our same four boxes

For the event “red,” the four boxes have these toy values:

| Box | Red circle | Red square | Blue circle | Blue square |
|---|---:|---:|---:|---:|
| Is its one object red? | 1 | 1 | 0 | 0 |

For the low-joint weights `w = (0.10, 0.40, 0.40, 0.10)`, the red exposure is

\[
 (1)(0.10)+(1)(0.40)+(0)(0.40)+(0)(0.10)=0.50.
\]

For “circle,” the row is `(1, 0, 1, 0)`, giving `0.10 + 0.40 = 0.50`. For the bound “red circle” event, the row is `(1, 0, 0, 0)`, giving 0.10.

In our toy world the entries are exactly 0 or 1 because each box always produces exactly one specified object. In the real project, a scene-generating condition can produce different scenes, so entries can be probabilities such as 0.2 or 0.7. The hat on `Â` reminds us that those probabilities are **estimated from sampled scenes**.

### What does “convex” mean in simple terms?

For the minimum-KL problem, the allowed region is built from linear constraints, and the KL objective is convex in the chosen mixture. Intuitively, convex problems have a well-behaved bowl-like structure: when a valid minimum is found, there is no separate lower hidden valley elsewhere in the feasible region. Numerical tolerances and sampling error still matter.

The point is **not** that the data or model are automatically correct. It only describes the mathematical shape of this particular optimization problem.

---

## 11. Now the second problem: “max–min”

The second equation asks a different question:

> “Can we make the weakest required design margin as strong as possible?”

A **margin** is a signed amount of room relative to a requirement. In the project formulation, each `g_i` is a particular design margin. A nonnegative value means the corresponding requirement is met under its stated sign convention; a negative value means the point estimate falls short of that requirement.

The displayed equation is

\[
 \max_{w^{(1)},\ldots,w^{(4)},z} z
 \quad\text{subject to}\quad
 g_i\!\left(w^{(1:4)};\widehat A^{(k)}\right)\ge z
 \quad\forall i,k.
\]

Do not read it all at once. Break it apart:

1. `w^(1), …, w^(4)` are **four complete mixtures**: four different weight lists.
2. `g_i` is the margin for requirement `i`.
3. `Â^(k)` is one of the estimated matrices used for fitting; `k` labels the matrix.
4. `z` is a number that must be **no larger than any required margin**.
5. The computer tries to make `z` as large as possible.

Because `z` must fit underneath every margin, the smallest margin holds it down. Raising `z` tries to raise the **weakest** margin. That is why this is called **max–min**: maximize the minimum.

The four-mixture investigation considers low/high levels of marginal-related and lift-related features. For the selected rules, it uses a mean of log singleton probabilities as a marginal feature and a log joint-to-product quantity as lift. It is an exploratory design investigation, not four separately trained models in the completed joint-exposure experiment.

---

## 12. A tiny max–min example with our red circles

This is a **teaching example of margins**, not the full four-mixture problem.

Let `a` be the fraction of our pretend tray that is red circles. Suppose we want `a` to be between 0.10 and 0.40:

\[
 0.10\le a\le0.40.
\]

There are two ways to fall short: go below 0.10 or above 0.40. Define two margins:

\[
 g_1=a-0.10,\qquad g_2=0.40-a.
\]

- `g_1` tells us the room above the lower limit.
- `g_2` tells us the room below the upper limit.

Let's try our three trays:

| Tray | Red-circle fraction `a` | Lower margin | Upper margin | Smaller margin |
|---|---:|---:|---:|---:|
| Low joint | 0.10 | 0.00 | 0.30 | 0.00 |
| Equal | 0.25 | 0.15 | 0.15 | 0.15 |
| High joint | 0.40 | 0.30 | 0.00 | 0.00 |

For this **toy problem only**, the best smallest margin is 0.15 at `a = 0.25`. The two sides are balanced. Moving toward either end makes one of the margins smaller.

This example shows how “protect the weakest margin” works. It does **not** show the project's actual four-mixture design or automatically create a low-versus-high comparison. The real design's margins encode multiple desired factor and cross-factor contrasts across rules and fitting matrices, rather than only these two toy bounds.

---

## 13. What are the other rules under the max–min equation?

The slide also states

\[
 \mathbf1^\top w^{(q)}=1,\quad
 w_{\min}\le w_j^{(q)}\le w_{\max},\quad
 q=1,\ldots,4,\quad -1\le z\le1.
\]

In plain language:

- Each of the four mixtures must add to 1.
- Each condition's weight in each mixture must stay between its chosen lower and upper limits.
- `q` tells us which of the four mixtures we mean.
- The displayed formulation bounds `z` between −1 and 1.

The actual exploratory candidate used four shared weight lists over the selected candidate conditions. Its fitted margins depend on estimated exposure matrices and multiple selected rules. It was fit using SLSQP, a numerical optimizer; the logged success is a **local numerical result**, not a proof that the global best design was found.

Unlike the first problem, this log-based max–min formulation is **not asserted to be convex**. The two problems have different goals and different mathematical properties.

---

## 14. What does “47 of 72 certified; 25 unresolved” mean?

This line is about a particular exploratory **six-rule design check**. It is not the number of images the diffusion model got right.

On one fresh renderer draw, all 72 design margins had nonnegative **point estimates**. But those estimates are uncertain because they come from a finite sample of generated scenes. A conservative simultaneous-bound analysis classified the 72 constraints as follows:

- **47 certified met:** their bounds supported meeting the corresponding requirements under the analysis assumptions.
- **0 certified violated:** none were established as violations by that bound procedure.
- **25 unresolved:** the bounds did not certify either met or violated.

“Unresolved” does **not** mean failed. It also does **not** mean passed. It means that this check did not give a definite classification for those constraints.

The simultaneous 95% statement is a property of the stated bound procedure under its sampling assumptions for the frozen design. It is not “95% accuracy” and it is not a guarantee that the whole project is correct. The six selected rules are an exploratory subset; other benchmark and checker requirements also remain.

---

## 15. One more distinction: choosing data versus training the model

We have talked about two different kinds of optimization:

### Choosing the data mixture

We choose **weights `w`**: how often to draw from each scene-generating condition. This is the topic of slide 5.

### Training the diffusion model

We update the neural network's **parameters** to reduce its denoising loss. The reported pilot uses AdamW. This is explained on slide 8.

The word “weight” can refer to both mixture weights and neural-network weights, but they are different things. One controls the training distribution; the other is learned inside the model.

---

## 16. Quick answers for questions you may get

**What is a mixture?** A list of probabilities for choosing the different scene-generating conditions.

**What is `u`?** The equal-weight reference mixture. In our toy story it is `(0.25, 0.25, 0.25, 0.25)`.

**Does `u` have four values in the real project?** Not necessarily. If the reference is over 57 candidate conditions, it has 57 values, each 1/57.

**Is the KL answer always zero?** No. It is zero when the chosen mixture equals `u`. If `u` is feasible, it is the unique minimum. If `u` is excluded but a feasible solution exists, the minimum is positive.

**Why not stop with minimum KL?** If equal is allowed, minimum KL returns the equal reference. It does not by itself create the contrasting exposure mixtures needed for the research question.

**What does max–min mean?** Improve the smallest of several required margins, so one weak requirement is not hidden by stronger ones.

**Are 47 of 72 margins the model's accuracy?** No. They are certified design constraints in an exploratory six-rule check.

**Are the 25 unresolved constraints failures?** No. The uncertainty bounds did not certify them as met or violated.

**Have we answered whether red-circle exposure improves a diffusion model?** Not yet. Our three trays teach the idea; the direct controlled model experiment remains to be done.

## What to tell the audience

> “Imagine four boxes of pictures: red circles, red squares, blue circles and blue squares. We can make two training mixtures with the same number of red pictures and the same number of circles, but different numbers of red circles. To choose such mixtures, we use optimization over the probabilities of selecting each condition. Our minimum-KL baseline chooses a feasible mixture close to equal; if equal is allowed, the answer is exactly equal and its KL value is zero. We also explore a separate max–min design that tries to strengthen the weakest required margin. That design is still being validated before the direct model experiment.”
