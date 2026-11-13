# Speaking guide

This guide provides quick presenter bullet notes keyed frame-by-frame (1–63) to `main.pdf`. For deep technical narratives, math derivations, and anticipated reviewer Q&A defenses, consult `detailed-speaker-notes.md`.

1. **Title:** Introduce the project as a study of what kinds of training examples help an image model follow combinations of instructions.
2. **Table of Contents:** Outline the four presentation acts: foundations, scene world, evaluation & optimization, and obstacles/applications.
3. **Adjacent research:** Situate work within MOSAIC (scene complexity vs imbalance), Okawa et al. (constituent abilities), GRASP (gradient conflicts), and Bradley (local scores).
4. **Proposed contributions:** Introduce the planned exposure manipulation, composition gap, $N_{50}$ thresholds, and constrained optimization. Novelty still requires further literature review.
5. **Question:** Explain the distinction between seeing ingredients separately and seeing their complete arrangement.
6. **Why care:** A model might recognize red and square independently yet struggle when asked to place objects in a particular relationship.
7. **Image model:** A conditional model receives a description and produces an image, learning patterns from training data.
8. **Pixels:** Briefly describe an image as a rectangular grid of numeric color channels; inputs and outputs are numbers.
9. **Noise:** Noise is random variation in pixel values. As corruption increases, the original image degrades into static.
10. **Diffusion: learn to reverse noise:** Training corrupts clean examples at different noise levels; the model learns to invert corruption.
11. **Reverse generation:** Synthesis starts from random noise and repeatedly applies learned denoising steps to uncover an image.
12. **Forward diffusion math:** Present the Gaussian Markov transition $q(x_t \mid x_{t-1})$ and closed-form sampling $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
13. **Reverse process & objective:** Formulate $p_\theta(x_{t-1} \mid x_t)$ and derive the simple noise estimation MSE loss $\mathcal{L}(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$.
14. **U-Net architecture:** Detail the multi-scale hierarchical U-Net with downsampling residual blocks, $16 \times 16$ bottleneck, skip connections, and upsampling.
15. **Conditioning via cross-attention:** Explain how tokenized rule predicates project to keys/values and attend to spatial feature queries at the bottleneck.
16. **Conditioning:** Clarify how conditioning vectors act as directional steering throughout reverse trajectory updates.
17. **Classifier-free guidance:** Explain training condition dropout ($p_{\text{drop}}=10\%$) and inference extrapolation $\tilde{\epsilon} = \epsilon_\emptyset + s(\epsilon_c - \epsilon_\emptyset)$.
18. **Training versus generation:** During training the clean image and added noise are known; during generation the model starts from pure noise without ground truth.
19. **Scene world:** We use a controlled synthetic 2D world so that we can count what is present and eliminate natural label noise.
20. **Vocabulary:** Present the prototype's four colors and three shapes. The final benchmark vocabulary has not been locked.
21. **Relations:** Explain that left and above compare bounding box centers across distinct objects.
22. **Predicates:** A predicate is one piece of a rule. The example combines object attributes and their relative spatial relation.
23. **Distinct objects:** The two rule variables must bind to different objects. They may share color, but one object cannot fulfill both roles.
24. **Exposure:** Define exposure as the fraction of training scenes satisfying an event, measured from ground-truth annotations.
25. **Multiple events:** A single annotated scene satisfies several simpler and more complex rules simultaneously.
26. **Hierarchy:** Contrast marginal exposure of one predicate with exposure of a pair and the complete bound rule.
27. **Counting:** Three qualifying scenes in ten gives an empirical 30% exposure. Sampling uncertainty is tracked via Wilson intervals.
28. **Co-occurrence:** Explain that individual concept frequencies may match across datasets even when joint co-occurrences differ radically.
29. **Lift:** Define lift as $\log(\text{observed joint} / \text{product reference})$. Positive values indicate synergistic co-occurrence; negative indicate suppression.
30. **Lift numbers:** Multiply 0.5 by 0.4 to obtain 0.2. If joint exposure is 0.1, the log ratio is negative. This is an illustration, not project data.
31. **Success:** Prompt the model, evaluate with automated checkers, and report checker abstentions separately from model errors.
32. **Reference:** Multiplying constituent success rates creates a simple independence reference. It is an empirical benchmark, not a physical law.
33. **Composition gap:** Compare observed rule success to the reference. A negative gap indicates difficulty coordinating familiar concepts.
34. **Zero gap:** The gap is zero when observed success equals the product reference. The proposal uses logit scale; drafts also explore log scale.
35. **Hypotheses:** State three falsifiable hypotheses: complexity scaling, joint exposure benefits, and pairwise transfer.
36. **N50:** Explain N50 as the exposure threshold where accuracy crosses 50%, measuring compositional sample efficiency.
37. **Learning curve:** Walk through the sigmoidal learning curve. Bracketing the threshold is required to avoid unsafe extrapolation.
38. **Model optimization:** Explain AdamW weight decay decoupling and Exponential Moving Average (EMA) shadow weights for stable sampling.
39. **Inference optimization:** Contrast stochastic DDPM ancestral sampling with deterministic DDIM / ODE accelerated samplers.
40. **Fair comparison:** The goal is to vary joint exposure while holding individual marginal concept exposures matched.
41. **Exposure matrix:** Map requested conditioning prompts (columns) to realized visual events (rows) via Monte Carlo annotation sampling.
42. **Exposure design optimization:** Frame mixture selection as convex optimization: $\min D_{\text{KL}}(w \| u)$ subject to marginal and joint exposure bounds.
43. **Controlled comparison:** Compare models trained on Mixture A (low joint) vs. Mixture B (high joint) to test whether co-occurrence drives learning.
44. **Checker:** An automated program detects objects and evaluates spatial rules from raw pixels without human grading fatigue.
45. **Errors and abstentions:** Explain false accepts, false rejects, and abstentions relative to gold-standard labels.
46. **Human audit:** The proposal mandates reviewing at least 300 complexity-stratified images to validate checker accuracy and per-predicate errors.
47. **Blinding:** Reviewers see the image and rule question without checker predictions; this reduces anchoring but does not guarantee unbiased or independent judgments.
48. **Pilot samples:** Examine samples from the 800,000-update exploratory run. Illustrates current prototypes, not validated benchmark accuracy.
49. **Work built:** Summarize software components built so far. Infrastructure availability is distinct from completed validation.
50. **Reviews:** The team completed small calibration reviews on a 4-condition pack. Full benchmark coverage remains outstanding.
51. **Checker obstacle:** Strict pixel and geometric heuristics cause false rejections of perceptually valid shapes.
52. **Relation obstacle:** In a 3-object scene, unconstrained spatial relations are almost always true, creating high baseline marginals.
53. **Design obstacle:** Exploratory mixture allocations suffer from multicollinearity (high VIF), requiring expanded template sets.
54. **Stage 0:** Name the three proposal gates: checker audit, exposure decorrelation, and adapted MOSAIC comparison. None is currently passed.
55. **Next phase:** After gates pass, freeze design, train controlled models, evaluate with validated checkers, and test hypotheses.
56. **Possible positive result:** Joint exposure matters after controlling basic exposure; compositional binding requires dedicated examples.
57. **Possible negative result:** Joint exposure has negligible effect; models compose concepts zero-shot from marginal exposure alone.
58. **Medical imaging implications:** Explain the clinical necessity of exposure controls: synthesizing rare co-morbidities without attribute leakage across organs.
59. **Robotics and safety implications:** Connect compositional binding to safety-critical edge case simulation in autonomous driving and robotic assembly.
60. **Importance:** Connect the inquiry to foundational understanding of generative architectures, dataset auditing, and training efficiency.
61. **Take-home:** Restate the central distinction between exposure to pieces and exposure to combinations.
62. **Questions:** Invite questions on synthetic controls, diffusion mechanics, exposure optimization, and next steps.
63. **References:** Cite the proposal, Jeong et al. (MOSAIC), Okawa et al., GRASP, Bradley, and foundational diffusion literature.
