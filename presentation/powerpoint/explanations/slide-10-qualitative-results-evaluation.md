# Slide 10 — Qualitative results and evaluation

This guide explains the tenth slide: the two selected images, the four-cell pilot, the human labels, checker coverage and what conclusions the evidence can support. A short audience script appears at the end.

## 1. What is the purpose of this slide?

Slide 10 connects pictures to the evaluation procedure. The images give the audience an intuition for the model outputs; the text explains how a larger, fixed set of generated samples was reviewed.

The key distinction is:

- **Qualitative examples** help us see what an output looks like.
- **A defined evaluation** specifies the sample set, question, labels, and treatment of uncertainty.

Neither a pleasing example nor a few difficult ones alone establishes overall model performance. The specific question and the complete set of labels matter.

## 2. What do the two pictures on the left mean?

The slide shows selected samples labeled **0003: yes** and **0001: unsure**. These labels refer to the review question about whether the requested-color circle is present. The “yes” example illustrates an output judged to satisfy that particular question; “unsure” illustrates that an image can be ambiguous to a reviewer.

The pictures are **selected examples**. They are not a random gallery, a paired before-and-after comparison, pixel-level reconstructions of a particular target, or an estimate of average image quality. Do not infer a percentage of successes by counting the two images on the slide.

A reviewer is not automatically asked whether every possible property of the picture is perfect. In this pilot, the primary question is the presence of the requested-color circle. Other questions, such as object count, were secondary and must be described separately.

## 3. What models generated the reviewed images?

The pilot crosses two factors:

| Factor | Levels |
|---|---|
| Object count in training scenes | 2 objects or 3 objects |
| Target-color balance in training | 50:50 or 90:10 |

That gives four conditions. In the first reported block, one PilotUNet model was trained per condition on 200 fixed training scenes, for 500,000 optimizer updates per model, with batch size 4. The target request is red or blue. Slide 8 explains the optimization and EMA sampling.

The generated-image condition supplies the **target color**. The expected number of objects was a secondary evaluation question, not an additional prompt to the model in the primary evaluation.

This is a data-factor comparison. It is not an ablation of the attention layer, skip connections or another component of the network. The pilot also does not manipulate the planned joint-exposure mixtures from slide 4.

## 4. How were the 64 images obtained?

The fixed protocol selected the four 500,000-update checkpoints, EMA weights, guidance 1, 1,000 reverse steps, two target colors and eight paired generation seeds.

For each cell:

\[
 2\;\text{colors}\times8\;\text{noise seeds}
 =16\;\text{generated images}.
\]

Across the four cells:

\[
 4\;\text{cells}\times16\;\text{images per cell}=64\;\text{images}.
\]

Using matched generation seeds helps make the sampling protocol systematic across cells. It does **not** create additional independently trained models. A noise seed controls generation randomness, while a training seed influences model initialization and training randomness.

The 64 images are generated evaluation samples. They are not a conventional held-out set of annotated photographs, and they do not mean the models trained on 64 images. Each pilot training condition contained 200 fixed scenes.

## 5. What exactly did the human review ask?

The primary question was whether a **circle of the requested color** was present in the output. Depending on what the image showed, reviewers could answer yes, no or uncertain. Original uncertain wordings such as “maybe” and “not sure” were retained in the source records and treated as unresolved for the analysis.

- **Yes:** the requested-color circle is judged present.
- **No:** it is judged absent under the review question.
- **Unresolved:** the available review does not give a defined yes/no decision under the scoring policy.

An uncertain answer is not automatically a negative. Likewise, a checker abstention is not a human “no.” Keeping these states separate preserves information about ambiguity and coverage.

The review pack had the same image set for the reviewers. Four returned files for this first block were byte-identical. Those files alone do not establish reviewer identity or independent authorship; identical files do not by themselves prove how the reviews were conducted. The team separately confirmed independent reviews for the second block, whose artifacts also do not independently establish reviewer identity.

## 6. Understand the 59 / 3 / 2 labels

The recorded first-block target-circle labels are:

| Training cell | Yes | No | Unresolved | Total |
|---|---:|---:|---:|---:|
| 2 objects, balanced | 16 | 0 | 0 | 16 |
| 2 objects, imbalanced | 15 | 0 | 1 | 16 |
| 3 objects, balanced | 14 | 2 | 0 | 16 |
| 3 objects, imbalanced | 14 | 1 | 1 | 16 |
| **Total** | **59** | **3** | **2** | **64** |

The 59 yes labels are about `59/64 = 92.19%` of this fixed image set. This is a descriptive count for this specific question and protocol. It is not a proven population success rate, a cross-dataset accuracy estimate, or evidence that complex relational prompts are solved.

If the two unresolved images were hypothetically resolved as no, the number of yes outcomes would remain 59; if both were resolved as yes, it would become 61. That gives a **finite-image assignment range** from `59/64` to `61/64`. It is a sensitivity calculation about these two labels, not a statistical confidence interval.

## 7. How do the pilot contrasts relate to this slide?

Slide 9 displays the quantitative contrast ranges. Let `s_LB`, `s_LI`, `s_HB` and `s_HI` be the four cell success rates, with L/H for the low/high object-count conditions and B/I for balanced/imbalanced color conditions. The prespecified descriptive contrasts are

\[
 D_C=\frac{s_{LB}+s_{LI}-s_{HB}-s_{HI}}{2},
 \qquad
 D_I=\frac{s_{LB}+s_{HB}-s_{LI}-s_{HI}}{2}.
\]

`D_C` compares the equally weighted two-object and three-object training conditions. `D_I` compares balanced and imbalanced conditions. They are defined as drops, so a positive value is in the direction specified by the subtraction.

Assigning the two unresolved labels in all allowed binary ways yields:

- **Complexity-drop bounds:** 0.0625 to 0.125, or 6.25 to 12.5 percentage points.
- **Imbalance-drop bounds:** −0.03125 to 0.03125, or −3.125 to 3.125 percentage points.

For example, the minimum complexity drop assigns the low/imbalanced unresolved label to no and the high/imbalanced unresolved label to yes: `(16 + 15 - 14 - 15) / 32 = 2/32 = 0.0625`. The maximum assigns them in the opposite direction: `(16 + 16 - 14 - 14) / 32 = 4/32 = 0.125`.

These are **exact finite-image bounds for uncertainty in those labels**, not confidence intervals over future images, training seeds or datasets. The later documented training-seed block shares the dataset and reported a descriptive complexity drop of zero. Neither block is a direct joint-exposure intervention.

## 8. What is the checker?

A checker is an automated procedure that attempts to determine whether a generated image satisfies a specified visual question. The project has a standard checker and an experimental variable-count checker. The slide discusses the experimental checker's behavior on the first fixed 64-image set.

A checker can return a positive or negative decision, or **abstain** if it cannot make a reliable decision under its rules. Abstention is a property of the evaluator; it is not a property of the model's intended training label.

### What did it do on these 64 images?

- It **answered 46** images: `46/64 = 71.875%` coverage.
- It **abstained on 18** images: `18/64 = 28.125%` abstention.
- All 46 answered cases were positive and agreed with the recorded yes labels.
- It abstained on all three recorded human-negative images and both unresolved images.

The remaining abstentions therefore include 13 images labeled yes by the recorded reviews: `59 - 46 = 13`. Together, `13 yes + 3 no + 2 unresolved = 18` abstentions.

### Why does this matter?

Agreement among the 46 answered images is **selected-sample agreement**. Because the checker answered only on cases labeled yes and abstained on every human-negative case in this set, these observations do not measure its false-accept rate on negative images. An always-positive selective checker could also agree on that particular answered subset.

The 46 agreed decisions are useful information about that subset, but “100% accurate overall” or “false-accept rate is zero” would not be justified. A complete validation should report a confusion matrix, coverage and abstention behavior, including challenging positive and negative cases.

## 9. What are false accept, false reject and abstention?

Consider a defined target-circle question and a trusted reference label:

| Checker output | Reference | Interpretation |
|---|---|---|
| Yes | Yes | True positive |
| Yes | No | False accept / false positive |
| No | No | True negative |
| No | Yes | False reject / false negative |
| Abstain | Either | No decision; report separately |

An evaluator can have high agreement on the cases it chooses to answer and still abstain frequently or disproportionately on difficult examples. That is why **coverage and errors must be considered together**.

The reference labels themselves can be uncertain or disputed, so a rigorous audit should also define how disagreements, unresolved cases and independent review are handled. It should not silently force an ambiguous case into yes or no.

## 10. Why is the direct joint-exposure question still open?

The project's main research question concerns whether exposure to a complete combination matters after accounting for its constituent exposures. The four-cell pilot instead changes object count and color balance in its training conditions. It does not hold predicate marginals fixed while deliberately varying full-rule joint exposure, and the pilot model's prompt is a requested color rather than a complex multi-predicate rule.

Moreover, target role is confounded with the pilot's object attributes, the datasets have 200 fixed scenes each, and geometries are not exact matched pairs. The evidence can support a description of the implemented pilot and reviewed outputs, but not a claim that joint exposure has been proven to improve compositional generalization.

## 11. What would make evaluation stronger?

A future evaluation should define the target rules and metrics in advance; include positives, hard negatives, wrong bindings, missing objects and relational failures; and assess a validated checker against independent blinded human judgments on representative generated images. It should report all predictions, abstentions and disagreements.

To test the exposure question, it should also use validated training mixtures and independent dataset and model-training seeds. More generated noise seeds from a single model increase the number of sampled outputs but do not create new independent trained-model replicates.

## 12. Likely evaluator questions

**Are the two pictures representative of the entire model?** They are selected illustrations. The fixed 64-image review gives a wider but still small descriptive view.

**Is “unsure” the same as “no”?** No. It remains unresolved unless a prespecified adjudication procedure resolves it.

**Does 46/64 mean the checker got 46 images right and 18 wrong?** No. It answered 46 and abstained on 18. The answered cases agreed with recorded labels; abstention is not automatically an error.

**Does agreement on all 46 prove zero false accepts?** No. It abstained on every human-negative case in this set, so these observations cannot estimate its false-accept rate.

**Did the four returned files prove four independent reviewers?** The files were byte-identical and do not by themselves verify reviewer identity or independence. That is a provenance limitation, not a conclusion about what any person did.

**Is this a model-component ablation?** No. The experiment compares dataset factors. It has not established the effect of removing an attention block or another U-Net component.

## What to tell the audience

> “These are two selected examples from our generated-image review: one was labeled yes for the requested-color circle and one was uncertain. For the full fixed pilot we generated 64 images, 16 from each of four trained conditions; the recorded labels were 59 yes, three no and two unresolved. Our experimental checker answered 46 and abstained on 18. The answered cases agreed with the recorded labels, but the checker abstained on all three negative examples, so the pilot also tells us what must be validated before relying on automated evaluation.”
