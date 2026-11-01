# Human checklist for Stage 0

**Status: not passed.** Only the three proposal go/no-go criteria below determine
whether Stage 0 passes. Everything else is enabling work. This checklist is
derived from `Which Exposure Matters Research Proposal (Rev. 4).pdf`,
Sections 5, 6, and 12 (pp. 4-5, 5, and 8-9), and the current repository docs.
It does not add sign-off requirements beyond the proposal's criteria.

## Proposal Stage 0 go/no-go criteria

- [ ] **Checker error ≤ 2% on real renders.** Evidence is a human-audited,
  complexity-stratified sample of at least 300 generated images from a pilot
  model, with false-accept and false-reject rates reported per predicate and per
  rule (Section 5, p. 5; Section 12, p. 9).
- [ ] **Lift is decorrelated from marginal exposure:** absolute correlation
  `|r| ≤ 0.3` and `VIF < 2` (Section 6, p. 5; Section 12, p. 9).
- [ ] **The small adapted MOSAIC replication matches MOSAIC's direction**
  (Section 12, pp. 8-9).

No criterion is currently claimed met. The current evidence is
instrument-development work, not a gate result.

## Start here: your decisions and access needs

These are decisions or access that need your involvement. We can draft options
and continue exploratory implementation while the final choices are pending.
"Blocks" names the confirmatory work that depends on each choice.

1. [ ] **Decide the final benchmark vocabulary.** Confirm whether texture is
   legible at 64×64 and whether to keep it, drop it, or move to 96×96; confirm
   the third relation after the pilot. The proposal specifies 3 shapes × 4
   colours, `left_of`, `above`, plus one further relation after the pilot
   (Section 5, p. 4). **You only.** Blocks template finalization, checker
   coverage, and the final A-matrix. We can prepare examples and a recommendation.
2. [ ] **Decide the final template set.** Treat the current 57-template set as
   provisional. Decide whether it is final, needs further counterbalancing, or
   needs changes, and define the held-out leave-out set for H3. The proposal
   calls for about 54 training templates plus a held-out set (Section 5, p. 4).
   **You only.** Blocks the A-matrix, decorrelation analysis, and leave-out
   probes. See [templates](templates.md).
3. [ ] **Decide the A-matrix Monte Carlo size.** The proposal estimates each
   entry once with about `10^5` scenes per template (Section 6, p. 5). The
   current smoke test uses 100 scenes per template. **You only.** Blocks
   credible matrix estimates and the decorrelation gate.
4. [ ] **Decide how uncertainty, breadth, and zeroes enter the decorrelation
   analysis.** Freeze the analysis table and uncertainty method before treating
   any result as the gate evaluation. **You only.** Blocks the decorrelation
   gate. See [design diagnostics](design-diagnostics.md).
5. [x] **SSH access confirmed for the pilot machine.** A user-authorized,
   read-only GPU preflight succeeded. The public repository is cloned on the
   laptop, locked sync and 123 Windows tests passed, and a short CUDA pilot ran.
   Its sample was noise, so the audit image pool still needs to be generated.
   At least 300 generated images must be human-audited in a
   complexity-stratified sample. Blocks the human audit
   and checker gate. See [pilot training](pilot-training.md) and
   [human audit workflow](human-audit.md).
6. [ ] **Confirm the reviewer pool and a neutral delivery method.** Four team
   members are reported as potential reviewers available later. Confirm their
   independence, participation, and blinding; reviewers must judge images
   without seeing checker predictions or gold labels. Actual reviews remain
   outstanding. **Human-only.** Blocks the checker gate.
7. [ ] **Confirm available GPU time and budget for project runs.** The laptop
   preflight reported Windows 11 AMD64, Python 3.13.6, an NVIDIA GeForce RTX
   3050 6GB Laptop GPU (6144 MiB, driver 581.83), Git 2.48.1, and pip 25.2. Its
   isolated environment is now complete with `uv` 0.13.6, PyTorch 2.9.1+cu128
   with CUDA 12.8, NumPy 2.5.4, Pillow 13.0.0, and SciPy 1.18.1;
   `torch.cuda.is_available()` is true, and a generic CUDA smoke completed.
   A 100-step project run fit on the GPU but its output was noise. The public
   repository is cloned into the laptop's `source` subdirectory, with the
   isolated `.venv` in its parent; locked sync and 123 Windows tests passed.
   Confirm the available GPU time and set a budget for longer project runs. See
   [research and hardware plan](research-and-hardware-plan.md) and
   [pilot training](pilot-training.md).
8. [ ] **Review the proposed MOSAIC adaptation and arrange any required access.**
   We can study the paper and implementation and propose the contrast, metrics,
   datasets, and classifier inputs for your review. Blocks the MOSAIC gate. See
   [MOSAIC notes](mosaic-replication.md).

## What Codex can do once unblocked

These are implementation or analysis tasks, not human judgments or final
decisions.

- [ ] Finalize the template set, validators, manifest, and held-out leave-out
  set after the vocabulary and template decisions.
- [ ] Run the A-matrix at the agreed sample size, propagate the agreed
  uncertainty treatment, sample designs, and produce the registered
  decorrelation analysis.
- [x] Prepare separate predicate questions and score independent judgments by
  predicate event using the existing audit workflow. Actual judgments and
  generated-image results remain outstanding.
- [x] In the cloned `source` directory, set the project environment with
  `$env:UV_PROJECT_ENVIRONMENT = "..\.venv"`, run
  `..\.venv\Scripts\python.exe -m uv sync --locked --inexact --extra train`,
  then run the exploratory pilot and record its provenance. Locked sync,
  Windows tests, and a 100-step CUDA run completed. The sample was noise;
  further model training and evaluation are needed. See [pilot training](pilot-training.md).
- [ ] Improve checker rejection handling under brightening and blur, rerun
  clean and perturbation audits, and evaluate the checker against the supplied
  generated images.
- [ ] Prepare and run the adapted MOSAIC experiment after access and contrast
  decisions; record seeds, configurations, code versions, results, uncertainty,
  and failed runs.
- [ ] Connect the Gaussian-mixture denoiser diagnostic to the eventual model and
  loss normalization. This is Stage 0 instrument work, not a go/no-go threshold
  (Section 12, p. 8).
- [ ] Draft evidence reports and the gate summary. Codex cannot supply human
  labels or make the final go/no-go call.

## Human image review

- [ ] Independent humans judge whether each generated image satisfies its
  requested rule. Four team members are reported as potential reviewers
  available later; independence, participation, and actual judgments remain to
  be confirmed. **Human-only.**
- [ ] Codex can prepare the blinded manifest, run checker predictions, validate
  the review CSV, score the judgments, and tabulate false accepts and false
  rejects.
- [ ] Report results per predicate and per rule, with complexity strata and
  rejection counts. The workflow now creates separate predicate questions and
  groups their scores by predicate event. A whole-rule judgment cannot supply
  the separate predicate judgments. Predicate-report strata describe singleton
  questions; use the whole-rule report for original complexity strata.

The audit depends on the pilot model and the generated image pool. Clean-render
counts alone do not establish the `≤ 2%` target on real renders. See
[checker](checker.md), [checker perturbations](checker-perturb.md), and
[human audit workflow](human-audit.md).

## GPU and MOSAIC work

- [ ] The proposal's working assumption is about 3 GPU-hours per run on one
  modern GPU for roughly 60 runs, plus about 50 GPU-hours for diagnostics, or
  **200-250 GPU-hours**. This is an initial planning figure, not a measured
  result. Replace it with pilot measurements and set a hard cap after the pilot
  (Section 12, p. 9). **Human decision.**
- [x] A user-authorized, read-only SSH GPU preflight succeeded. It reported
  Windows 11 AMD64, Python 3.13.6, an NVIDIA GeForce RTX 3050 6GB Laptop GPU
  (6144 MiB, driver 581.83), Git 2.48.1, and pip 25.2. The isolated environment
  is now complete with `uv` 0.13.6, PyTorch 2.9.1+cu128 with CUDA 12.8, NumPy
  2.5.4, Pillow 13.0.0, and SciPy 1.18.1; `torch.cuda.is_available()` is true.
  A generic two-layer 64×64 synthetic convolution model completed two warmup and
  three timed CUDA forward/backward/AdamW steps with finite loss, 0.0695 seconds
  measured time, 1,804,288 bytes peak allocated, and 4,194,304 bytes peak
  reserved. This is a generic CUDA smoke, not project pilot fit, learning
  evidence, Stage 0 passage, or a study. A 100-step project run also completed
  on CUDA; its sample was noise. The public repository is cloned into `source`,
  with the isolated `.venv` in its parent; locked sync and 123 tests passed.
- [ ] Run the adapted replication and compare only the direction with MOSAIC
  unless a full reproduction is achieved. Codex can prepare and run the
  experiment; the human decides whether the direction matches and whether the
  comparison is credible. See [MOSAIC replication](mosaic-replication.md).

## Current evidence at a glance

| Area | Current evidence | What is still missing |
| --- | --- | --- |
| Templates | Provisional 57-template set; every `K_attr = 1` cell covers all four colours and three shapes; 100-scene-per-template sampling smoke test | Final vocabulary and template decision, full counterbalancing, held-out leave-out set, proposal-scale A-matrix |
| Pilot training | 100-step exploratory CUDA run and sample completed; sample was noise | Useful generated images, reference model, resume, controlled guidance sweep and human audit |
| Checker | Clean-render and controlled-perturbation audits; rejection counts reported separately | Generated-image human audit, per-predicate/per-rule error report, evidence for `≤ 2%` on real renders |
| Decorrelation | 20 exploratory designs and 1,140 rule-by-design rows from the 57-condition pilot; pooled correlation `-0.446`; VIF `11.61` for log lift and `13.72` for relation count | Registered gate analysis with breadth, uncertainty, and the final matrix. The current values do not establish the proposal criterion |
| MOSAIC | Preliminary paper/code review only | Adapted contrast specification, GPU run, results, and direction comparison |
| GPU | Windows RTX 3050 6GB; locked CUDA environment and 123 tests passed; 100-step project run used about 42 MiB allocated and 62 MiB reserved | Longer-run memory and research compute budget |
| Human audit | Blinded rule and predicate prepare/predict/score workflow; four team members reported as potential reviewers available later | Confirmed independence/blinding, at least 300 real generated images, actual judgments |

See [Stage 0 status](stage0-status.md) for the full current report and commands.

## Final go/no-go

- [ ] The human makes the Stage 0 decision against the three proposal criteria
  above. No additional threshold or sign-off requirement is implied by this
  checklist.
- [ ] For project records, note the decision date, evidence locations, and any
  deviations if useful. This is record-keeping, not a proposal requirement.

Passing tests, generating reports, completing exploratory diagnostics, or
finishing the enabling work does not by itself change the current **Not passed**
status.
