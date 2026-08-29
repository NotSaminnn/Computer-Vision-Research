# Experiment Plan

Every experiment specifies: research question, hypothesis, inputs, outputs,
baseline, model, dataset, configuration, metrics, expected artefacts, required
plots, and success/failure criteria.

**Status vocabulary:** `RUN` (executed, results below) · `IMPLEMENTED, NOT RUN` ·
`NOT IMPLEMENTED`.

---

## E0 — Smoke test · **RUN**

| | |
|---|---|
| **Question** | Does the complete pipeline execute from a clean environment? |
| **Inputs** | `configs/synthetic/smoke.yaml`, `configs/smoke_test.yaml` |
| **Model** | analytical transition, ground-truth encoder |
| **Metrics** | all of them, on a tiny split |
| **Command** | `bash scripts/run_smoke_test.sh` |
| **Artefacts** | a full run directory; 22 figures (PDF + PNG) |
| **Success** | every one of the 17 required steps passes; non-zero exit on any failure |
| **Result** | **PASSES.** 12 base scenes → 36 variants; 6 non-resolvable; ~15 s end to end. |

---

## E1 — Sanity checks · **RUN** (as the test suite)

| | |
|---|---|
| **Question** | Do the primitives obey the conventions and the physics? |
| **Command** | `.venv/bin/python -m pytest tests -q` |
| **Success** | all pass |
| **Result** | **276 tests pass** (~35 s). Includes the two independent optics cross-validations, the matched-counterfactual invariant (deviation `0.000e+00`), and the provable non-separability of a view-tracked display. |

---

## E2 — Phase 1: problem existence · **RUN**

| | |
|---|---|
| **Question (RQ1)** | Can competing physical explanations of appearance-matched geometry be distinguished through controlled observer motion? |
| **Hypothesis (H1)** | `single-frame < passive multi-view < intervention-aware` in causal explanation accuracy. |
| **Inputs** | Intervene3D-Synth Phase 1: 96 base scenes × {direct, display, mirror} = 288 variants (36 % non-identifiable); 33 candidate actions; ε = 1 px-equivalent; τ = 0.45 |
| **Outputs** | per-scene predicted mechanism, posterior, identifiability score, selected action, contact depth, predicted MCRB |
| **Baselines** | single-frame classifier; passive multi-view classifier; random; max-baseline; generic-NBV proxy |
| **Ablations** | no hypothesis conditioning; no abstention; noisy encoder |
| **Model** | analytical transition, ground-truth encoder, max-separability selector, likelihood belief update, epsilon abstention |
| **Config** | `configs/experiments/phase1_problem_existence.yaml` |
| **Metrics** | CEA (overall + per mechanism), identifiability AUROC, FPCR (+ τ sweep), AbsRel/RMSE contact, MAE MCRB, intervention regret, motion cost |
| **Command** | `python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed 1` (seeds 1-10 and 42) |
| **Artefacts** | 11 run directories, `metrics.json`, `predictions.csv`, `method_comparison.{csv,md}`, 22 figures each, `results/phase1_problem_existence/aggregate.{json,csv}` |
| **Success criteria** | (a) single-frame at chance; (b) passive > single-frame; (c) intervention-aware > passive; (d) FPCR much lower with abstention than without |

### Result (11 seeds, benchmark redrawn per seed, mean ± std)

| single-frame classifier | `0.333 ± 0.000` | 0.000 | `0.505 ± 0.066` | `0.000 ± 0.000` | n/a | n/a | 0.000 |
| generic NBV proxy | `0.371 ± 0.050` | 0.475 | `0.898 ± 0.026` | `0.000 ± 0.000` | 0.246 | `0.627 ± 0.047` | 0.091 |
| random intervention | `0.409 ± 0.056` | 0.440 | `0.925 ± 0.024` | `0.000 ± 0.000` | 0.248 | `0.567 ± 0.038` | 0.139 |
| max-baseline intervention | `0.502 ± 0.061` | 0.390 | `0.963 ± 0.028` | `0.000 ± 0.000` | 0.211 | `0.353 ± 0.034` | 0.300 |
| passive multi-view classifier | `0.621 ± 0.046` | 0.000 | `0.748 ± 0.065` | `0.868 ± 0.107` | n/a | n/a | 0.200 |
| no hypothesis conditioning | `0.000 ± 0.000` | 1.000 | `0.500 ± 0.000` | `0.000 ± 0.000` | 0.406 | `0.605 ± 0.049` | 0.050 |
| ours, noisy encoder | `0.557 ± 0.034` | 0.342 | `0.994 ± 0.016` | `0.008 ± 0.026` | 0.188 | `0.223 ± 0.066` | 0.261 |
| ours, forced choice | `0.722 ± 0.029` | 0.000 | `1.000 ± 0.000` | `0.832 ± 0.107` | 0.182 | `0.225 ± 0.076` | 0.259 |
| **ours** | `0.560 ± 0.032` | 0.343 | `1.000 ± 0.000` | `0.000 ± 0.000` | 0.182 | `0.225 ± 0.076` | 0.259 |

Chance = 0.333. Evaluation split: 90 variants per seed.

**Verdict — all four success criteria met.**

- (a) single-frame CEA is `0.333 ± 0.000`, **exactly** chance in every seed. The
  variants are pixel-identical at `C_0`, so this is a property of the benchmark's
  construction, not a discovery — but it is the premise the whole project rests
  on, and it is now measured rather than assumed.
- (b, c) `0.333 < 0.621 ± 0.046 < 0.722 ± 0.029`, and the ordering is **strictly
  increasing in all 11 seeds individually**, not only on the mean. The
  intervention-aware tier uses the forced-choice variant so all three tiers are
  compared like for like.
- (d) FPCR `0.832 ± 0.107` without abstention → **`0.000 ± 0.000`** with it, while
  the passive baseline's ordinary confidence calibration sits at `0.868 ± 0.107`.
  This is the strongest result in the run and directly supports H2.

**Also observed:**

- **Hypothesis conditioning is load-bearing.** Remove it and every pairwise
  separability is identically zero, so nothing is ever identifiable and the system
  abstains on 100 % of scenes. The ablation does not degrade the method — it
  destroys it, which is the correct outcome.
- **Intervention selection matters (RQ3, partially).** Normalised regret
  `0.225` (ours) vs `0.353` (max-baseline), `0.567` (random), `0.627` (generic NBV).
- **Robustness.** A noisy encoder (0.6 px, 2 % depth) changes CEA by `0.003` and
  AUROC by `0.006`.

**Changes from the earlier 3-seed reading** (kept visible rather than silently
overwritten — every one of these moved because the sample grew, not because
anything was retuned):

| quantity | 3 seeds | 11 seeds |
|---|---|---|
| passive multi-view AUROC | `0.699 ± 0.045` | `0.748 ± 0.065` |
| passive multi-view FPCR | `0.841 ± 0.160` | `0.868 ± 0.107` |
| forced-choice CEA | `0.714 ± 0.017` | `0.722 ± 0.029` |
| ours CEA | `0.556 ± 0.024` | `0.560 ± 0.032` |
| ours normalised regret | `0.235 ± 0.074` | `0.225 ± 0.076` |
| **ours, noisy encoder FPCR** | `0.000 ± 0.000` | **`0.008 ± 0.026`** |

The last row is the one that matters. At 3 seeds the noisy-encoder ablation
produced a false physical certainty rate of *exactly* zero; across 11 seeds a
single seed produces one. The abstention mechanism is therefore **not**
unconditionally safe under encoder noise, and the claim must be stated as
"zero in 10 of 11 seeds" rather than "zero". The clean configuration remains at
`0.000 ± 0.000` across all 11.

**What these numbers do NOT show** (repeated in every `summary.md`):

- The `intervene3d` identifiability AUROC of exactly `1.000` is **degenerate**:
  the model's identifiability computation is the same one that produced the
  ground-truth label. The meaningful contrast is against the classifiers
  (`0.748`, `0.505`), not the absolute value. `MAE_MCRB ≈ 0.001` has the same cause.
- Raw motion cost is *higher* for our method (`0.259`) than for random (`0.139`)
  or generic NBV (`0.091`), because those pick small actions and fail. RQ3's
  "less motion" claim needs a motion-to-resolution comparison restricted to
  successful resolutions — Phase 4 work, not claimed here.

## E3 — Analytical optics validation (MCRB theory) · **RUN**

| | |
|---|---|
| **Question** | Does the measured resolving baseline obey `B_min > δ / (f\|1/Z₁ − 1/Z₂\|)`? |
| **Method** | Compensate the screen plane's projective warp — which is exactly what the derivation assumes — leaving only the differential parallax a planar display cannot reproduce. Then regress `1/B_min` on `f\|1/Z₁ − 1/Z₂\|`. |
| **Success** | a linear relation; `R² > 0.8` |
| **Result** | **`R² = 0.851 / 0.833 / 0.907`** across seeds 1/2/3, slopes `0.210 / 0.148 / 0.166`, over 22 / 22 / 17 applicable static-display scenes. |
| **Figure** | `fig18_mcrb_theory_validation` |

The **slope is not expected to equal `f/δ`**: the derivation uses the extremal
depth pair while the measurement is an RMS residual after absorbing the best
planar warp. The informative quantity is the linearity, and it holds.

Note also that the *operational* (anchored) MCRB is systematically smaller than
the analytic one, as it must be: anchoring both hypotheses to a shared reference
view is a strictly easier setting than compensating an unknown screen homography.
The two are stored separately and never conflated.

---

## E4 — Benchmark core (Phase 2) · **RUN** (2026-08-29, 11 seeds)

Adds `H_T` (planar glass). `H_M` (mixed) remains excluded — still unvalidated.
Config `configs/synthetic/benchmark_core.yaml` (192 base scenes), experiment
`configs/experiments/phase2_benchmark_core.yaml`. Chance is 0.250, 256 eval
scenes per seed.

**The prediction held.** Adding transmission raises non-identifiability from
36 % (Phase 1) to roughly **70 %** — the graded ambiguity ladder this entry
predicted, because the paraxial axial shift `d(1 − 1/n)` is millimetric and only
a large baseline resolves it.

| method | CEA | abstains | committed: direct · display · mirror · glass |
|---|---|---|---|
| single-frame | 0.250 ± 0.000 | 0 % | — (exactly chance, as constructed) |
| passive multi-view | 0.510 ± 0.021 | 0 % | 0.424 · 0.747 · 0.276 · 0.595 |
| max-baseline | 0.264 ± 0.025 | 73 % | 1.000 · 1.000 · 0.925 · 1.000 |
| ours, summed objective | 0.306 ± 0.022 | 69 % | 1.000 · 1.000 · 0.956 · 1.000 |
| **ours, maximin** | 0.280 ± 0.034 | 72 % | **1.000 · 1.000 · 1.000 · 1.000** |
| ours, maximin, forced | **0.808 ± 0.024** | 0 % | — |

Forced-choice CEA beats the passive baseline by **+0.297** (t = +41.6,
11/11 seeds). Maximin holds committed accuracy at **1.000 ± 0.000 on all four
mechanisms** across all 11 seeds; the summed objective is the one that fails, and
it fails only on mirrors (0.956 ± 0.037) — the mechanism whose deciding
pair a sum can leave unseparated.

**Honest caveat:** in Phase 2 maximin is significantly *worse* than the summed
objective on normalised regret (+0.197, t = +19.56) and identifiability AUROC
(-0.121, t = -16.29). Oracle utility is defined as `sep[:, true, hardest]` — one
pair — so a criterion satisfying all six is penalised by a metric scoring one.
Report both; do not tune to it.

**Seed count corrected.** This entry previously reported 3 seeds. At 11 the
qualitative result is unchanged and the mirror gap widens slightly; the
forced-choice margin moved from +0.315 to +0.297.

---

## E5 — Geometry world model (Phase 3) · **NOT IMPLEMENTED**

A frozen geometry foundation encoder plus a hypothesis-conditioned transition.
Blocked on Gate 6: `moge` and `vggt_like` adapters exist but raise
`NotImplementedError` with an installation checklist. The learned residual
(`models/learned_transition.py`) is real and tested but is a placeholder, not a
world model.

---

## E6 — Active intervention (Phase 4) · **PARTIALLY RUN**

Selection and regret are implemented and measured (E2). Not done:
multi-step sequential intervention (the engine supports `max_steps > 1` but Phase 1
uses one step), and the motion-to-resolution comparison RQ3 actually needs.

---

## E7 — Abstention study · **RUN as part of E2**

The `intervene3d` vs `intervene3d_no_abstention` pair isolates it exactly: same
predictions, same posterior, same selected action; the only difference is whether
the system is permitted to decline. FPCR `0.810 → 0.000`; CEA(committed) rises
while CEA(all) falls, which is the expected and correct trade.

`metrics.json` records an FPCR sweep over `τ ∈ {0.40 … 0.90}` so no conclusion
rests on one threshold. This matters here: with a three-hypothesis family whose
residual ambiguity is two-way, a forced argmax over an unresolvable pair sits at
exactly `p = 0.5`, so any `τ ≥ 0.5` reports zero for the wrong reason. `τ = 0.45` is used for the headline number. At `τ = 0.40` the
no-abstention sweep reads `0.727` and the passive classifier `1.000`, against
`0.000` for the full method.

---

## E8 — Generalisation · **PARTIALLY RUN** (action noise: 2026-08-29, 10 seeds)

| Study | Mechanism | Status |
|---|---|---|
| Action-execution noise `ΔC + ε` | `action_noise` config; re-simulates for the perturbed action | **RUN** — see below |
| Encoder noise | `mock` encoder | **run** (`intervene3d_noisy_encoder`) |
| Unseen cameras | `camera` block is free (`fx`, resolution) | implemented, not run |
| Unseen materials / scene categories / mixed optics | needs Phase 2 + real data | not implemented |

**Action-execution noise (±1 cm, ±0.5°), paired on the 10 seeds present in both
the clean and noisy experiments.** Pairing matters: an earlier unpaired reading
across 11 clean and 3 noisy seeds gave different figures and must not be used.

| method | clean CEA | noisy CEA | change | t |
|---|---|---|---|---|
| single-frame classifier | 0.333 | 0.333 | +0.0 % | — |
| passive multi-view | 0.617 | 0.617 | +0.0 % | — |
| generic NBV proxy | 0.368 | 0.190 | **−48.4 %** | −12.68 |
| random intervention | 0.405 | 0.289 | −28.7 % | −9.64 |
| max-baseline | 0.496 | 0.472 | −4.7 % | −3.30 |
| **intervene3d** | 0.557 | 0.544 | **−2.3 %** | −2.86 |

Both non-intervening baselines are unchanged at exactly 0.0 %, which is the
sanity check: execution noise cannot affect a method that does not act. Among
methods that do act, the hypothesis-blind proxy loses about half its accuracy
while this one loses 2.3 %.

---

## E9 — Ablations · status

| Ablation | Status | Result |
|---|---|---|
| single image only | **RUN** | CEA 0.333 (exactly chance) |
| random camera movement | **RUN** | CEA 0.405, regret 0.563 |
| maximum-baseline movement | **RUN** | CEA 0.505, regret 0.360 |
| generic uncertainty / NBV | **RUN** | CEA 0.365, regret 0.642 |
| no hypothesis conditioning | **RUN** | CEA 0.000, abstains on everything |
| no abstention | **RUN** | FPCR 0.810 vs 0.000 |
| RGB instead of geometry features | NOT IMPLEMENTED | needs `D_feature` |
| learned-only transition | IMPLEMENTED, training stage IMPLEMENTED (`experiments/learned.py`), NOT RUN | **On the synthetic benchmark the `hybrid` residual target is identically zero, so `hybrid` is EQUAL to `analytical`. This ablation is uninformative there and must not be reported from synthetic data.** |
| analytical-only transition | **RUN** (the default) | — |
| no matched-counterfactual training | NOT IMPLEMENTED | needs an unmatched generator variant |
| noisy camera actions | IMPLEMENTED, NOT RUN | `action_noise.enabled: true` |

---

## E10 — External benchmark evaluation · **RUN** (2026-08-29)

Four external datasets (eight variants, ~57 GB) acquired, checksum-verified
against publisher SHA-256, pinned to immutable revisions;
`intervene3d.data.external.loaders` reads all four formats.

**Evaluated on 3D Visual Illusion (NeurIPS 2025), real stereo**, with
Depth-Anything-V2-Small used as released — nothing trained or tuned on it.
Scale and shift are fitted on the **non-illusion** region and applied inside, so
this is a test rather than a fit. `scripts/evaluate_external.py`.

Of 455 stereo pairs, **296 have ground truth able to arbitrate the question**.
The other 159 are excluded because stereo matching on a display locks onto the
*displayed content* rather than the panel, leaving a measured disparity that is
not planar where the physical surface is. That exclusion is itself a finding —
see the caveat below.

| quantity | value |
|---|---|
| relative error outside the illusion | 0.0266 |
| relative error inside | **0.0504** — 1.90× worse |
| images where inside error exceeds outside | **65.2 %** |
| planarity, ground truth vs prediction | **0.995 vs 0.919** |

| category | n | ratio | fooled |
|---|---|---|---|
| PaperOnTable | 97 | 0.78× | 33 % |
| PosterAndObject | 104 | 1.47× | 63 % |
| **video** (monitor fills frame) | 95 | **2.23×** | **100 %** |

`PaperOnTable` is below 1.0 — the illusion penalty is real for posters and
screens but not for printed paper on a table. Recorded as measured.

**Caveat, and it generalises beyond this dataset.** 35 % of a published
benchmark carries ground truth that cannot support the measurement it is used
for. Before filtering, the `video_monitor` category reported a 1 % failure rate —
the exact inverse of the truth. Evaluating apparent-versus-physical geometry
requires ground truth that is itself immune to the illusion, and existing
benchmarks do not guarantee it.

---

## E11 — Selector study: maximin vs the summed objective · **RUN** (2026-08-29, 11 seeds)

`configs/experiments/phase1_selector_study.yaml`. Motivated by a measurement,
not a hunch: on every mirror scene the full method got wrong, the chosen action
scored `Δ(H_D, H_R) = 0.00` exactly while actions worth 4–12 px-equivalent
existed, and the posterior then sat at a perfect 0.500 tie broken arbitrarily
toward `H_D`. A **summed** objective can be maximised by separating pairs that
are not in contention.

| | summed | **maximin** |
|---|---|---|
| committed accuracy | 0.855 | **1.000** |
| mirror committed accuracy | 0.529 | **1.000** |
| abstention rate | 34 % | 44 % |
| motion cost | 0.259 | **0.160** |

Forced-choice CEA 0.781 vs 0.633 for the passive baseline (+0.148, t = +16.8,
4/4 seeds at the time of that test). The change is structural, not tuned: an
action scoring zero on a pair cannot separate it at any threshold.

**Prior art, stated plainly.** Model-discrimination experiment design is a mature
statistical field — T-optimality (Atkinson & Fedorov 1975), the Hunter–Reiner
criterion. The maximin criterion is **not novel there**. What is defensible is
its transfer into active vision, where expected information gain is the near
universal default, together with the measurement that the default fails.

---

## E12 — Identifiability versus confidence on real data · **RUN** (2026-08-29)

`scripts/evaluate_identifiability.py`. The question selective prediction has
never asked: is the right quantity the model's *confidence*, or whether the
evidence can decide at all? Both signals are computed from the input alone, on
the same 296 images.

| signal | AUROC predicting failure |
|---|---|
| confidence — Chow's rule, TTA flip | **0.339** |
| confidence — 4-member TTA ensemble (strongest) | 0.344 |
| **identifiability (ours)** | **0.632** |

Confidence is **below chance**: the model is more sure when it is wrong. Mean
confidence is −0.98 where the evidence cannot decide against −3.93 where it can.
Strengthening the confidence baseline from a single flip to a four-member
ensemble moves it 0.339 → 0.344, so the gap is not an artefact of a weak opponent.

Absolute honesty on the headline: **0.632 is a modest predictor**. The result is
a contrast, not a strong absolute score, and should be reported as such.

---

## E13 — Split-conformal selective prediction · **RUN** (2026-08-29)

`scripts/evaluate_conformal.py`. The strongest principled opponent: conformal
prediction gives a finite-sample coverage guarantee regardless of the score's
quality. 400 random calibration/test splits per level.

| score | AUROC | α=0.1 | α=0.2 | α=0.3 | α=0.4 | α=0.5 |
|---|---|---|---|---|---|---|
| confidence, conformal | 0.339 | 63.6 % | 64.2 % | 65.8 % | 69.2 % | **76.6 %** |
| **identifiability, conformal** | 0.632 | 62.4 % | 57.9 % | **55.1 %** | **54.4 %** | 56.2 % |

No abstention: 65.2 %. **Both honour their coverage guarantee at 5/5 levels** —
conformal is working correctly for each. The separation is entirely in selective
risk, because coverage constrains *how many* images are kept and never *which*.
Calibration inherits the ranking of its score: wrapping a signal that is
anti-correlated with error yields a procedure that is provably valid and
practically worse than not abstaining at all.

---

---

## E14 — Why does identifiability degrade on larger models? · **RUN** (2026-08-29) — **HYPOTHESIS REFUTED**

The sweep (E12b) found identifiability AUROC falling monotonically across the
Depth-Anything-V2 family — `0.632` → `0.578` → `0.448` — while the fooled rate
stayed flat at 65–67 %. The explanation offered at the time was a *hypothesis*,
not a measurement: `delta_DE` is a planarity deviation read off the model's own
prediction, so a smoother large model would leave less deviation to measure.

> **Both halves of that story have since been retired, and the premise went
> first.** The "monotone decline with scale" was a three-point trend inside a
> single family. Extending the sweep to eight checkpoints across three families
> breaks it outright: DepthPro at 952 M scores `0.659`, well above DA-V2-Large's
> `0.448` at 335 M, and the DA3 checkpoints sit near chance at every size
> (`0.481` / `0.566` / `0.496`). There is no scale effect to explain — there is
> per-checkpoint variance that three points inside one family made look like a
> trend. The experiment below still stands as a refutation of the mechanism, and
> is kept because the mechanism was asserted publicly before it was measured.

This experiment tests it. `python scripts/diagnose_smoothness.py`, 296 images,
all four checkpoints, gradient and Laplacian computed on IQR-normalised maps so
that models with different output scales are comparable.

| model | params | AUROC | grad | roughness | plane residual |
|---|---|---|---|---|---|
| dav2b | 97.5 M | 0.578 | 0.0033 | 0.0057 | 0.2981 |
| dav2l | 335.3 M | 0.448 | 0.0031 | 0.0066 | 0.3150 |
| dav2s | 24.8 M | 0.632 | 0.0031 | 0.0064 | 0.2672 |
| transparent | 24.8 M | 0.727 | 0.0144 | 0.0408 | 0.9447 |

**The hypothesis is refuted.** It predicts all three smoothness measures FALL
with model size inside the DA-V2 family. None of them do:

| measure | decreasing with size? | Spearman vs AUROC (n=4) |
|---|---|---|
| `grad_mag` | False | +0.400 |
| `roughness` | False | +0.400 |
| `residual_rms` | False | +0.200 |

Gradient magnitude is flat across a 13× parameter range (`0.0031` → `0.0033` →
`0.0031`), and the plane residual — the raw material `delta_DE` is built from —
*increases* with size (`0.267` → `0.298` → `0.315`) while the AUROC falls. The
deviation is not disappearing. There is more of it, and it has stopped tracking
whether the model was fooled.

**What this means, and it is not comfortable.** The convenient reading — "a
limitation of this particular `Delta`, not of the formulation" — is no longer
available. Larger models deviate from planarity *more*, and that deviation is
increasingly unrelated to whether the model was fooled. The planarity-deviation
instantiation is measuring something that only coincidentally tracked failure.
E16 confirms this from the other direction: replacing the proxy with the
separability the methodology actually defines changes the answer substantially,
which a faithful proxy would not have done.

**Also observed.** The transparency-tuned checkpoint is a strong outlier on every
measure — 4.6× the gradient, 3.5× the plane residual of its same-size sibling —
and has the best AUROC (`0.727`). Across all four checkpoints the sign of the
correlation is positive, but n = 4 and it is driven almost entirely by that one
model. Directional evidence only; not a claim.

---

## Reproducing everything above

```bash
bash scripts/run_smoke_test.sh
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml
for s in 1 2 3; do
  .venv/bin/python scripts/run_experiment.py \
    --config configs/experiments/phase1_problem_existence.yaml --seed $s
done
.venv/bin/python scripts/aggregate_results.py --experiment phase1_problem_existence
.venv/bin/python scripts/generate_all_figures.py
```
