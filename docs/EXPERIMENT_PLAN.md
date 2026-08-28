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
| **Result** | **224 tests pass** (~9 s). Includes the two independent optics cross-validations, the matched-counterfactual invariant (deviation `0.000e+00`), and the provable non-separability of a view-tracked display. |

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
| **Command** | `python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed 1` (also seeds 2, 3) |
| **Artefacts** | 3 run directories, `metrics.json`, `predictions.csv`, `method_comparison.{csv,md}`, 22 figures each, `results/phase1_problem_existence/aggregate.{json,csv}` |
| **Success criteria** | (a) single-frame at chance; (b) passive > single-frame; (c) intervention-aware > passive; (d) FPCR much lower with abstention than without |

### Result (3 seeds, benchmark redrawn per seed, mean ± std)

| Method | CEA | abstain | AUROC | FPCR | AbsRel | regret | motion |
|---|---|---|---|---|---|---|---|
| single-frame classifier | **0.333 ± 0.000** | 0.000 | 0.515 ± 0.132 | 0.000 | n/a | n/a | 0.000 |
| generic NBV proxy | 0.365 ± 0.045 | 0.467 | 0.902 ± 0.011 | 0.000 | 0.241 | 0.642 ± 0.049 | 0.090 |
| random intervention | 0.405 ± 0.056 | 0.441 | 0.922 ± 0.013 | 0.000 | 0.238 | 0.563 ± 0.032 | 0.142 |
| max-baseline intervention | 0.505 ± 0.031 | 0.379 | 0.969 ± 0.014 | 0.000 | 0.211 | 0.360 ± 0.056 | 0.300 |
| passive multi-view classifier | **0.621 ± 0.051** | 0.000 | 0.699 ± 0.045 | **0.841 ± 0.160** | n/a | n/a | 0.200 |
| no hypothesis conditioning | **0.000 ± 0.000** | **1.000** | 0.500 ± 0.000 | 0.000 | 0.398 | 0.612 ± 0.040 | 0.050 |
| ours, noisy encoder | 0.552 ± 0.031 | 0.342 | 0.997 ± 0.005 | 0.000 | 0.192 | 0.243 ± 0.069 | 0.260 |
| ours, forced choice | **0.714 ± 0.017** | 0.000 | 1.000 ± 0.000 | **0.810 ± 0.081** | 0.186 | 0.235 ± 0.074 | 0.259 |
| **ours** | 0.556 ± 0.024 | 0.339 | **1.000 ± 0.000** | **0.000 ± 0.000** | 0.186 | **0.235 ± 0.074** | 0.259 |

Chance = 0.333. Evaluation split: 90 variants per seed; resolvable fraction
`0.633 / 0.719 / 0.632`.

**Verdict — all four success criteria met.**

- (a) single-frame CEA is `0.333 ± 0.000`, **exactly** chance in every seed. The
  variants are pixel-identical at `C_0`, so this is a property of the benchmark's
  construction, not a discovery — but it is the premise the whole project rests
  on, and it is now measured rather than assumed.
- (b, c) `0.333 < 0.621 ± 0.051 < 0.714 ± 0.017`, and the ordering is **strictly
  increasing in all three seeds individually** (`0.333/0.644/0.733`,
  `0.333/0.656/0.708`, `0.333/0.563/0.701`) — not only on the mean. The
  intervention-aware tier uses the forced-choice variant so all three tiers are
  compared like for like.
- (d) FPCR `0.810 ± 0.081` without abstention → **`0.000 ± 0.000`** with it, while
  the passive baseline's ordinary confidence calibration sits at `0.841 ± 0.160`.
  This is the strongest result in the run and directly supports H2.

**Also observed:**

- **Hypothesis conditioning is load-bearing.** Remove it and every pairwise
  separability is identically zero, so nothing is ever identifiable and the system
  abstains on 100 % of scenes. The ablation does not degrade the method — it
  destroys it, which is the correct outcome.
- **Intervention selection matters (RQ3, partially).** Normalised regret
  `0.235` (ours) vs `0.360` (max-baseline), `0.563` (random), `0.642` (generic NBV).
- **Robustness.** A noisy encoder (0.6 px, 2 % depth) changes CEA by `0.004` and
  AUROC by `0.003`.

**What these numbers do NOT show** (repeated in every `summary.md`):

- The `intervene3d` identifiability AUROC of exactly `1.000` is **degenerate**:
  the model's identifiability computation is the same one that produced the
  ground-truth label. The meaningful contrast is against the classifiers
  (`0.699`, `0.515`), not the absolute value. `MAE_MCRB ≈ 0.001` has the same cause.
- Raw motion cost is *higher* for our method (`0.259`) than for random (`0.142`)
  or generic NBV (`0.090`), because those pick small actions and fail. RQ3's
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

## E4 — Benchmark core (Phase 2) · **IMPLEMENTED, NOT RUN**

Adds `H_T` (planar glass) and `H_M` (mixed). Config exists:
`configs/synthetic/benchmark_core.yaml` (192 base scenes, 4 mechanisms).
Not run because `H_M` is unvalidated and the plan gates Phase 2 behind Phase 1.

**Expected:** `H_T` should be resolvable only at large baselines, since the
paraxial axial shift `d(1 − 1/n)` is millimetric — giving a graded ambiguity
ladder and more non-identifiable cases.

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

## E8 — Generalisation · **IMPLEMENTED, NOT RUN**

| Study | Mechanism | Status |
|---|---|---|
| Action-execution noise `ΔC + ε` | `action_noise` config; re-simulates for the perturbed action | implemented, not run |
| Encoder noise | `mock` encoder | **run** (`intervene3d_noisy_encoder`) |
| Unseen cameras | `camera` block is free (`fx`, resolution) | implemented, not run |
| Unseen materials / scene categories / mixed optics | needs Phase 2 + real data | not implemented |

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
| learned-only transition | IMPLEMENTED, NOT RUN | needs a training stage |
| analytical-only transition | **RUN** (the default) | — |
| no matched-counterfactual training | NOT IMPLEMENTED | needs an unmatched generator variant |
| noisy camera actions | IMPLEMENTED, NOT RUN | `action_noise.enabled: true` |

---

## E10 — External benchmark evaluation · **NOT RUN**

No external dataset has been downloaded. See `docs/DATASET_MATRIX.md`.
Gate 5 work.

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
