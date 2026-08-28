# Intervene3D

**Interventional Identifiability of Physical Geometry under Optical Ambiguity**

Preliminary research codebase — version 0.1.0

---

## Motivation

Computer vision normally assumes that enough visual evidence will eventually
reveal the correct 3-D interpretation. This project challenges that assumption.

A corridor seen through a doorway, a corridor reflected in a mirror, and a
corridor displayed on a screen can produce the **same image**. Existing depth
systems answer "what is the depth?" and are known to be fooled by exactly these
cases. Intervene3D asks a different question:

> **Is the physical interpretation of this visual geometry identifiable under the
> allowed observer interventions?**

and, when the answer is no, says so:

> *physical explanation unresolved under available visual evidence*

## Research questions

| | |
|---|---|
| **RQ1** | Can competing physical explanations of visually ambiguous geometry be distinguished by their predicted responses to controlled observer motion? |
| **RQ2** | Can a system predict when **no** available intervention is sufficient? |
| **RQ3** | Can intervention-conditioned selection resolve ambiguity with less motion than passive or generic next-best-view strategies? |
| **RQ4** | Does identifiability-aware inference reduce confidently incorrect physical geometry predictions? |

This milestone answers RQ1, RQ2 and RQ4 on a synthetic benchmark, and RQ3
partially. See **Results** below and `docs/EXPERIMENT_PLAN.md`.

---

## The formulation

Competing optical hypotheses:

```
H_D  direct physical geometry     H_T  transmission / refraction
H_R  reflection (planar mirror)   H_E  emissive / display
H_M  mixed optical mechanism
```

At the reference view they are **matched by construction**:
`p(I₀|H_D) = p(I₀|H_R) = p(I₀|H_E)`, exactly, not approximately.

Intervene on observer pose, `a = ΔC ∈ SE(3)`, and each hypothesis implies a
different distribution over future observations. Separability and
**action-set identifiability**:

```
Δ_ij(a) = D( p(O'|H_i,a), p(O'|H_j,a) )
I_A(H_i,H_j) = max_{a ∈ A} Δ_ij(a)
I_A < ε   →   not identifiable under A   →   abstain
```

Identifiability is **relative to an action set and a perceptual resolution**, not
an absolute property. "Unidentifiable" is deliberately *not* a physical
hypothesis — it is an inference outcome about the triple `(H, A, O)`.

---

## Architecture

```
Synthetic scene → reference observation I₀
      ↓
candidate hypotheses {H_D, H_E, H_R}
      ↓
candidate camera actions A (bounded, configurable)
      ↓
analytical hypothesis-conditioned prediction  F̂ᵏ,ᵃ = W(F_t, H_k, a)
      ↓
pairwise separability Δ_ij(a)
      ↓
choose the intervention a* = argmax_a Σ_{i<j} p_i p_j Δ_ij(a)
      ↓
observe → compare with predictions → belief update p ∝ e^{−β e_k} p
      ↓
identifiability decision
   ├── resolved   → physical explanation + contact geometry
   └── unresolved → explicit abstention
```

Six swappable components (`models/interfaces.py`): `GeometryEncoder`,
`TransitionModel`, `SeparabilityEstimator`, `BeliefUpdater`,
`InterventionSelector`, `IdentifiabilityEstimator`. Physics stays explicit and
inspectable — it is not hidden inside a network. Full detail in
`docs/SOFTWARE_ARCHITECTURE.md`.

---

## Repository structure

```
├── configs/          default, smoke, synthetic/, experiments/, datasets/, models/
├── scripts/          setup, download, validate, generate, smoke test, run, aggregate, figures
├── src/intervene3d/  config data geometry optics hypotheses interventions
│                     models inference metrics experiments visualization reproducibility
├── tests/            unit (11 files) · integration · smoke   — 224 tests
├── data/             raw/ interim/ processed/ manifests/
├── experiments/      ONE top-level directory; one immutable directory per run
├── results/          cross-seed aggregates
├── figures/          exported figure sets
└── docs/             the audits, plans and this project's honesty record
```

---

## Installation

```bash
bash scripts/setup_environment.sh          # creates .venv, installs, verifies
source .venv/bin/activate
```

CPU-only, and that is the fully supported path — the preliminary pipeline needs
only NumPy, Matplotlib, PyYAML and SciencePlots. PyTorch is **optional** and is
needed only for a real geometry foundation encoder (Gate 6) or a scaled world
model (Gate 7):

```bash
bash scripts/setup_environment.sh --gpu
conda env create -f environment.yml        # alternative
```

The setup script verifies the interpreter, dependencies, headless rendering, the
IEEE style (no LaTeX required), PyTorch/CUDA if present, the package import, and
runs a small end-to-end validation of the analytical optics.

---

## Smoke test

```bash
bash scripts/run_smoke_test.sh
```

Tiny, CPU-only, no network, ~15 s. It exits non-zero if **any** step fails.

<details>
<summary>Expected output</summary>

```
================ Intervene3D smoke test ================
python : Python 3.13.14
repo   : /path/to/Computer-Vision-Research

[01] Steps 1-11: package, config, scene, hypotheses, actions, predictions,
     separability, selection, observation, belief, metrics
     intervene3d 0.1.0 imported
     config validated: intervene3d_synth_smoke, mechanisms=['direct', 'emissive', 'reflection']
     config validation correctly rejects an invalid config
     scene created: 24 content landmarks, 3 observer markers, interface at 2.42 m
     hypotheses: ['H_D', 'H_E', 'H_R'] (['Direct', 'Display', 'Mirror'])
     matched at C_0: max deviation 0.00e+00 px across all variants
     action set |A| = 15, bounds: <= 0.35 m / 15.0 deg
     predicted 45 consequences; separability tensor (15, 3, 3)
     H_D: action=  translate_y-0.300m  p_max=0.500  I_A= 0.00  ABSTAIN  AbsRel_contact=0.0000
     H_E: action=  translate_y-0.300m  p_max=0.500  I_A= 0.00  ABSTAIN  AbsRel_contact=0.3148
     H_R: action=  translate_y-0.300m  p_max=1.000  I_A=12.00  resolved -> reflection
     metrics: CEA(all)=0.333  CEA(committed)=1.000  abstention=0.667  FPCR=0.0
     steps 1-11 OK
     PASS package, configuration, scene, hypotheses, actions, predictions,
          separability, selection, observation, belief update and metrics

[02] Generating the tiny synthetic benchmark
generated 36 scene variants over 12 base scenes
  resolvable      : 30
  non-identifiable: 6 (16.7%)
  [PASS] checksums: 41/41 files match
  [PASS] matched_counterfactuals: max reference-view pixel deviation = 5.684e-14
  [PASS] split_leakage: no base scene spans splits
  [PASS] contains_non_identifiable_cases: 30 resolvable / 6 non-resolvable
  [PASS] reproducibility_metadata: seed, config_hash and action_space recorded
  [PASS] array_shapes: all arrays well-shaped
  RESULT: PASSED

[03] Steps 12-13: full experiment run (unique run directory + figures)
RUN SUCCEEDED: experiments/smoke_test/run_<UTC>_seed0_<hash>
figures : 44 files

[04] Steps 14-16: verifying figures, metrics and the run manifest
     run manifest OK (seed=0, config_hash=..., git=...)
     config / command / git / environment / dataset manifest / summary all present
     metrics OK (4 methods, 15 eval scenes)
     figures OK (22 PDF + 22 PNG, all non-empty)

[05] Regenerating figures from saved result files only
     PASS every figure is reproducible from metrics/figure_data.json

================ SMOKE TEST PASSED ================
```

Note the first block: on this scene `H_D` and `H_R` are **correctly abstained on**
(`I_A = 0`, no allowed action separates them) while `H_R`'s own scene resolves. A
CEA of 0.333 with a *committed* accuracy of 1.000 is the abstention mechanism
working, not a failure.
</details>

Tests: `.venv/bin/python -m pytest tests -q` → **224 passed** in ~9 s.

---

## The synthetic benchmark

Built first, on purpose: nothing depends on an external dataset.

```bash
python scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml
python scripts/validate_datasets.py --synthetic data/processed/intervene3d_synth_phase1
```

Every base scene yields matched causal variants sharing one optical interface and
one action set, so the reference views are **pixel-identical** and only the
mechanism differs. Each variant stores RGB-free landmark observations for every
action, contact geometry, the oracle separability tensor, resolvability, and three
resolving-baseline estimates.

**The benchmark is deliberately not always solvable.** In the Phase 1 draw,
**103 of 288 variants (36 %) are non-identifiable**, arising from geometry rather
than by declaration:

- a **view-tracked display** is geometrically identical to a real scene inside its
  aperture — no baseline resolves it;
- a **mirror** is unresolvable when no allowed action brings the virtual image of
  observer-attached structure inside the aperture;
- a **shallow scene** has a resolving baseline beyond the action bounds.

Splits are by **base scene**, never by frame or pose, so matched counterfactuals
can never straddle a split.

---

## Dataset acquisition

```bash
python scripts/validate_datasets.py --list       # registry with status and licence
bash   scripts/download_datasets.sh --dataset transphy3d
python scripts/validate_datasets.py --all
```

Thirteen external datasets are registered with verified paper citations, licence
status and acquisition instructions. **None is downloaded automatically**, and
none is required here. `download_datasets.sh` attempts a download only when both
the licence and the download permission are recorded as `verified`; as of the
2026-08-28 audit no dataset meets that bar, so it prints instructions instead.
See `docs/DATASET_MATRIX.md`.

---

## Running experiments

```bash
python scripts/run_experiment.py \
  --config configs/experiments/phase1_problem_existence.yaml --seed 42

python scripts/run_experiment.py \
  --config configs/experiments/phase1_problem_existence.yaml --seed 3 \
  --set model.abstention.tau=0.9

python scripts/aggregate_results.py --experiment phase1_problem_existence
python scripts/generate_all_figures.py
```

Each run creates a unique, immutable directory and prints the command that
reproduces it. `make help` lists every target.

---

## Results — Phase 1 problem existence

3 seeds, **benchmark redrawn per seed**, 90 evaluation variants per seed,
66 % resolvable, chance = 0.333. Mean ± std across seeds.

| Method | CEA | abstain | AUROC | FPCR | regret |
|---|---|---|---|---|---|
| single-frame classifier | **0.333 ± 0.000** | 0.000 | 0.515 ± 0.132 | 0.000 | n/a |
| generic NBV proxy | 0.365 ± 0.045 | 0.467 | 0.902 ± 0.011 | 0.000 | 0.642 ± 0.049 |
| random intervention | 0.405 ± 0.056 | 0.441 | 0.922 ± 0.013 | 0.000 | 0.563 ± 0.032 |
| max-baseline intervention | 0.505 ± 0.031 | 0.379 | 0.969 ± 0.014 | 0.000 | 0.360 ± 0.056 |
| passive multi-view classifier | **0.621 ± 0.051** | 0.000 | 0.699 ± 0.045 | **0.841 ± 0.160** | n/a |
| no hypothesis conditioning | **0.000 ± 0.000** | **1.000** | 0.500 ± 0.000 | 0.000 | 0.612 ± 0.040 |
| ours, noisy encoder | 0.552 ± 0.031 | 0.342 | 0.997 ± 0.005 | 0.000 | 0.243 ± 0.069 |
| ours, forced choice | **0.714 ± 0.017** | 0.000 | 1.000 ± 0.000 | **0.810 ± 0.081** | 0.235 ± 0.074 |
| **ours** | 0.556 ± 0.024 | 0.339 | **1.000 ± 0.000** | **0.000 ± 0.000** | **0.235 ± 0.074** |

- **RQ1 supported.** `0.333` (chance) `< 0.621 ± 0.051 < 0.714 ± 0.017`, and the
  ordering is **strictly increasing in every one of the three seeds**. The
  intervention-aware tier uses the forced-choice variant so all three tiers are
  compared like for like — counting an abstention as an error would otherwise
  penalise the only method able to abstain for doing the right thing.
- **RQ2 supported.** Identifiability AUROC `1.000` vs `0.699 ± 0.045` for passive
  confidence and `0.515 ± 0.132` for single-frame confidence.
- **RQ4 supported.** FPCR `0.810 ± 0.081 → 0.000 ± 0.000` when abstention is
  enabled, against `0.841 ± 0.160` for the passive baseline's ordinary confidence
  calibration. `metrics.json` records a sweep over `τ ∈ {0.40 … 0.90}` so the
  conclusion does not rest on one threshold.
- **Hypothesis conditioning is load-bearing.** Remove it and every pairwise
  separability is identically zero: nothing is identifiable and the system
  abstains on 100 % of scenes. The ablation does not degrade the method, it
  destroys it — which is the correct outcome.
- **MCRB theory validated.** `1/B_min` is linear in `f|1/Z₁−1/Z₂|` with
  `R² = 0.85, 0.83, 0.91` across the three seeds.
- **Robust to encoder noise.** 0.6 px + 2 % depth noise changes CEA by `0.004`.
- **RQ3 partial.** Regret is clearly better (`0.235` vs `0.563` random, `0.642`
  generic NBV), but raw motion cost is *higher* than for strategies that move
  little and fail. The motion-to-resolution comparison RQ3 actually calls for is
  Phase 4 work and is **not** claimed here.

## Where results live

```
experiments/<name>/run_<UTC>_seed<N>_<confighash>/
├── config.yaml  command.txt  git_commit.txt  environment.txt
├── dataset_manifest.json  run_manifest.json  summary.md
├── metrics/{metrics.json, summary.json, figure_data.json}
├── predictions/predictions.csv
├── tables/method_comparison.{csv,md}
└── figures/*.{pdf,png}          22 IEEE-style figures
results/<name>/aggregate.{json,csv}
experiments/registry.jsonl       append-only, one line per run
```

Nothing lives only in a log. Every figure is drawn from
`metrics/figure_data.json`, so `scripts/generate_all_figures.py` rebuilds the
whole set without re-running anything.

---

## Reproducibility

Every run records the seed, Python version, OS, CPU/GPU, CUDA availability,
package versions, a full frozen environment, the Git commit and dirty state, the
configuration and its hash, the dataset manifest and checksums, the exact command,
the reproduction command, timestamps, hostname, the relevant environment variables
and the deterministic settings applied.

Verified guarantees: the same seed produces the same synthetic scene; regenerating
a dataset produces byte-identical files; split assignment is stable as the dataset
grows; run directories are never overwritten; failed runs are recorded with their
traceback rather than silently lost. Details in `docs/REPRODUCIBILITY.md`.

---

## Known limitations

Read these before quoting any number.

1. **The simulator and the analytical transition share the same forward optics.**
   Results with the `ground_truth` encoder are an **upper bound and a pipeline
   check**, not evidence about real imagery. This is why identifiability AUROC is
   exactly `1.000` and `MAE_MCRB ≈ 0` — the model's identifiability computation is
   the same one that produced the label. The meaningful contrasts are against the
   baselines. Two independent cross-checks (mirror virtual camera, display
   homography) validate the *optics*, but not the renderer.
2. **The near-chance single-frame result is a property of the benchmark's
   construction**, not a discovery. The variants are pixel-identical at `C_0` by
   design; measuring it confirms the construction.
3. **Interface parameters are assumed known** (plane pose, aperture, slab
   thickness). A real system would estimate them and that error would propagate.
4. **The observation model is landmark-based**, not dense.
5. **`H_M` (mixed optics) is implemented but unvalidated** and excluded from Phase 1.
6. **No external or real-world data has been used.**
7. **A static planar mirror is not resolvable by content parallax at all** — a
   correction to the source specification, documented in
   `docs/RESEARCH_SPEC_AUDIT.md` §5 (A2). Mirrors are identifiable here only via
   virtual images of observer-attached structure.
8. **`D_feature` is not implemented** and `D_geometry` is off by default; a
   non-zero `λ_feature` is rejected rather than silently ignored.
9. **RGB rendering is a splat renderer**, adequate for figures only.

## Novelty position

Not claimed: multilayer depth, physical-vs-apparent depth, mirror/glass
correction, reflection/transmission decomposition, camera motion as a cue, active
next-best-view, counterfactual world models, or causal identifiability generally.
All are occupied — see `docs/NOVELTY_RISK_REGISTER.md`, which names the specific
prior work for each and gives safe wording.

Claimed, with the qualifier *"to our knowledge, based on a targeted literature
audit through August 2026"*: **action-set-dependent identifiability of competing
optical explanations, explicit prediction of non-identifiability, and a matched
counterfactual benchmark containing unresolvable cases.**

---

## Documentation

| Document | Contents |
|---|---|
| `docs/RESEARCH_SPEC_AUDIT.md` | every equation → implementation; ambiguities found and how they were resolved; conceptual vs testable claims |
| `docs/LITERATURE_CROSS_RESEARCH.md` | independent verification of the novelty-critical citations; new work found |
| `docs/NOVELTY_RISK_REGISTER.md` | claim / existing work / overlap / evidence / safe wording |
| `docs/DATASET_MATRIX.md` | 13 external datasets: source, licence, access status, role |
| `docs/SOFTWARE_ARCHITECTURE.md` | conventions, package boundaries, interfaces, testing strategy |
| `docs/EXPERIMENT_PLAN.md` | every experiment with its results, or an explicit `NOT RUN` |
| `docs/REPRODUCIBILITY.md` | seeds, manifests, hashing, multi-seed protocols |
| `docs/VISUALIZATION_PLAN.md` | the IEEE style system and all 22 figures |
| `docs/DEVELOPMENT_ROADMAP.md` | Gates 1–7, what passed, what is next |
| `docs/TROUBLESHOOTING.md` | symptom → cause → fix |
| `docs/IMPLEMENTATION_LOG.md` | the complete record of the work: build chronology, the six defects found and fixed during development, tuning sweeps and their evidence, file-by-file inventory, commands actually executed |

## Licence

MIT (`LICENSE`). Chosen as a permissive default for a research repository; change
it if your institution requires otherwise. External datasets carry their own
licences — see `docs/DATASET_MATRIX.md`.
