# Implementation Log — the complete record of work done

**Session date:** 2026-08-28 / 29 · **Codebase version:** 0.1.0
**Task:** implement `CLAUDE_INTERVENE3D_PRELIMINARY_CODEBASE_PLAN_PROMPT.md`
**Source of truth for the science:** `research(1).md` (2,549 lines, read in full)

---

## 0. Why this document exists

`README.md` documents **the project** — what it is, how to install and run it,
what the results are, what its limitations are. It does **not** document
**the work**: the order things were built in, the bugs found and fixed along the
way, the decisions taken and why, what was tuned and on what evidence, or the
exact commands actually executed.

This file is that record. Everything here is either verifiable from the
repository or was executed in this session; nothing is projected or planned.

**Reading guide.** §1 is the summary. §2 is the file-by-file inventory. §3 is the
build chronology. **§4 is the most important section: the six defects found and
fixed during the build.** §5 covers the tuning decisions, §6 the deliberate
non-implementations, §7 the literature verification, §8 the executed commands,
§9 what the README does and does not cover.

---

## 1. Summary of what exists

| Category | Amount | Verified |
|---|---|---|
| Python modules under `src/intervene3d/` | 80 files, **10,583 lines** | `find src -name '*.py' \| wc -l` |
| Tests | 13 files, 2,214 lines, **224 test cases** | `pytest tests` → 224 passed in 8.6 s |
| Scripts | 8 executable + 1 bootstrap | all run |
| Config files | 10 YAML | all validate (`test_shipped_configs_are_all_valid`) |
| Documentation | 10 docs + README, **2,140 lines** | — |
| Synthetic benchmark | 288 scene variants over 96 base scenes | generated, validated |
| Experiment runs | 3 Phase-1 seeds + 1 smoke, all `success` | `experiments/registry.jsonl` |
| Figures | 22 per run × 2 formats = 44 files/run | regenerated from saved data |
| Lint | `ruff check src scripts tests` | **All checks passed** |

---

## 2. File-by-file inventory

### 2.1 Package (`src/intervene3d/`, 80 files, 10,583 lines)

| Module | Lines | What it does |
|---|---|---|
| `__init__.py` | 16 | package docstring, version |
| **config/** | | |
| `config/loader.py` | 172 | YAML load, `_base` inheritance, `--set` overrides, canonical JSON, config hashing |
| `config/schema.py` | 253 | explicit validation + defaults for synthetic and experiment configs |
| `config/__init__.py` | 37 | public surface |
| **utils/** | | |
| `utils/io.py` | 122 | deterministic JSON / JSONL / CSV writers with NumPy handling |
| `utils/logging.py` | 63 | one log stream to stdout + `run.log`; silences matplotlib/fontTools |
| `utils/__init__.py` | 24 | |
| **geometry/** | | |
| `geometry/se3.py` | 112 | rotations, exact SE(3) inverse, composition, magnitudes |
| `geometry/camera.py` | 201 | pinhole intrinsics, posed camera, projection, back-projection, `look_at`, `moved` |
| `geometry/planes.py` | 334 | `Plane`, `Aperture`, ray–plane intersection, reflection, virtual camera pose, plane-induced homography, normalised DLT fit, homography residual |
| `geometry/__init__.py` | 71 | **the coordinate conventions, stated once** |
| **hypotheses/** | | |
| `hypotheses/base.py` | 262 | `OpticalMechanism` (closed 5-member enum), `Hypothesis` with strict validation / equality / hashing / serialisation, `HypothesisSet` |
| `hypotheses/families.py` | 117 | constructors for H_D/H_R/H_T/H_E/H_M, `phase1_hypothesis_set` |
| `hypotheses/__init__.py` | 44 | |
| **interventions/** | | |
| `interventions/actions.py` | 111 | `Action` (SE(3) delta in the reference camera frame), scaling, execution noise |
| `interventions/action_space.py` | 228 | bounded candidate generation, bound enforcement, lateral sweep, serialisation |
| `interventions/__init__.py` | 11 | |
| **optics/** — the analytical transition layer | | |
| `optics/base.py` | 273 | `HypothesisWorld`, the shared `project_world` renderer, aperture visibility, contact depth |
| `optics/direct.py` | 44 | H_D |
| `optics/mirror.py` | 79 | H_R + the virtual-camera formulation used for cross-validation |
| `optics/display.py` | 101 | H_E static and view-tracked + the homography formulation used for cross-validation |
| `optics/transmission.py` | 123 | H_T paraxial axial shift, exact slab lateral displacement, paraxial validity angle |
| `optics/mixed.py` | 56 | H_M (marked PRELIMINARY, unvalidated) |
| `optics/registry.py` | 32 | mechanism → transition lookup |
| `optics/__init__.py` | 50 | |
| **data/** | | |
| `data/types.py` | 224 | `GeometryFeature`, `Observation`, `SceneContent`, the 3-channel landmark layout |
| `data/dataset.py` | 183 | on-disk dataset reader with lazy `.npz` loading |
| `data/splits.py` | 69 | base-scene split policy + leakage detector |
| `data/synthetic/scene_generator.py` | 242 | interface placement, corridor content sampling, observer-marker rejection sampling |
| `data/synthetic/camera_generator.py` | 61 | intrinsics, random rig→world transform, pointing jitter |
| `data/synthetic/optical_variants.py` | 188 | hypothesis-set construction, reference observation, simulation |
| `data/synthetic/trajectory_generator.py` | 32 | action set, MCRB lateral sweep |
| `data/synthetic/ground_truth.py` | 167 | oracle separability, resolvability labels, three MCRB variants |
| `data/synthetic/dataset_writer.py` | 305 | generation loop, splat renderer, manifest with per-file SHA-256 |
| `data/synthetic/validator.py` | 135 | six scientific invariants |
| `data/external/registry.py` | 173 | external dataset registry + the download-refusal policy |
| **models/** | | |
| `models/interfaces.py` | 106 | the six `Protocol`s |
| `models/encoders.py` | 156 | `ground_truth`, `mock`; `moge`/`vggt_like` raise with instructions |
| `models/transition.py` | 197 | analytical, no-hypothesis-conditioning, hybrid, learned-only |
| `models/learned_transition.py` | 207 | 2-hidden-layer NumPy MLP + Adam, hand-written backprop |
| `models/separability.py` | 221 | the composite distance `D` and the separability estimator |
| `models/belief.py` | 105 | log-space likelihood update, entropy helpers |
| `models/identifiability.py` | 93 | `I_A` matrix, epsilon decision, belief-weighted score |
| `models/selector.py` | 213 | six intervention strategies |
| `models/baselines.py` | 228 | softmax regression, reference/response feature extractors |
| `models/__init__.py` | 66 | |
| **inference/** | | |
| `inference/engine.py` | 197 | the scientific loop; `AbstentionPolicy` |
| `inference/result.py` | 104 | `InferenceResult`; returns `"abstain"`, never a silent argmax |
| **metrics/** | | |
| `metrics/mcrb.py` | 203 | analytic MCRB with applicability enforcement, numeric MCRB, MAE |
| `metrics/classification.py` | 109 | CEA (all / committed / per-mechanism), FPCR, confusion matrix |
| `metrics/identifiability.py` | 117 | rank-based AUROC, exact ROC curve, decision metrics |
| `metrics/depth.py` | 66 | AbsRel, RMSE, δ<1.25 on contact depth |
| `metrics/regret.py` | 42 | intervention regret, motion cost |
| `metrics/aggregate.py` | 85 | Student-t confidence intervals, metric flattening |
| **experiments/** | | |
| `experiments/phase1.py` | 885 | the Phase-1 experiment: evaluation, metrics, figure data, tables, summary |
| `experiments/figures.py` | 232 | figure-data assembly and rendering (strictly separated) |
| `experiments/methods.py` | 133 | `MethodSpec` → engine or classifier |
| `experiments/runner.py` | 159 | CLI, run directory creation, dispatch, failure recording |
| `experiments/registry.py` | 58 | registry queries |
| **visualization/** | | |
| `visualization/ieee_style.py` | 210 | **the only rcParams in the repository** |
| `visualization/metric_plots.py` | 193 | CEA, ROC, FPCR, contact depth, MCRB validation/error, summary table |
| `visualization/ambiguity_plots.py` | 184 | initial-view similarity, separability matrix, posteriors, resolvability, uncertainty decomposition |
| `visualization/intervention_plots.py` | 157 | separability vs baseline, action utility, trajectory, regret, motion, predicted vs observed |
| `visualization/pipeline_figure.py` | 134 | the conceptual overview, matched-variant strip |
| `visualization/geometry_plots.py` | 111 | contact vs apparent, landmark views, renders, error maps |
| `visualization/ablation_plots.py` | 52 | ablation grid, generalisation curves |
| `visualization/export.py` | 55 | PDF + PNG export, figure index |
| **reproducibility/** | | |
| `reproducibility/run_dir.py` | 161 | immutable run directories, all metadata capture |
| `reproducibility/environment.py` | 142 | Python/OS/CPU/GPU/CUDA/packages/git/env-vars capture |
| `reproducibility/manifest.py` | 87 | run manifest finalisation, append-only registry |
| `reproducibility/seeds.py` | 53 | global seeding, independent named RNG streams |
| `reproducibility/hashing.py` | 31 | file and tree checksums |

### 2.2 Tests (13 files, 224 cases)

| File | Cases | Lines | Focus |
|---|---|---|---|
| `tests/unit/test_config.py` | 35 | 161 | inheritance, overrides, hashing, **every shipped config validates** |
| `tests/unit/test_visualization.py` | 27 | 143 | every plot family renders; grayscale-safety; no LaTeX needed |
| `tests/unit/test_metrics.py` | 26 | 214 | perfect / wrong / abstained / degenerate inputs per metric |
| `tests/unit/test_identifiability.py` | 19 | 204 | distance terms; known separable and **known non-separable** cases; ε boundary |
| `tests/unit/test_optics.py` | 18 | 293 | the two independent cross-validations; matched counterfactuals; contact geometry |
| `tests/unit/test_models.py` | 17 | 195 | encoders, four transitions, learned residual, classifiers |
| `tests/unit/test_geometry.py` | 14 | 140 | SE(3), projection, **the conventions themselves** |
| `tests/unit/test_reproducibility.py` | 14 | 158 | seed determinism, split stability, leakage, matched-property sweep |
| `tests/integration/test_pipeline.py` | 14 | 222 | full loop per mechanism; dataset round trip; corruption detection |
| `tests/unit/test_interventions.py` | 13 | 114 | bounds, rejection of out-of-bounds actions, candidate generation |
| `tests/unit/test_belief.py` | 12 | 102 | normalisation, stability at 1e6 errors, ordering, floor |
| `tests/unit/test_hypotheses.py` | 11 | 123 | validation, equality, "unidentifiable is not a mechanism" |
| `tests/smoke/test_smoke.py` | 4 | 145 | end to end; run immutability; **failed runs are recorded** |

Roughly a dozen of these encode **scientific** invariants rather than code
behaviour — that a view-tracked display is provably unresolvable, that
separability is exactly zero under the null action for every scene, that a
classifier on identical features cannot beat chance. Those are the ones that
would catch a silent change to the research question.

### 2.3 Scripts, configs, docs

**Scripts (8 + bootstrap):** `setup_environment.sh`, `download_datasets.sh`,
`validate_datasets.py`, `generate_synthetic_data.py`, `run_smoke_test.sh`,
`run_experiment.py`, `aggregate_results.py`, `generate_all_figures.py`,
`_bootstrap.py`.

**Configs (10):** `default.yaml`, `smoke_test.yaml`,
`synthetic/{smoke,phase1,benchmark_core}.yaml`,
`experiments/phase1_problem_existence.yaml`,
`models/{analytical,hybrid,foundation_encoder}.yaml`, `datasets/external.yaml`.

**Docs (10 + README, 2,140 lines):** `RESEARCH_SPEC_AUDIT` (272),
`SOFTWARE_ARCHITECTURE` (226), `EXPERIMENT_PLAN` (224), `REPRODUCIBILITY` (209),
`TROUBLESHOOTING` (163), `LITERATURE_CROSS_RESEARCH` (142), `DATASET_MATRIX` (141),
`DEVELOPMENT_ROADMAP` (131), `NOVELTY_RISK_REGISTER` (117),
`VISUALIZATION_PLAN` (102), plus this file.

**Root:** `README.md`, `LICENSE` (MIT), `pyproject.toml`, `requirements.txt`,
`requirements-dev.txt`, `requirements-gpu.txt`, `environment.yml`, `Makefile`,
`.gitignore`, `.pre-commit-config.yaml`, `notebooks/README.md`.

---

## 3. Build chronology

1. **Read the plan and all 2,549 lines of `research(1).md`.**
2. **Environment probe.** Found a bare Python 3.13 with *no* packages. Created
   `.venv` via `uv`, installed numpy / matplotlib / pyyaml / pytest /
   scienceplots. Confirmed SciencePlots' IEEE style works with `no-latex` — the
   plain `science` style sets `text.usetex=True` and would fail without a LaTeX
   install.
3. **Literature verification (§7).** Fetched and checked the novelty-critical
   citations before writing any audit prose.
4. **Core library, bottom-up:** utils → geometry → hypotheses → interventions →
   optics → data types.
5. **Physics validated interactively** before proceeding: confirmed all variants
   pixel-identical at `C_0`, contact depths differing by mechanism, and a
   sensible `I_A` matrix. This is where defects D1–D3 surfaced.
6. **Models, metrics, inference, config, reproducibility, visualization.**
7. **Synthetic generator**, then the experiment layer.
8. **Scripts and configs**, then the first smoke-test run.
9. **Tests written per subsystem**, each run immediately.
10. **Phase 1 executed**, three seeds; defects D4–D6 surfaced here and during the
    final verification.
11. **Documentation written last**, from measured numbers — never from
    expectations.
12. **Final pass:** ruff clean, 224 tests, clean-state smoke test, full
    regeneration of the benchmark, all three seeds and all figures.

---

## 4. Defects found and fixed during the build

This is the section with the most information in it. Each of these was a real
bug that would have produced wrong or misleading results.

### D1 — `H_D` had no occluding aperture → an unintended discriminative cue

**Symptom.** `I(H_D, H_R)` measured ≈ 26 px-equivalent when the only physical
difference between those hypotheses should have been the observer-marker channel
(worth 12).

**Cause.** `direct_hypothesis()` was constructed with `interface=None`, so H_D's
content was visible everywhere while the mirror's content was clipped by the
aperture. Aperture clipping alone therefore betrayed the mechanism, and no
reasoning about optics was required.

**Fix.** `Hypothesis` now permits `OpticalMechanism.DIRECT` to carry an
interface, interpreted as a purely **occluding opening** (a doorway or window
frame) rather than an optical surface — contact geometry stays on the content.
The matched benchmark gives every variant the same opening.

**Locked in by** `test_direct_may_carry_an_occluding_opening`,
`test_all_mechanisms_are_pixel_identical_at_the_reference_view`.

### D2 — the occlusion distance term was diluted by the landmark count

**Symptom.** A mirror whose three virtual markers were plainly visible scored
`Δ = 0.68`, below the `ε = 1.0` threshold, and was labelled non-identifiable.

**Cause.** `D_occlusion` was a *fraction*: `3 / 47 = 0.064`, times `λ_o = 8`,
gives `0.51`. Whether a decisive appearance crossed the threshold therefore
depended on how many landmarks the scene happened to contain.

**Fix.** `D_occlusion` is now a **count** of visibility disagreements, with
`λ_o = 4.0` pixel-equivalents per disagreement. One unambiguous
appearance/disappearance is now worth 4 px against a 1 px threshold, regardless
of scene size.

**Locked in by** `test_occlusion_term_counts_disagreements_not_fractions`, which
asserts a single disagreement is worth the same in a 5-landmark and a
50-landmark scene.

### D3 — aperture size was specified in metres, independent of the camera

**Symptom.** Only 33 of 40 content landmarks were visible at the reference view;
apertures were routinely wider than the field of view.

**Cause.** `aperture_half_width` was an absolute range in metres, unrelated to
focal length or resolution.

**Fix.** Replaced with `aperture_half_width_frac`, a fraction of the image
half-width projected onto the interface plane. `1.0` exactly fills the frame.
The difficulty knob is now resolution- and focal-length-independent.

### D4 — the config hash included the seed, silently breaking multi-seed aggregation

**Symptom.** Three Phase-1 seeds produced three *different* config hashes
(`bb8db53b`, `41907923`, `5dbaf8e5`). `aggregate_results.py` refuses by design to
average across differing configurations, so it would have reported **a single run
as if it were the whole three-seed study** — exactly the "one lucky seed" failure
the plan warns against.

**Cause.** `experiment.seed` was part of the hashed configuration.

**Fix.** `HASH_EXCLUDED_PATHS = ("experiment.seed",)`. The hash now identifies the
experimental *configuration*; the seed is recorded separately in the run
directory name, the run manifest and the registry.

**Locked in by** `test_config_hash_is_seed_independent`.

### D5 — the reference view was *forced* to match rather than genuinely matching

**Symptom (the most serious one).** The smoke test reported a median MCRB of
exactly `0.0000 m` — meaning many scenes were separable at zero baseline, i.e.
without moving at all.

**Cause.** `reference_observation` rendered *every* variant using the **direct**
world, which suppressed the mirror's own virtual markers at `C_0`. The simulator
therefore showed no markers at the reference view while the model *predicted*
markers there. Two consequences, both bad:

- the matched-counterfactual property was imposed by fiat rather than earned, so
  the simulator was physically inconsistent with its own optics;
- `Δ(H_D, H_R)` was non-zero under the **null action**, so "separable without
  intervention" was being recorded as a resolving baseline of zero.

**Fix.** Two changes.
1. `reference_observation` now renders from the variant's **own** world, so if a
   mirror would show the observer's reflection at `C_0`, it does. Only the
   perceived *depth* is overridden to the apparent reading — which encodes the
   documented empirical finding that a single-frame model reports the illusory
   depth, and prevents the depth channel from betraying a display to a
   single-frame classifier.
2. `generate_base_scene` now **rejection-samples** the observer rig: any draw
   whose virtual image would be visible at `C_0` is rejected. This models a
   careful experimenter choosing a reference viewpoint from which their own
   reflection is out of frame.

**Effect on results.** Median MCRB moved from `0.0000` to `0.0229 m`, and the
non-identifiable fraction became a genuine geometric property.

**Locked in by** `test_no_scene_reveals_a_mirror_at_the_reference_view`,
`test_separability_is_exactly_zero_under_the_null_action_for_every_scene` (swept
over 12 scenes rather than one), and
`test_no_generated_scene_ever_violates_the_matched_property` (swept over 5 seeds
× 8 scenes).

### D6 — the rejection-sampling fallback could not move a near-axis rig

**Symptom.** After D5's fix, one scene in eight still violated the matched
property (`smoke_pytest` base scene 7), and the corresponding pytest smoke test
failed with `max reference-view pixel deviation = inf`.

**Cause.** When resampling failed, the fallback pushed the rig outward
**multiplicatively** (`markers[:, :2] *= 1.05`). A rig sitting near the optical
axis — say a lateral offset of 0.01 m — stays near the axis no matter how many
times it is multiplied.

**Fix.** The fallback now displaces the rig **additively** in 0.05 m steps along
the sign of its current offset, and **raises a `RuntimeError`** if it still
cannot succeed, rather than emitting a silently broken scene.

**Verified** across `configs/synthetic/{smoke,phase1}.yaml` and all three
multi-seed variants: **0 violations in 300 base scenes.**

### Also fixed during the final pass

- **Separability-vs-baseline figure** reserved half its height for a negative
  region no datum can occupy (symlog axis). Clipped at zero.
- **Pipeline overview diagram**: the dashed feedback arrow crossed through the
  boxes (`rad=0.28` bulges the wrong way). Rerouted below (`rad=-0.35`); two
  labels shortened to stop them overflowing their boxes.
- **CEA figure**: a 9-entry legend covered the bars it was explaining. Moved
  below the axes, short method names added, zero-height bars annotated `0`
  instead of appearing as unexplained gaps.
- **Uncertainty-decomposition figure**: points coincided exactly, so a cluster of
  40 scenes looked identical to a single one. Markers now sized and annotated by
  multiplicity.
- **Smoke test** printed `SMOKE TEST FAILED` after passing — the `trap` misread
  `$?`. Replaced with an explicit success flag.
- **`fontTools`** logged every glyph of every PDF at INFO, burying the
  experiment's own log. Silenced along with `matplotlib` and `PIL`.
- **`ruff`**: 78 findings → 0. Fixes included splitting semicolon-joined Adam
  updates in the classifier, narrowing an over-broad `pytest.raises(Exception)`
  to `FileNotFoundError`, and adding explicit `strict=` to 12 `zip()` calls so a
  length mismatch fails loudly instead of truncating silently. `E741` is ignored
  by policy: `I` is the identifiability matrix `I_A` from the specification, and
  renaming it to satisfy a linter would make the code harder to check against the
  maths.

---

## 5. Tuning decisions, and the evidence for each

Every one of these was measured, not guessed.

### 5.1 Observer-marker rig offset → benchmark difficulty

The single most consequential knob: it sets what fraction of mirror scenes are
identifiable. Swept over 40 base scenes:

| `rig_offset_lateral` | H_D | H_E | H_R | overall resolvable |
|---|---|---|---|---|
| ±2.5 m | 40 % | 80 % | 48 % | 56 % |
| **±1.8 m (chosen)** | **55 %** | **80 %** | **68 %** | **68 %** |
| ±1.4 m | 65 % | 80 % | 85 % | 77 % |
| ±1.1 m | 78 % | 80 % | 92 % | 83 % |

**Chose ±1.8 m.** Wider ranges push every mirror permanently outside the frustum,
inflating the non-identifiable fraction for an uninteresting reason (the virtual
image is simply never in view); narrower ranges make almost everything solvable
and remove the phenomenon the benchmark exists to study. ±1.8 m places the
virtual image near the aperture boundary, which is the regime where *whether a
bounded action can reach it* is a genuine question.

### 5.2 Content depth parameterisation → the MCRB ladder

Originally `content_depth_near` / `content_depth_far` were sampled
independently, which gave scenes with a large depth offset behind the interface
and hence operational MCRBs of 1.6–25 mm — **below the smallest action step
(50 mm)**, so every display resolved at the first action and MCRB was
uninformative.

Replaced with `content_depth_near` + `content_depth_spread`. Since
`B_min ∝ 1/|1/Z₁ − 1/Z₂|`, sampling the *spread* directly controls the resolving
baseline. Result: a genuine ladder with median MCRB `0.023 m`, comparable to the
action bounds, and some scenes beyond them.

### 5.3 FPCR threshold `τ = 0.45`

At the default `τ = 0.8`, FPCR read `0.000` for **every** method — but for the
wrong reason. With a three-hypothesis family whose residual ambiguity is two-way,
a forced argmax over an unresolvable pair sits at exactly `p = 0.5`, so any
`τ ≥ 0.5` scores that case as "not confident" and the metric silently stops
measuring anything.

The sweep made this visible:

| method | τ=0.40 | τ=0.50 | τ=0.60 | τ=0.80 |
|---|---|---|---|---|
| ours, forced choice | 0.727 | 0.000 | 0.000 | 0.000 |
| passive multi-view | 1.000 | 0.636 | 0.182 | 0.000 |
| **ours** | **0.000** | 0.000 | 0.000 | 0.000 |

`τ = 0.45` sits just below the tie, so a forced choice over a genuine tie counts
as false certainty — which is precisely what FPCR exists to detect. `metrics.json`
records the full sweep so the choice is never load-bearing.

### 5.4 Multi-seed protocol

The first three-seed run produced `± 0.000` on almost every metric. That was
technically true and scientifically empty: the benchmark was fixed and the
methods are deterministic, so the seed varied nothing that mattered.

Added `data.reseed_with_experiment_seed`, which **redraws the benchmark for each
experiment seed**. The Phase-1 config now uses it, so the reported confidence
intervals measure variation over the data-generating process. Both protocols are
documented, along with what each one's error bars mean.

### 5.5 Distance weights

`λ_motion = 1.0` (pixels, the unit the MCRB theory is written in),
`λ_occlusion = 4.0` (px-equivalents per disagreement, see D2),
`λ_geometry = 0.0` (partly redundant with motion, unvalidated at this stage),
`λ_feature = 0.0` **enforced** — a non-zero value raises rather than silently
contributing zero.

---

## 6. Deliberate non-implementations

Each of these is a decision, not an omission, and each fails loudly rather than
degrading silently.

| Component | Status | Why, and what happens if you ask for it |
|---|---|---|
| `moge`, `vggt_like` encoders | **NOT IMPLEMENTED** | Requires verifying repo, licence, checkpoint, compatibility, GPU memory. `encode()` raises `NotImplementedError` with a six-step installation checklist. No run can claim a foundation model it did not use. |
| `D_feature` | **NOT IMPLEMENTED** | Needs a real geometry-foundation-model feature space. `DistanceWeights(feature≠0)` raises; the config validator rejects it too. |
| Mutual-information action selection (`argmax_a I(H;O'\|a)`) | **NOT IMPLEMENTED** | Predictions are deterministic in the preliminary version, so the mutual information is degenerate. Faking it would have been worse than omitting it. The belief-weighted separability form is used instead. |
| `H_M` mixed optics | implemented, **unvalidated** | Marked PRELIMINARY in its own module docstring and excluded from Phase 1. |
| Learned/hybrid transitions | implemented, **never trained in a run** | The MLP is real, tested and deterministic, but it is a Gate-7 placeholder operating on landmarks, not a world model. Requesting it without a trained model raises with a clear message. |
| Multi-step intervention | implemented, **untested at `max_steps > 1`** | Phase 1 uses one step. |
| External datasets | **none downloaded** | `download_datasets.sh` auto-downloads only when licence *and* permission are both `verified`; no registered dataset meets that bar, so it prints instructions for all 13. |
| Parquet predictions | replaced with CSV | Avoids a pandas/pyarrow dependency for no benefit at this scale. Documented as a deviation. |
| Ablation / generalisation figures | plot code written and unit-tested, **not wired into a run** | The corresponding experiments have not been executed, so there is nothing honest to plot. |

---

## 7. Literature verification actually performed

Fetched and confirmed against official pages (arXiv abstract pages, CVF Open
Access, GitHub, Hugging Face) — **not taken from the source document on trust**:

GIFT (arXiv:2608.02068) · LayeredDepth (arXiv:2503.11633, ICCV 2025) ·
SeeGroup (arXiv:2605.28735, CVPR 2026 Oral) · GLINT (arXiv:2603.26181, CVPR 2026
Oral + Award Candidate) · DepthFocus (arXiv:2511.16993) · 3D Visual Illusion
(arXiv:2505.13061, NeurIPS 2025) · MD-3k (arXiv:2606.29600) · TransPhy3D /
DKT (arXiv:2512.23705) · VGGT-World (arXiv:2603.12655) · Mirror3D
(arXiv:2106.06629) · ClearPose (arXiv:2203.03890) · DREDS (CC BY-NC 4.0
confirmed) · Booster (arXiv:2301.08245) · CITRIS · NBV for reflective objects
(arXiv:2202.13263).

**Outcome: no fabricated citation found among those checked.**

**Two findings the source document does not contain:**

1. **The 3D Mirage (arXiv:2512.15423)** — Nguyen, Xu, Huang; submitted Dec 2025,
   revised Jul 2026. Probes and *tames* 3D hallucination on planar, perceptually
   ambiguous inputs with a benchmark, metrics (DCS, CCS) and Grounded
   Self-Distillation. **The closest work to Intervene3D's motivation, and it is
   missing from the source document's related-work list.** Recorded as risk R4.
2. A near-miss: the first search for "GIFT" surfaced arXiv:2408.06083 (*Towards
   Robust Monocular Depth Estimation in Non-Lambertian Surfaces*) instead.
   Fetching the cited ID directly confirmed GIFT is real, but the two are close
   neighbours and a reviewer may raise the 2024 one as additional prior art.

Targeted searches for `"interventional identifiability"` / `"action-set
identifiability"` combined with camera motion, competing image-formation
hypotheses and abstention returned **nothing overlapping**. Recorded with the
explicit caveat that absence of evidence in a targeted search is weak evidence of
absence.

---

## 8. Commands actually executed (not merely documented)

| Command | Result |
|---|---|
| `bash scripts/setup_environment.sh --check-only` | passes; reports CPU-only, torch absent |
| `bash scripts/run_smoke_test.sh` | **PASSES** from clean state, all 17 required steps |
| `pytest tests -q` | **224 passed in 8.6 s** |
| `ruff check src scripts tests` | **All checks passed** |
| `generate_synthetic_data.py --config configs/synthetic/phase1.yaml` | 288 variants, 96 base scenes, 103 non-identifiable (35.8 %), validation PASSED |
| `generate_synthetic_data.py --dry-run` | validates and writes nothing |
| `run_experiment.py --config .../phase1_problem_existence.yaml --seed {1,2,3}` | 3 successful runs, ~28 s each |
| `aggregate_results.py --experiment phase1_problem_existence` | 3 runs, mean ± std ± 95 % CI |
| `aggregate_results.py --list` | registry summary |
| `generate_all_figures.py` | 176 files regenerated from saved data alone |
| `validate_datasets.py --all` | 13 external datasets reported `NOT DOWNLOADED` |
| `validate_datasets.py --list` | registry with licence status |
| `download_datasets.sh --dataset dreds` | correctly **refuses**, prints instructions |
| `download_datasets.sh --list` | registry listing |
| `make help` | 12 documented targets |

**Artefacts on disk:** 5 generated datasets (`phase1`, `phase1_s1/s2/s3`,
`smoke`), 4 successful runs in the registry, 22 figures × 2 formats per run.

---

## 9. What the README covers, and what it does not

**In the README:** motivation, research questions, the formulation, architecture,
repository structure, installation, smoke test with real expected output, the
synthetic benchmark, dataset acquisition, running experiments, the Phase-1 result
table with all nine methods, where results live, reproducibility, nine known
limitations, the novelty position, the documentation index, licence.

**Not in the README, and now recorded here instead:**

| Not in README | Where it is now |
|---|---|
| The six defects found and fixed during the build | §4 of this file |
| The tuning sweeps and the evidence behind each parameter | §5 |
| The build chronology | §3 |
| The file-by-file inventory with line counts | §2 |
| Which literature was verified and how | §7 (summary in `LITERATURE_CROSS_RESEARCH.md`) |
| The full list of commands actually executed | §8 |
| Why specific components were *not* implemented | §6 |
| Figure-level layout fixes | §4, final pass |
| The lint clean-up and its rationale | §4, final pass |

**Partial overlaps.** Some of this material exists in `docs/` but in a different
form and for a different purpose: `RESEARCH_SPEC_AUDIT.md` §5 lists the seven
*specification* ambiguities (which are properties of the source document), not
the six *implementation* defects listed here (which were bugs in my code).
`DEVELOPMENT_ROADMAP.md` records which gates passed, but not what broke on the
way. This file is the only place the development history is written down.

**One correction to a claim in the README's own framing:** the README describes
the three specification corrections (mirror parallax, anchored vs compensated
MCRB, mutual-information selection) as "documented not hidden", and they are —
in `RESEARCH_SPEC_AUDIT.md` §5. But it does not mention that D1–D6 above were
found *by me, in my own implementation*, during the build. That distinction
matters for anyone auditing this work, and it is why this file exists.

---

## 10. Honest status

**What is established:** the pipeline runs end to end from a clean environment;
the benchmark's matched-counterfactual property is exact and exhaustively tested;
the analytical mirror and display optics are cross-validated against independent
derivations; single-frame classification sits at exactly chance; intervention
strictly improves on passive in every seed; abstention drives false physical
certainty from 0.810 to 0.000; and the MCRB scaling law holds at `R² = 0.83–0.91`.

**What is not:** the simulator shares its forward optics with the transition
model, so oracle-encoder AUROC = 1.000 and MAE_MCRB ≈ 0.001 are **degenerate** —
they confirm internal consistency, not generalisation. Interface parameters are
assumed known. No real imagery has been touched. Those limits are stated in the
README, in `RESEARCH_SPEC_AUDIT.md` §9, and in every run's `summary.md`.
