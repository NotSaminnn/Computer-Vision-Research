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

---

## 11. Session 2 — 2026-08-29: environment on Windows, and real dataset acquisition

A second session, on a different platform (Windows 11, Python 3.10.11, `uv`),
with two goals: stand the environment up from the clean checkout, and turn the
external-dataset registry from a documentation artefact into working acquisition.

### 11.1 Environment

`uv venv` + `requirements-dev.txt` + `pip install -e .`, then the repository's own
verification. Everything passed on the first attempt except the shell scripts'
interpreter discovery, which assumed `.venv/bin/python`; a Windows virtualenv puts
it in `.venv/Scripts/python.exe`.

| File | Change |
|---|---|
| `scripts/setup_environment.sh` | `venv_python()` resolves either layout; base interpreter falls back to `python` where `python3` does not exist; the closing banner now prints the paths that actually exist on the running platform |
| `scripts/run_smoke_test.sh`, `scripts/download_datasets.sh` | same discovery order, `PYTHON=` still overrides |
| `Makefile` | `PY ?=` picks `Scripts/python.exe` when present |

Verified on this machine: **276 tests pass**, `ruff check src scripts tests`
clean, the 16-step smoke test passes end to end (22 PDF + 22 PNG figures, all
regenerated from `metrics/figure_data.json` alone), and all three synthetic
benchmarks regenerate and validate — `phase1` reproducing **288 variants, 103
non-identifiable (35.8 %)**, matching the figure recorded in session 1.

### 11.2 The registry was under-claiming, and it mattered

Session 1 recorded eight entries as `ACCESS UNVERIFIED` because the project pages
did not state a licence. That was the correct call given what was checked, but
the check was aimed at the wrong target: the machine-readable source
(`/api/datasets/<id>`) states licence and gating explicitly, and the file tree
endpoint gives exact per-file byte counts. Re-verifying against those:

| Dataset | Recorded 2026-08-28 | Actual, 2026-08-29 |
|---|---|---|
| LayeredDepth | licence CC0 `verified`, auto-download `unknown` | `cc0-1.0`, ungated — **fetchable** |
| LayeredDepth-Syn | not a separate entry | `bsd-3-clause` — a **different licence** from the benchmark repo, so it cannot share a row |
| 3D Visual Illusion | licence not stated | `apache-2.0`, ungated |
| TransPhy3D | licence not stated | `apache-2.0`, ungated |
| ClearPose | licence not stated | **MIT**, per the repository README |

Two findings that would have caused silent damage if the data had simply been
pulled and used:

1. **3D Visual Illusion's 455 GB training split has no ground-truth depth.** Its
   `depth` files are DepthAnythingV2 *predictions* rescaled by the provided
   `scale_factors*.csv`. Training on it would import another model's errors as
   labels. Only the 8.10 GB RealData split carries true metric disparity
   (RealSense L515). The registry wires up RealData only.
2. **LayeredDepth's 15.15 GB test split cannot be scored locally** — labels are
   withheld for a submission server, three submissions per user per seven days.
   The 4.13 GB validation split is the one that supports a local metric.

`ClearPose` (MIT) and `DREDS` (CC BY-NC 4.0) stay **manual** despite verified
licences: both are distributed via hosts that expose no stable, checksummable
URL, so an automated fetch could not be made reproducible. Permission is about
mechanism as much as licence, and the registry now separates the two.

### 11.3 `data/external/fetchers.py` — 380 lines, no new dependency

The previous `download_datasets.sh` printed
`NOT IMPLEMENTED: no registered dataset currently satisfies the automated-download
condition`. Four now do, so the fetcher was written. Pure standard library —
adding `huggingface_hub` for four registry entries would have cost the
"NumPy-only" property for nothing.

Order of operations, enforced in code rather than documented:

1. re-check policy at fetch time (licence `verified` **and** permission `true`);
2. **list before transferring** — exact file count and byte total printed first;
3. hold anything over 1 GB until `--yes`, and refuse if free disk < size + 10 %;
4. transfer, resuming partial files;
5. verify against **the publisher's own** SHA-256 (Hugging Face `lfs.oid`), so
   integrity is checked against the source rather than against whatever arrived;
6. write `manifest.json`: per-file SHA-256, **resolved commit SHA**, source URL,
   allow-patterns, licence.

Step 6 is what makes an external result reproducible — the manifest pins an
immutable revision, so a future claim reads
`princeton-vl/LayeredDepth@a2aad776030144950f8cbc2f12e2903b26316ff8`, not a
moving `main`.

**Variants.** A dataset can publish splits four orders of magnitude apart
(`layereddepth`: 0.07 GB → 15.15 GB; `transphy3d`: 0.75 GB → ~1.5 TB). Each entry
declares named variants and `--variant` selects one. TransPhy3D deliberately
exposes only `sample` and `test`; its training set is not fetchable through this
tool at all.

### 11.3.1 Five defects found by running it for real

Only the first two were visible from reading the code; the rest needed an actual
multi-gigabyte transfer against a live host.

**1. `expand=true` on the tree endpoint caps a page at 50 entries** instead of
1000, while returning the same `lfs.oid`. On TransPhy3D's 11,149 files that is
223 round trips for no added information. Removed.

**2. One connection per file saturates nothing when a variant is one large
tarball.** 3D Visual Illusion ships 8.1 GB as a single file, so `--workers 6` ran
at one stream. Spare workers are now spread across byte ranges of the same file,
each range its own resumable `.partN`, with a fallback to single-stream if the
server ignores `Range`.

**3. `local_root` turned `expected_layout` into a directory named `...`.**
Pre-existing, and the worst of the five. The property was

```python
repo_root() / expected_layout.split("{")[0].rstrip("/")
```

correct for `data/raw/layereddepth/{validation,test}/...` but not for
`data/raw/transphy3d/...` — with no `{`, the trailing `...` survived as a literal
path component. **Nine of the fourteen entries were affected** (`transphy3d`,
`clearpose`, `booster`, `md3k`, `magd`, `mvmd`, `gift_benchmark`, `pdi_dataset`,
`depthfocus_synth`), and it stayed invisible until something actually wrote to
one: the fetch reported `downloading to data/raw/transphy3d/.../sample`.
`local_root` now drops every all-dots component. The regression test asserts that
no shipped entry resolves to a path containing `...` and that each stays directly
under `data/raw/`.

**4. A variant could not override its repository.** `plan_fetch` read
`fetch.repo_id` only, so 3D Visual Illusion's `scale_factors` variant — which
lives in the *virtual* repo while `real` lives in the *Real* repo — would have
silently fetched from the wrong one. Per-variant `repo_id` / `repo_type` now win.

**5. The listing had no retry while the transfer did.** TransPhy3D needs ~12
paginated calls and the Hub closed one: `WinError 10054`. The whole fetch aborted
on a transient network event, having already passed every policy check. API calls
now retry five times with exponential backoff — but **not** on 401/403/404, where
the server has given a definitive answer and retrying would be both useless and
rude. `_MAX_LIST_PAGES` stops a runaway cursor.

A sixth issue was self-inflicted, and is worth recording as method rather than as
a defect: the first batch run silently skipped 3D Visual Illusion because
`external.yaml` was being edited *while* the fetch loop was reading it, so
`ExternalRegistry()` parsed a half-written file. The failure was invisible because
the run's `grep` filter matched `DONE|FAILED` but not `could not list the remote
repository`. Two lessons — do not edit a config a running job reads, and **a
progress filter that only matches known-good lines cannot tell failure from
silence**.

### 11.4 Tests

`tests/unit/test_external_datasets.py`, **20 cases**, all offline. The policy is
the part worth testing, and it must hold without a network:

- refusal on unverified licence, on withheld permission, on a `verified` entry
  with no `fetch:` block, on an unimplemented backend, on an unknown variant;
- an invariant over the *shipped* registry — no entry may claim
  `automated_download_permitted: true` without `licence_status: verified`, and
  every permitted entry must actually plan;
- the 1 GB hold;
- transfer mechanics against a local `Range`-capable HTTP server: segmented
  reassembly is byte-exact, a half-written segment resumes rather than
  duplicating, the no-`Range` fallback works, a truncated response is rejected
  rather than accepted, and a complete file is not re-fetched;
- manifest checksums detect post-hoc corruption;
- regressions for all four code defects in §11.3.1 — no shipped entry resolves to
  a path containing `...`, a per-variant `repo_id` wins over the entry default, a
  dropped listing request is retried, the retry budget is finite, and a 404 is
  *not* retried.

Total: **224 → 276 tests.**

### 11.5 What was acquired

~13.8 GB into `data/raw/`, all verified against publisher checksums, all with
manifests. **None of it has entered any reported result** — every number in
`EXPERIMENT_PLAN.md` still comes from the synthetic benchmark alone. It is
staged for Gate 5, not used.

---

## 12. Session 3 — 2026-08-29: Gates 5-7, and a number that would have shipped

### 12.1 What was built

| Gate | Module | State |
|---|---|---|
| **5** — external loaders | `data/external/loaders.py` | four formats: LayeredDepth parquet + ordinal `tuples.json`; LayeredDepth-Syn parquet + 8 depth layers; TransPhy3D WebDataset tars + per-frame extrinsics; 3D Visual Illusion `tar.gz` stereo + `.pfm` |
| **6** — real encoder | `models/foundation_encoders.py` | Depth-Anything-V2-Small (Apache-2.0) on GPU. **24 ms/image, 0.35 GB VRAM, r² = 0.9797** against TransPhy3D depth after scale/shift alignment |
| **7** — torch training | `models/torch_transition.py`, `scripts/train_transition.py` | residual-vs-rigid-reprojection on frame pairs with known relative pose |
| — | `experiments/learned.py` | the NumPy training stage `methods.py` had promised via a config key nothing read |

The Base and Large Depth-Anything checkpoints are **CC-BY-NC-4.0**; only Small is
Apache-2.0, so Small is the default and selecting another records the restriction
in the run manifest.

### 12.2 Review before running, and what it caught

Two independent reviews ran before the first training: a domain expert on the
geometry, and an integrity audit on the claims. Both returned **NO-GO**. The
expert quantified the cost:

| pipeline | rigid baseline | model | reported ratio |
|---|---|---|---|
| **as written** | 0.3735 | 0.0697 | **0.187** |
| both bugs fixed | 0.1255 | 0.0451 | 0.359 |

It would have printed **0.187** — "the learned residual explains 81 % of what
rigid reprojection cannot" — and the number was an artifact.

**Four defects, each fixed and each now pinned by a test:**

1. **The relative pose was applied inverted.** `iter_pairs` returned
   `inv(E0) @ E1` and `warp_depth` then applied `inv(T_rel)`. The extrinsics are
   **camera-from-world**, so the correct point transform is `E1 @ inv(E0)`,
   applied directly. Two independent proofs: under the correct reading the camera
   centre height is constant at `0.5753 ± 0.0000 m` (an exact level turntable,
   3°/frame), and the dominant table-plane normal resolves to
   `(0, cos 15°, sin 15°)` — the constants that appear in the extrinsics
   themselves. Under the shipped reading, neither holds. **The shipped warp
   scored worse than not moving the camera at all**, which is the diagnostic that
   makes this catchable at all: a wrong pose still trains and still shows a
   falling loss.
2. **Intrinsics de-normalised per-axis**, giving `fx = 605.7`, `fy = 454.3` — a
   4:3 pixel aspect no Blender render has. Both focals normalise by width; only
   the principal point is per-axis. World-normal spread across frames:
   `5.92° → 0.13°`.
3. **`VisualIllusionReader` collapsed 455 samples into 28.** Frame names repeat
   across 83 scene directories (`frame_0000`…`frame_0027`), and the key was the
   bare stem. Every emitted sample spliced a left image from one scene onto a
   right image and disparity from another, and no mask ever attached. Keyed on
   `scene/frame`: **455 samples, 83 scenes, 455/455 masks**.
4. **The sequence split leaked 3.06 %.** Group labels were rebuilt positionally
   from a parallel list, but pairs yielding no rows are skipped, so every label
   after the first skip shifted. Labels now travel with their rows out of
   `build_pair_dataset`.

### 12.3 The finding neither review predicted

Occlusion boundaries carry **100.0 %** of the loss: MSE 0.794 on the 17 % of
pixels with `|z_obs − z_rigid| > 5 cm`, against `2.9 × 10⁻⁵` on the rest — a
factor of 27,000. On the geometrically consistent pixels, where transparency and
reflection actually live, the model was **73× worse than predicting zero**, i.e.
worse than the `H_D` null hypothesis it exists to beat.

Occlusion and optical anomaly both produce large residuals and **no magnitude
threshold separates them** — gating would delete the signal. So rows are *tagged*
rather than dropped, and every metric is reported on both subsets separately with
the pooled figure explicitly labelled `_POOLED`. An occlusion detector can no
longer masquerade as an optics model.

### 12.4 The first training run

`transphy3d/test`, 11,264,000 samples from 2,750 frame pairs across 25 sequences,
6 sequences held out, 80 epochs on an RTX 5070 (124 s). Deterministic: a second
run reproduced it bit-identically.

| subset | rows | ratio vs rigid |
|---|---|---|
| pooled | 2,469,888 | 0.719 |
| **consistent** | 2,416,960 (97.9 %) | **2956.7** |
| occlusion-affected | 52,928 (2.1 %) | 0.634 |

**The pooled number reads as a 28 % win and is worthless.** The model learned only
to predict occlusion.

The reason is a property of the data, and it is the most useful thing this run
produced: target RMS on consistent pixels is `0.00095 1/m` against a 16-bit depth
quantisation floor of `0.00089 1/m`. **They are the same number — there was
nothing to learn.** TransPhy3D stores *rendered geometric depth*: Blender writes
the true surface distance for a transparent object and does not simulate what a
stereo or ToF sensor reports looking *through* glass. The transparency is in the
RGB and absent from the depth, and depth is the only channel this model reads.

Consequence for the plan: rendered depth cannot supervise an optical-anomaly
transition. 3D Visual Illusion Real can — its disparity comes from a physical
sensor observing mirrors and screens, so the anomaly is in the measurement rather
than absent from it.

### 12.5 Smaller corrections in the same pass

- `learned.py` pooled an MSE over channels in **different units** (px, px, m);
  the pixel terms dominated by ~10⁶ so the depth channel was never learned.
  Now reported per channel.
- The degenerate-target flag used exact float equality, so a base matching the
  simulator to rounding rather than bit-for-bit would have slipped through.
- `_rotvec` silently returned "no rotation" at exactly 180°, reachable at
  `--stride 60` given the 3°/frame orbit.
- Depth bit-depth was inferred from pixel *values* rather than dtype — a 16-bit
  frame with all codes below 256 would have been scaled 257× too large.
- Corrupt frames (`max_depth == 0`, 5 of 600) were counted anonymously as
  "skipped"; they are now named.
- `LayeredDepthSynReader` compacted absent layers, silently renumbering a
  *ray-ordered* stack. It now records which layer indices are present.
- Provenance: the run wrote `dataset_manifest.json` saying "NOT RUN — no dataset
  attached" while having trained on 4 GB of it, and a reproduction command
  pointing at the wrong script. Both fixed; runs now pin
  `Daniellesry/TransPhy3D@3b023eb8`.

**Tests: 261 → 276**, including a real-data guard that fails if reprojection ever
stops beating the identity warp.
