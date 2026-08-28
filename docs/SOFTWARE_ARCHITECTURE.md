# Software Architecture

**Design goal:** the smallest rigorous system that can falsify or support the
Intervene3D hypothesis, structured so that larger models, datasets and
experiments plug in without restructuring the repository.

---

## 1. Conventions (fixed repository-wide)

These are stated once here and in `intervene3d.geometry.__init__`, and everything
else obeys them. A convention bug is the most expensive kind, so they are covered
by `tests/unit/test_geometry.py`.

- **World frame:** right-handed, metres.
- **Camera frame:** OpenCV — `+x` right, `+y` down, `+z` forward.
- **Pose:** stored as `T_wc` (camera-to-world). Camera centre is `T_wc[:3,3]`.
- **Projection:** `u = fx·x/z + cx`, `v = fy·y/z + cy`, valid only for `z > 0`.
  Points behind the camera yield `NaN` and `visible = False`, never an exception —
  hypothesis-conditioned prediction routinely evaluates counterfactual geometry
  that falls behind the camera.
- **Action:** a delta in the **reference camera frame**: `T_wc_new = T_wc_ref · Δ`.
  Lateral translation is therefore motion along the reference camera's own x-axis.
- **Distance units:** the composite `D` is in **pixel-equivalents**, so `ε` and the
  MCRB threshold `δ` are the same quantity.

To stress-test the conventions, the synthetic generator builds each scene in a
canonical rig frame and then maps it into the world through a **random** rigid
transform. A mistake in a pose composition cannot hide behind an identity
reference pose.

---

## 2. Package boundaries

```
src/intervene3d/
├── config/          YAML loading, _base inheritance, overrides, hashing, validation
├── utils/           logging, deterministic JSON/JSONL/CSV I/O
├── geometry/        SE(3), pinhole cameras, planes, apertures, homographies
├── hypotheses/      the H_D/H_R/H_T/H_E/H_M family; validation, equality, serialisation
├── interventions/   Action, ActionSpace, bounds, candidate generation
├── optics/          the analytical transition layer (one module per mechanism)
├── data/            shared types, the synthetic benchmark, splits, external registry
├── models/          encoders, transitions, separability, belief, selection, identifiability, baselines
├── inference/       the engine and its abstention-capable result object
├── metrics/         CEA, contact depth, identifiability AUROC, MCRB, FPCR, regret, aggregation
├── experiments/     method specs, Phase 1, the runner, figure assembly, the registry
├── visualization/   the centralised IEEE plotting system
└── reproducibility/ seeds, environment capture, run directories, manifests, hashing
```

**Dependency direction is strictly downward** — `geometry` knows nothing about
`optics`; `optics` knows nothing about `models`; `models` knows nothing about
`experiments`. Nothing outside `visualization/` touches `matplotlib`.

---

## 3. The six model interfaces

Defined as `Protocol`s in `models/interfaces.py`. Everything downstream is written
against them, so swapping a component is a configuration change.

| Interface | Contract | Implementations |
|---|---|---|
| `GeometryEncoder` | `F_t = E(I_t)` | `ground_truth`, `mock`; `moge` / `vggt_like` raise |
| `TransitionModel` | `F̂_{t+1} = W(F_t, H_k, a)` | analytical, no-hypothesis-conditioning, hybrid, learned-only |
| `SeparabilityEstimator` | `Δ_ij(a)` | deterministic geometry distance |
| `BeliefUpdater` | `p_{t+1} ∝ e^{−β e_k} p_t` | likelihood update (log-space) |
| `InterventionSelector` | `a*` | max-separability, entropy-NBV proxy, random, max-baseline, fixed, null |
| `IdentifiabilityEstimator` | `I_A`, resolvable | epsilon threshold |

---

## 4. The central data type

Everything the model sees is one `GeometryFeature` with a **fixed,
hypothesis-independent landmark layout**, so a prediction and an observation can
always be compared index-wise:

```
[0 : N]        CHANNEL_CONTENT   the apparent scene content
[N : N+4]      CHANNEL_FRAME     the four corners of the optical interface
[N+4 : N+4+K]  CHANNEL_MARKER    virtual images of observer-attached markers
```

The `MARKER` channel is what makes a planar mirror identifiable at all. A static
planar mirror reflecting a static scene produces a *static virtual scene*, whose
parallax is indistinguishable from a real scene behind an opening of the same
shape. What is not static is the virtual image of anything rigidly attached to the
observer. Whether an allowed action brings it inside the aperture is precisely an
action-set-dependent identifiability question. (See `docs/RESEARCH_SPEC_AUDIT.md`
§5, ambiguity A2 — this is a documented correction to the source specification.)

---

## 5. The optical transition layer

Two-step semantics, so all mechanism-specific physics lives in exactly one place:

1. **`build_world(F_t, H_k)`** — turn the reference feature into the world `H_k`
   asserts. The mechanisms differ here and only here: each assigns a **different
   depth** to the same reference pixels.
2. **`project_world(world, a)`** — render from the intervened camera. Shared by
   every mechanism, so any difference between predictions is attributable to
   step 1 alone.

| Mechanism | Content depth assignment | Contact surface | Identifiable by |
|---|---|---|---|
| `H_D` | the perceived depth | the content | — (the null hypothesis) |
| `H_E` static | ray ∩ screen plane | the screen | parallax of the plane; MCRB theory applies |
| `H_E` view-tracked | the perceived depth | the screen | **nothing** — identical to `H_D` inside the aperture |
| `H_R` | the perceived depth (a virtual image is a real static structure) | the mirror | moving virtual images of observer markers |
| `H_T` | perceived range + `d(1 − 1/n)` | the pane | a small, baseline-dependent parallax difference |
| `H_M` | as `H_T`, plus observer reflection | the pane | PRELIMINARY, unvalidated |

**Independent cross-validation.** Two mechanisms are checked against a second,
independent derivation rather than against themselves:

- **mirror** — the virtual-point formulation is asserted equal to the
  virtual-camera formulation (`reflect_pose`), up to the horizontal image flip a
  mirror introduces;
- **static display** — the screen-point construction is asserted equal to the
  plane-induced homography, to sub-micron pixel accuracy.

Both live in `tests/unit/test_optics.py`. They validate the **optics**; they do
not validate the *renderer*, since the simulator and the transition share the
same forward model. That limitation is recorded everywhere it matters.

---

## 6. Configuration system

Plain nested dictionaries loaded from YAML, so a config round-trips losslessly
into every run directory.

- **`_base:`** gives single-inheritance with a deep merge (child wins).
- **`--set dotted.key=value`** overrides, parsed as YAML scalars.
- **`config_hash`** is a SHA-256 over the canonical JSON, with `_`-prefixed
  transient keys **and `experiment.seed`** removed. Excluding the seed is
  deliberate: the hash must identify the *experimental configuration*, not one run
  of it, or every seed would get a different hash and multi-seed aggregation would
  silently compare a single run against itself.
- **Validation is explicit**, not schema-library-driven: the error names the key
  and says what was expected. `λ_feature ≠ 0` and any split policy other than
  `base_scene` are hard errors.

---

## 7. Experiment layer

```
scripts/run_experiment.py
  └─ runner.run_experiment          load, validate, seed, create the run directory
       └─ EXPERIMENT_KINDS[kind]    dispatch (currently: phase1_problem_existence)
            └─ phase1.run           evaluate every method, write every artefact
                 ├─ methods.py      MethodSpec → engine or classifier
                 ├─ metrics/        every headline metric
                 └─ figures.py      build figure_data.json, then draw from it
```

A **method** is one row of the baseline/ablation table, expressible entirely in
configuration: selector, transition, encoder, abstention, hypothesis conditioning.
Every baseline family and every ablation the research specification lists is a
config entry, not a code path.

Figure generation is split in two on purpose: `build_figure_data` produces a
plain JSON dictionary, and `generate_figures` reads *only* that. This is what lets
`scripts/generate_all_figures.py` rebuild the entire figure set from a finished
run without recomputing anything — and it makes hand-transcribing a number into a
plot structurally impossible.

---

## 8. Artefact layout

```
experiments/<experiment_name>/run_<UTC>_seed<N>_<confighash>/
├── config.yaml            the exact configuration, archived
├── command.txt            the command, plus the reproduction command
├── git_commit.txt         commit hash
├── environment.txt        python, OS, hardware, CUDA, packages, env vars, pip freeze
├── environment.json       the same, machine-readable
├── dataset_manifest.json  which dataset, which hash, which statistics
├── run_manifest.json      seed, config hash, git, status, metrics file, figures
├── summary.md             human-readable verdict, with interpretation limits
├── logs/run.log
├── checkpoints/
├── predictions/predictions.csv
├── metrics/{metrics.json, summary.json, figure_data.json}
├── tables/{method_comparison.csv, method_comparison.md}
└── figures/*.{pdf,png}
```

Run directories are **immutable**: if a directory of the same name somehow exists,
a numeric suffix is appended rather than any file being replaced.

`experiments/registry.jsonl` is append-only, one line per run — so aggregation is
a file read and a crashed run leaves a legible record rather than a hole.

---

## 9. Testing strategy

224 tests, all CPU-only, whole suite in about nine seconds.

| Layer | Covers |
|---|---|
| `tests/unit/test_geometry.py` | SE(3), inverses, projection, the conventions themselves |
| `tests/unit/test_optics.py` | the two independent cross-validations; matched counterfactuals; contact geometry |
| `tests/unit/test_hypotheses.py` | validation, equality, serialisation; "unidentifiable is not a mechanism" |
| `tests/unit/test_interventions.py` | bounds, rejection of out-of-bounds actions, candidate generation |
| `tests/unit/test_identifiability.py` | the distance terms; known separable and known **non**-separable cases; epsilon boundary |
| `tests/unit/test_belief.py` | normalisation, stability under `1e6` errors, posterior ordering, floor |
| `tests/unit/test_metrics.py` | perfect / wrong / abstained / degenerate inputs for every metric |
| `tests/unit/test_config.py` | inheritance, overrides, hashing, and that **every shipped config validates** |
| `tests/unit/test_models.py` | encoders, all four transitions, the learned residual, the classifiers |
| `tests/unit/test_reproducibility.py` | same seed → same sample; split determinism and leakage |
| `tests/unit/test_visualization.py` | every plot family renders; grayscale-safety; no LaTeX required |
| `tests/integration/test_pipeline.py` | the full loop per mechanism; dataset round trip; corruption detection |
| `tests/smoke/test_smoke.py` | end to end; run immutability; **failed runs are recorded** |

Several tests encode *scientific* invariants rather than code behaviour — that a
view-tracked display is provably unresolvable, that separability is exactly zero
under the null action, that a classifier on identical features cannot beat chance.
Those are the ones that would catch a silent change to the research question.
