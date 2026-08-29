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
├── data/            shared types, the synthetic benchmark, splits, and the
│                    external layer: registry, licence-respecting fetchers,
│                    per-publisher format readers
├── models/          encoders (oracle and foundation), transitions (analytical,
│                    NumPy residual, PyTorch residual), separability, belief,
│                    selection, identifiability, baselines
├── inference/       the engine and its abstention-capable result object
├── metrics/         CEA, contact depth, identifiability AUROC, MCRB, FPCR, regret, aggregation
├── experiments/     method specs, Phase 1, the residual training stage, the
│                    runner, figure assembly, the registry
├── visualization/   the centralised IEEE plotting system
└── reproducibility/ seeds, environment capture, run directories, manifests, hashing
```

**Dependency direction is strictly downward** — `geometry` knows nothing about
`optics`; `optics` knows nothing about `models`; `models` knows nothing about
`experiments`. No `rcParams` configuration exists outside `visualization/`; the
one other `matplotlib` reference in the package is
`data/synthetic/dataset_writer._save_preview`, which calls `image.imsave` to
write a preview PNG without adding a Pillow dependency and sets no style.

**Heavy dependencies stay optional and localised.** `torch` is imported only by
`models/torch_transition.py` and `models/foundation_encoders.py` (plus
`reproducibility/` for environment capture, guarded); `transformers` only by
`models/foundation_encoders.py`; `pyarrow` only by
`data/external/loaders.py`, and `Pillow` only by that module and
`models/foundation_encoders.py`. Every one of them raises a named error explaining
what to install rather than degrading silently, so the NumPy-only preliminary
pipeline still runs from a bare install.

---

## 3. The six model interfaces

Defined as `Protocol`s in `models/interfaces.py`. Everything downstream is written
against them, so swapping a component is a configuration change.

| Interface | Contract | Implementations |
|---|---|---|
| `GeometryEncoder` | `F_t = E(I_t)` | `ground_truth`, `mock`, `depth_anything_v2`; `moge` / `vggt_like` raise |
| `TransitionModel` | `F̂_{t+1} = W(F_t, H_k, a)` | analytical, no-hypothesis-conditioning, hybrid, learned-only |
| `SeparabilityEstimator` | `Δ_ij(a)` | deterministic geometry distance |
| `BeliefUpdater` | `p_{t+1} ∝ e^{−β e_k} p_t` | likelihood update (log-space) |
| `InterventionSelector` | `a*` | max-separability (summed), **maximin separability**, entropy-NBV proxy, random, max-baseline, fixed, null |
| `IdentifiabilityEstimator` | `I_A`, resolvable | epsilon threshold |

`depth_anything_v2` (`models/foundation_encoders.py`) is the first encoder that
looks at a pixel; the two oracle encoders remain the default so the encoder is
never a hard dependency. `MaxMinSeparabilitySelector` maximises the **weakest**
contending pair rather than the belief-weighted sum: a summed objective can be
maximised by separating pairs that are not in contention while leaving the
deciding pair at exactly zero, which is what was measured on the mirror scenes
the summed selector got wrong (`docs/EXPERIMENT_PLAN.md` E11).

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

## 6. The external data layer

`data/external/` is the only part of the package that reads bytes this
repository did not generate. It is split so that acquisition and interpretation
cannot be confused with one another.

| Module | Responsibility |
|---|---|
| `registry.py` | which datasets exist, their licence status, and whether automated download is permitted |
| `fetchers.py` | acquisition from the Hugging Face Hub |
| `loaders.py` | turning acquired bytes into `ExternalSample` objects |

**`fetchers.py`** re-checks the registry rather than trusting it: a fetch runs
only when the entry records *both* a `verified` licence and
`automated_download_permitted: true`; anything else prints instructions and
exits. It then lists the remote files and reports the **exact byte count before
transferring anything** — `docs/DATASET_MATRIX.md` forbids downloading an unknown
quantity — and a transfer above 1 GB additionally requires `--yes`. Partial
files resume rather than restart. Each file is verified against the
**publisher's own SHA-256** where the host exposes one (Hugging Face publishes it
as `lfs.oid`), and the fetch writes a `manifest.json` beside the data recording
per-file SHA-256, the source URL, the recorded licence and the **resolved commit
revision**, so a later `validate_external_dataset` can re-check integrity and a
result can cite an immutable version. The backend is pure standard library:
adding a download dependency for four registry entries was not worth growing
`requirements.txt`.

**`loaders.py`** provides one reader per publisher format, all behind
`get_reader(dataset, variant=...)` and all yielding the same `ExternalSample`:

| Reader | Format |
|---|---|
| `LayeredDepthReader` | parquet; `image.png` + `tuples.json` **ordinal** multi-layer relations (pairs/triplets/quads with an `is_real` flag). No metric depth at all. |
| `LayeredDepthSynReader` | parquet; `image.png` + `depth_1..depth_8.png` — eight ray-ordered depth layers with true ground truth. |
| `TransPhy3DReader` | WebDataset `.tar` shards; per frame `image.png`, `depth.png` + `depth.json` (a `max_depth` scale), `normal.png`, and `metadata.json` carrying 4×4 extrinsics and normalised 3×3 intrinsics. **The only acquired data with real observer motion**, hence the only source that supports `(F_t, a) → F_{t+1}` pairs. |
| `VisualIllusionReader` | one `tar.gz`; `test/{left,right}/*.png`, `test/disp/*.pfm` float disparity, `test/mask/*.jpg`, `calib/*.yaml`. |

Two rules are enforced in code rather than documented as intent: **nothing is
fabricated** — a missing depth, an unreadable layer or an absent calibration
yields `None` or a masked array, never a zero and never an interpolated guess,
and a genuinely broken file raises `LoaderError`; and **metric scale is applied,
not assumed** — TransPhy3D's 16-bit depth PNG means nothing without its per-frame
`max_depth`, so the reader applies it and records that it did. `reader.verify()`
re-checks the manifest checksums and `reader.provenance()` returns the repo,
revision and licence a published result must cite.

---

## 7. Real encoders and learned transitions

These are the Gate 6 and Gate 7 modules. Both are deliberately *outside* the
default path: the preliminary pipeline must keep running without `torch`.

**`models/foundation_encoders.py` (Gate 6).** `MonocularDepthEncoder` runs a
published monocular depth network through `transformers` and samples it at the
landmark pixels, returning a `GeometryFeature` with the same layout as the
simulator, so it is a drop-in replacement for the oracle. Three refusals are
structural, because each alternative would produce a plausible wrong number:

- it **never invents an image** — `encode()` raises if the observation carries no
  RGB, rather than falling back to the oracle, so a run cannot claim a foundation
  encoder it did not execute;
- it **never claims metric depth** — the network predicts *relative inverse
  depth*, and metric values appear only through `align_scale_shift()`, which fits
  the two free parameters against a reference and reports the residual, so the
  alignment is visible rather than assumed;
- it **never hides the licence** — the default checkpoint is Apache-2.0, the
  larger ones are CC-BY-NC-4.0, and `to_dict()` carries the checkpoint identifier
  and its restriction into the run manifest.

**`models/torch_transition.py` (Gate 7).** A PyTorch residual transition trained
on external rendered sequences. Its target is the residual between the observed
next frame and **exact rigid reprojection** of the reference depth, so the
network only ever has to explain what rigid geometry cannot; predicting zero is
precisely the `H_D` hypothesis. Occlusion is gated with a **z-buffer** so that
disoccluded pixels are separated from geometrically consistent ones and reported
separately, and validation holds out **whole sequences**, never frames. Two
honesty constraints are stated in the module itself and repeated wherever its
numbers appear: the network takes **no hypothesis input**, so it is not the
hypothesis-conditioned transition the project claims as its mechanism, and its
training data is Blender/Cycles rendering, not photography.

**`experiments/learned.py`** is the NumPy residual training stage that makes the
`hybrid` and `learned_only` transitions runnable at all. It trains a **separate**
model per base — against the analytical base the target is near zero, against the
identity it is the whole optical effect — uses only the **train** split, takes
targets from the simulator's pre-simulated observation for that exact action
rather than from the model being trained, and drops landmarks that are not
visible and finite in both terms instead of zero-filling them, because a zero
residual on an invisible point is a fabricated correct answer rather than a free
one.

---

## 8. Configuration system

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

## 9. Experiment layer

```
scripts/run_experiment.py
  └─ runner.run_experiment          load, validate, seed, create the run directory
       └─ EXPERIMENT_KINDS[kind]    dispatch (currently: phase1_problem_existence)
            └─ phase1.run           evaluate every method, write every artefact
                 ├─ methods.py      MethodSpec → engine or classifier
                 ├─ learned.py      residual training, when a method asks for it
                 ├─ metrics/        every headline metric
                 └─ figures.py      build figure_data.json, then draw from it
```

Four further scripts run outside `run_experiment.py`, because none of them is a
sweep over synthetic methods. Each still creates an immutable run directory with
the same provenance guarantees — config, seed, git commit, environment, the
dataset's resolved upstream revision — and each still appends to the registry:

| Script | What it runs |
|---|---|
| `scripts/train_transition.py` | Gate 7: builds `(F_t, a) → F_{t+1}` supervision from an external sequence dataset and trains `models/torch_transition.py` |
| `scripts/evaluate_external.py` | Gate 6: a published depth checkpoint on external photographs, scored inside vs outside the illusion region |
| `scripts/evaluate_identifiability.py` | identifiability against a confidence baseline, both computed from the input alone, on the same images |
| `scripts/evaluate_conformal.py` | split-conformal selective prediction over an existing identifiability run's per-image CSV; repeats no inference |

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

## 10. Artefact layout

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

## 11. Testing strategy

**283 tests**, all CPU-only, whole suite in about 37 seconds
(`.venv/Scripts/python.exe -m pytest tests -p no:cacheprovider`, measured
2026-08-29 — `283 passed in 36.86s`; 265 unit in 24.8 s, 14 integration in 2.1 s,
4 smoke in 15.2 s). The "about nine seconds" this line used to claim was true at
224 tests and is no longer; the suite has since gained the external-dataset,
Torch-transition and figure-legibility files.

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
| `tests/unit/test_visualization.py` | every plot family renders; grayscale-safety; no LaTeX required; the four figure-legibility regressions of 2026-08-29 |
| `tests/unit/test_external_datasets.py` | registry rules; every fetch refusal (unverified licence, permission not granted, unknown variant, unconfirmed large transfer); resumable and segmented download byte-exactness; manifest checksum verification catching a corruption |
| `tests/unit/test_torch_transition.py` | rigid warp and its pose convention, z-buffer occlusion gating, unmeasured pixels dropped rather than zero-filled, sequence labels surviving skipped pairs — plus, when TransPhy3D is present, that **rigid reprojection beats the identity warp** on real frames |
| `tests/integration/test_pipeline.py` | the full loop per mechanism; dataset round trip; corruption detection |
| `tests/smoke/test_smoke.py` | end to end; run immutability; **failed runs are recorded** |

Several tests encode *scientific* invariants rather than code behaviour — that a
view-tracked display is provably unresolvable, that separability is exactly zero
under the null action, that a classifier on identical features cannot beat chance.
Those are the ones that would catch a silent change to the research question.
