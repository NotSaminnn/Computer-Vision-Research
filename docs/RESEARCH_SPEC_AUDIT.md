# Research Specification Audit

**Source document:** `research(1).md` (2,549 lines, read in full)
**Concept:** Intervene3D — Interventional Identifiability of Physical Geometry under Optical Ambiguity
**Audit date:** 2026-08-28
**Codebase version:** 0.1.0

This document converts a research proposal into an implementation specification.
It records every equation, metric, dataset, baseline and ablation the source
document proposes; which of them are implemented; every ambiguity found and how
it was resolved; and — most importantly — which claims are **conceptual** versus
**experimentally testable at this stage**.

---

## 1. The research question, as implemented

The source document is explicit that the intellectual centre is *not* "what is
the depth?" and *not* "where should the camera move next?", but:

> Is the physical interpretation of this visual geometry identifiable under the
> allowed observer interventions?

Everything in the codebase is organised around that question. In particular the
codebase refuses, by construction, to output a physical label when the answer is
"no" (see `intervene3d.inference.result.InferenceResult.predicted_mechanism`,
which returns `"abstain"` rather than an argmax).

---

## 2. Equations: specification → implementation

| # | Equation (source) | Implemented in | Status |
|---|---|---|---|
| E1 | `p(I_0\|H_D) ≈ p(I_0\|H_R) ≈ p(I_0\|H_T) ≈ p(I_0\|H_E)` — matched initial view | `data/synthetic/optical_variants.py::reference_observation` | **Exact, not approximate.** All variants are pixel-identical at `C_0` *by construction*, verified by `validate_dataset` and by `tests/unit/test_optics.py`. |
| E2 | `Δ_ij(a) = D(p(O'\|H_i,a), p(O'\|H_j,a))` | `models/separability.py::GeometrySeparabilityEstimator.pairwise` | Implemented. Distributions are treated as **deterministic point predictions** (see §5, Ambiguity A1). |
| E3 | `I_A(H_i,H_j) = max_{a∈A} Δ_ij(a)` | `models/identifiability.py::identifiability_matrix` | Implemented exactly. |
| E4 | `I_A < ε → not identifiable under A` | `models/identifiability.py::evaluate` | Implemented. `ε` is a configured perceptual threshold in pixels. |
| E5 | `D = λ_g D_geom + λ_f D_feat + λ_m D_motion + λ_o D_occ` | `models/separability.py::feature_distance` | Implemented, **but only two terms are enabled by default** (§6). `λ_f ≠ 0` raises `NotImplementedError`. |
| E6 | `F̂_{t+1}^{k,j} = W_θ(F_t, H_k, a_j)` | `optics/base.py::OpticalTransition.predict`, `models/transition.py` | Implemented analytically; a small learned residual exists (§8). |
| E7 | `e_k = D(F_{t+1}, F̂_{t+1}^{k,a*})` | `inference/engine.py::Intervene3DEngine.run` | Implemented. |
| E8 | `p_{t+1}(H_k) ∝ exp(−β e_k) p_t(H_k)` | `models/belief.py::LikelihoodBeliefUpdater` | Implemented in log space with a log-sum-exp normalisation and a posterior floor. |
| E9 | `a* = argmax_a E_{i,j}[Δ_ij(a)]` | `models/selector.py::MaxSeparabilitySelector` | Implemented as the **belief-weighted** `Σ_{i<j} p_i p_j Δ_ij(a)`. |
| E10 | `a* = argmax_a I(H;O'\|a)` — expected posterior entropy reduction | — | **NOT IMPLEMENTED.** Requires a likelihood model over observations; deterministic predictions make the mutual information degenerate. See §5, Ambiguity A1. |
| E11 | `Φ_D: X' = RX + t` | `geometry/camera.py::Camera.moved`, `optics/direct.py` | Implemented. |
| E12 | `C_virt = Reflect(C, Π)` | `geometry/planes.py::reflect_pose` | Implemented **and cross-validated** against the independent virtual-point formulation. |
| E13 | Screen-plane homography | `geometry/planes.py::plane_induced_homography` | Implemented **and cross-validated** against the screen-point construction. |
| E14 | `Δu ≈ fB/Z` | `metrics/mcrb.py::differential_parallax` | Implemented. |
| E15 | `B_min > δ / (f\|1/Z_1 − 1/Z_2\|)` | `metrics/mcrb.py::mcrb_analytic` | Implemented, with the applicability conditions **enforced in code** (§7). |
| E16 | `CEA = P(Ĥ = H*)` | `metrics/classification.py` | Implemented, overall and per mechanism. |
| E17 | `AbsRel_contact`, `RMSE_contact` | `metrics/depth.py` | Implemented. |
| E18 | Identifiability AUROC | `metrics/identifiability.py` | Implemented (rank-based Mann–Whitney form). |
| E19 | `MAE_MCRB = \|B̂_min − B^GT_min\|` | `metrics/mcrb.py::mcrb_absolute_error` | Implemented. See the honesty note in §9. |
| E20 | `FPCR = P(max_k p(H_k) > τ \| y_id = 0)` | `metrics/classification.py` | Implemented, plus a sweep over `τ`. |
| E21 | `Regret = Δ(a*) − Δ(â)` | `metrics/regret.py` | Implemented (absolute and normalised). |
| E22 | `do(C_{t+1} = C')` | `interventions/actions.py::Action` | Implemented as an explicit, bounded `SE(3)` delta in the reference camera frame. |
| E23 | `ΔC + ε` — action-execution noise | `interventions/actions.py::Action.perturbed`, `action_noise` config | Implemented; wired into the experiment runner. |

---

## 3. Hypothesis family

Implemented exactly as specified, in `hypotheses/`:

`H_D` direct · `H_R` reflection · `H_T` transmission · `H_E` emissive/display · `H_M` mixed

The source document's instruction that **"unidentifiable" must not be a physical
hypothesis** is enforced structurally: `OpticalMechanism` is a closed enum with
five members and `OpticalMechanism("unidentifiable")` raises. Non-identifiability
lives only in `InferenceResult.abstained` / `.resolvable`. A regression test
(`test_unidentifiable_is_not_a_mechanism`) locks this in.

`H_E` has two sub-modes, which the source document's "ambiguity ladder"
(section 15) motivates:

- `static` — poster/monitor. Content is on the screen plane, so it exhibits the
  parallax of the plane rather than of its apparent depth. **This is the case
  the MCRB theory covers.**
- `view_tracked` — perspective-correct/head-tracked display. Inside the screen
  aperture it is *geometrically identical* to `H_D`; no baseline resolves it.
  This is the source document's "particularly valuable real subset".

---

## 4. Requirements list (what had to exist)

| Requirement | Source § | Status |
|---|---|---|
| Environment creation | plan 21 | `scripts/setup_environment.sh` |
| Dataset download/prep | plan 20 | `scripts/download_datasets.sh` (refuses unverified sources by design) |
| Dataset validation | plan 20 | `scripts/validate_datasets.py` |
| Synthetic benchmark | plan 8, 31 | `data/synthetic/`, `scripts/generate_synthetic_data.py` |
| Optical hypotheses | plan 10 | `hypotheses/` |
| Camera interventions | plan 11 | `interventions/` |
| Analytical optical transitions | plan 9 | `optics/` |
| Separability / identifiability | plan 12 | `models/separability.py`, `models/identifiability.py` |
| Belief updating | plan 13 | `models/belief.py` |
| Intervention selection | plan 11, 22 | `models/selector.py` |
| Abstention | plan 14 | `inference/` |
| MCRB | plan 15 | `metrics/mcrb.py` |
| Metrics | plan 16 | `metrics/` |
| Phase 1 experiment | plan 17 | `experiments/phase1.py` |
| Smoke test | plan 18 | `scripts/run_smoke_test.sh`, `tests/smoke/` |
| Unit/integration tests | plan 19 | `tests/` (283 tests) |
| Reproducibility | plan 6, 26, 27 | `reproducibility/` |
| IEEE visualisation | plan 24 | `visualization/` |
| Experiment registry | plan 26 | `experiments/registry.jsonl` |
| Multi-seed evaluation | plan 27 | `scripts/aggregate_results.py` |

---

## 5. Ambiguities and contradictions found, and how they were resolved

**A1 — `p(O'|H,a)` is a distribution, but nothing specifies its form.**
E2 and E10 require a probability distribution over future observations; E7's
belief update only needs a point prediction. *Resolution:* the preliminary
codebase treats predictions as deterministic, so `D` reduces to a distance
between predicted features and E10 (mutual-information selection) is left
unimplemented rather than faked with a degenerate distribution. `E9` is used
instead. Documented at `models/separability.py`.

**A2 — A static planar mirror is not resolvable by content parallax.**
The source document lists reflection as separable under camera motion. That is
**geometrically false** for a static planar mirror reflecting a static scene: the
virtual image is itself a static 3-D structure, so its parallax is identical to a
real scene seen through an opening of the same shape. This is a genuine problem
with the specification, not with the implementation.
*Resolution:* the benchmark encodes the fact rather than hiding it. Mirrors are
made identifiable only through the **virtual image of observer-attached
structure**, which moves when the observer moves — the same cue the video-mirror-
detection literature exploits. Whether any allowed action brings that virtual
image into the aperture is then a genuine action-set-dependent identifiability
question, and roughly 40 % of mirror scenes turn out to be unresolvable. This is
a strengthening of the research question, but it **is** a change and is recorded
here per the plan's "never silently change the research question" rule.

**A3 — `Z_1`, `Z_2` in the MCRB derivation are unspecified.**
The derivation compares two scene points but does not say which. *Resolution:*
the extremal content depths (`min`, `max`) are used, giving the largest available
differential parallax and hence the *smallest* resolving baseline consistent with
the theory. Stated in the docstring and stored per scene.

**A4 — Anchored vs homography-compensated resolving baselines are different quantities.**
The MCRB derivation assumes the display's projective warp has been compensated.
The system's own separability is *anchored* (both hypotheses are tied to the same
reference view), which is a strictly easier setting. These are different numbers
and conflating them would misreport the theory.
*Resolution:* both are computed and stored separately —
`mcrb` (operational, anchored) and `mcrb_compensated` (homography-compensated).
Only the compensated one is compared with `mcrb_analytic`.

**A5 — "Physical" vs "apparent" depth needs an operational definition for a display.**
*Resolution:* the reference-view percept is defined as the *apparent* content
depth for every mechanism, which is the empirical finding of the NeurIPS 2025
3D-illusion work (a monocular model sees the depicted corridor, not the screen).

> **Scope correction (2026-08-29).** That finding holds for *emissive and printed*
> surfaces and must not be stated as a general property of depth models. On
> LayeredDepth (E15), where the ambiguous surface is real glass, the direction
> reverses: on the 2,304 point pairs where the two layer definitions disagree,
> the model follows the *interface* rather than the content 70.5 % of the time.
> The benchmark's own layer labels still show the ambiguity is real and costly
> (0.848 vs 0.737 accuracy, 307 k pairs) — what is *not* supported is a fixed
> direction of error. The defensible claim is commitment under ambiguity, not
> "models report the depicted scene". Only 49 of 299 images carry a
> discriminating pair, so this refutes the universal claim without establishing
> its converse.
Ground-truth `contact_depth` is the first physical surface along the ray and is
supervision only, never an input.

**A6 — Interface parameters (plane pose, aperture, slab thickness) are assumed known.**
The source document assumes controlled, known camera motion but is silent on
whether the *scene's* optical interface is known. *Resolution:* the preliminary
codebase gives the agent the interface parameters as part of the hypothesis
specification — in a real system these would come from a plane detector. This is
a real limitation, recorded in §9 and in the README.

**A7 — Mixed optics (`H_M`) is specified but the source document defers it.**
*Resolution:* implemented minimally (`optics/mixed.py`), explicitly marked
PRELIMINARY AND NOT VALIDATED, and excluded from the Phase 1 experiment.

---

## 6. Deliberate deviations from the specification

| Deviation | Reason |
|---|---|
| `D_feature` not implemented (`λ_f = 0` enforced) | It needs a real geometry-foundation-model feature space. Silently contributing zero would let a config claim a term the code does not compute. |
| `D_geometry` off by default | Partly redundant with `D_motion` at this stage; enabling it before validation would violate the plan's "do not pretend all terms are meaningful". |
| `D_occlusion` is a **count**, not a fraction | A fraction is diluted by the landmark count, so whether a single decisive appearance crosses `ε` would depend on how many landmarks a scene happens to have. |
| CSV rather than Parquet for predictions | Avoids a pandas/pyarrow dependency for no benefit at this scale. |
| Mutual-information action selection not implemented | See A1. |
| `H_D` may carry an occluding aperture | Required for matched counterfactuals; without it, aperture clipping alone would betray the mechanism. |

---

## 7. MCRB: exactly when the equation is used

Enforced in `metrics/mcrb.py::analytic_pair_is_applicable`, which returns `False`
(and `mcrb_analytic` then returns `None` with a reason) unless **all** hold:

- the competing pair is exactly `(direct, static planar display)`;
- pure lateral translation, no rotation;
- pinhole camera, `f` in pixels, `B ≪ Z`;
- `Z_1, Z_2` finite, positive, and distinct;
- `δ` a pixel threshold.

It is explicitly **not** applied to view-tracked displays (no baseline resolves
them), mirrors, or transmission. The name **MCRB** is retained: the literature
audit found no naming collision (`docs/LITERATURE_CROSS_RESEARCH.md`).

---

## 8. Model components: what is real at this stage

| Component | Implementation | Honest status |
|---|---|---|
| `GeometryEncoder` | `ground_truth`, `mock`, `depth_anything_v2` | `moge` / `vggt_like` are still **NOT IMPLEMENTED** — they raise `NotImplementedError` with instructions and are never silently substituted. A real encoder now exists alongside them: `depth_anything_v2` (`models/foundation_encoders.py`) runs a published monocular depth checkpoint through `transformers` and is the encoder used in the external evaluation (E10/E12). It reports **relative inverse depth, not metric depth**, and it raises rather than falling back to the oracle when an observation carries no image — so a synthetic run cannot silently claim it. The Phase 1 and Phase 2 synthetic numbers in this document were **all produced with `ground_truth` or `mock`**; none of them has been re-run with a foundation encoder. |
| `TransitionModel` | analytical, no-hypothesis-conditioning, hybrid, learned-only | Analytical is the validated path. The NumPy learned residual is real and tested but a placeholder, not a world model — and on the synthetic benchmark its target is identically zero, so `hybrid` is *equal to* `analytical` there and the ablation is uninformative. A PyTorch residual (`models/torch_transition.py`) has since been trained on external rendered sequences; it is **not hypothesis-conditioned**, and it returned a negative result (`docs/DEVELOPMENT_ROADMAP.md` Gate 7). |
| `SeparabilityEstimator` | deterministic geometry distance | Real. |
| `BeliefUpdater` | likelihood update | Real, numerically hardened. |
| `InterventionSelector` | max-separability (summed), maximin separability, entropy-NBV proxy, random, max-baseline, fixed, null | Real. The NBV entry is a deliberately generic *proxy*, not a reimplementation of any published method. The maximin criterion is standard in the model-discrimination design literature (T-optimality, Hunter–Reiner) and is **not claimed as novel**; what is claimed is the measurement that the summed default fails. |
| `IdentifiabilityEstimator` | epsilon threshold | Real. |

---

## 9. Conceptual vs experimentally testable claims

This section exists so that no number in this repository is over-read.

### Experimentally testable **now**, and tested

*Seed count, stated once.* The numbers quoted below are the **3-seed** readings
this audit was written against. `results/phase1_problem_existence/aggregate.json`
now holds **11 seeds** and several values have moved (for example passive
identifiability AUROC `0.699 → 0.748`, FPCR without abstention `0.810 → 0.831`,
forced-choice CEA `0.714 → 0.722`). Every *ordering* asserted below still holds in the
11-seed means, but the exact figures do not, and the per-seed statements
("strictly increasing in each of the three seeds", the three MCRB `R²` values)
have **not** been re-verified across all 11 — treat them as unverified beyond the
original three. Take `results/` as authoritative and the numbers here as the
reading of record at the time of the audit.

- **Appearance matching.** The variants are pixel-identical at `C_0`. Verified
  exactly (deviation `0.000e+00`), so a single-frame classifier is at chance
  (measured: `0.333 ± 0.000` against a chance level of `0.333`).
- **Intervention adds information.** Measured:
  single-frame `0.333` < passive multi-view `0.621 ± 0.051` <
  intervention-aware `0.714 ± 0.017` (forced-choice, like-for-like), strictly
  increasing in each of the three seeds individually.
- **Explicit resolvability beats ordinary confidence.** Identifiability AUROC
  `1.000` (ours) vs `0.699 ± 0.045` (passive confidence) vs `0.515 ± 0.132`
  (single-frame confidence).
- **Abstention removes false physical certainty.** FPCR `0.810 ± 0.081` without
  abstention, `0.000 ± 0.000` with it, `0.841 ± 0.160` for the passive baseline.
- **Hypothesis conditioning is load-bearing.** Without it every pairwise
  separability is identically zero, so nothing is ever identifiable and the
  system abstains on 100 % of scenes.
- **The MCRB scaling law.** `1/B_min` is linear in `f|1/Z_1 − 1/Z_2|` with
  `R² = 0.85 / 0.83 / 0.91` across seeds, over 17–22 applicable scenes each.

### Conceptual, or not yet testable

- **Novelty.** Not an experimental result. See `docs/NOVELTY_RISK_REGISTER.md`.
- **Generalisation to real imagery.** The simulator and the analytical transition
  share the same forward optics for `D/R/E/T`. Oracle-encoder results are
  therefore an **upper bound and a pipeline check**, not evidence about photographs.
  The two independent cross-checks (mirror virtual camera, display homography)
  validate the *optics*; they do not validate the *renderer*.
- **`MAE_MCRB ≈ 0`.** A consequence of the previous point, not a finding.
- **Identifiability AUROC = 1.000.** Same cause: the model's identifiability
  computation is the same one that produced the label. The meaningful contrast is
  against the classifier baselines, not the absolute value.
- **RQ3 ("less motion").** Partially testable. Regret is clearly better
  (`0.235` vs `0.563` random, `0.642` generic NBV), but raw motion cost is *higher*
  than for strategies that pick small actions and fail. A motion-to-resolution
  comparison restricted to successful resolutions is Phase 4 work.
- **`H_M` mixed optics.** Implemented but unvalidated; excluded from Phase 1.
- **Interface-parameter estimation.** Assumed known (A6).
- **Everything involving external datasets** *(revised 2026-08-29)*. Four
  datasets, eight variants, ~57 GB have since been acquired, checksum-verified
  against publisher SHA-256 and pinned to immutable revisions, and three external
  evaluations have been run (`docs/EXPERIMENT_PLAN.md` E10, E12, E13), all on the
  same dataset. What that
  changes and what it does not:
  - it **does** give a first measurement on real photographs with a published
    encoder, outside this repository's own simulator;
  - it does **not** supply resolvability labels or matched counterfactuals, so it
    cannot substitute for Intervene3D-Bench, and no CEA, FPCR, regret or MCRB
    number in this document has an external counterpart;
  - the external identifiability result (AUROC `0.632` ours vs `0.339`
    confidence) is a **contrast on a modest absolute score**, on a single dataset
    with a single checkpoint, at one seed. It is not a replacement for the
    synthetic AUROC and it is not a multi-seed result.
  - `LayeredDepth`, `LayeredDepth-Syn` and the TransPhy3D splits are acquired and
    readable but have **not** been used for any reported evaluation of this
    system; TransPhy3D has been used only for the Gate-7 training run.

---

## 10. Components in the source document deliberately NOT built

Per the plan's section 42 ("what not to do in the first implementation"), and the
source document's own section 37:

diffusion depth · generic camera calibration · VLM reasoning · SLAM · navigation
policy learning · full 3-D reconstruction · semantic object recognition ·
generative RGB video · arbitrary object actions · any giant transformer.

Controlled camera motion is assumed known, which the source document explicitly
accepts for a first paper.
