# Novelty Risk Register

**Audit date:** 2026-08-28 · **Revised:** 2026-08-29 · **Basis:** `docs/LITERATURE_CROSS_RESEARCH.md`

Every claim Intervene3D might make, the existing work that threatens it, how much
they overlap, the evidence, and wording that is safe to use.

**Rule:** if a row is `HIGH` overlap, the corresponding claim must not appear as a
contribution anywhere — paper, README or code comment.

---

## A. Claims that must NOT be made (the source document already concedes these)

| Claim | Existing work | Degree of overlap | Evidence | Safe wording |
|---|---|---|---|---|
| Multi-layer depth / multiple depths per ray | LayeredDepth (ICCV 2025); SeeGroup (CVPR 2026 Oral); One Scene, Two Depths (MD-3k) | **HIGH — fully occupied** | arXiv:2503.11633 (VERIFIED); arXiv:2605.28735 (VERIFIED); arXiv:2606.29600 (VERIFIED) | "Building on established multi-layer depth formulations, we ask a different question: which layer's *cause* is identifiable." Never claim the task. |
| Physical vs apparent geometry on mirrors/glass | GIFT (arXiv:2608.02068); Depth-Aware Mirror Segmentation (CVPR 2021); **The 3D Mirage (arXiv:2512.15423)** | **HIGH** | GIFT abstract VERIFIED: recovers true physical geometry of non-Lambertian surfaces. The 3D Mirage VERIFIED: benchmark + metrics + training for 3D hallucination | "The distinction between apparent and physical (contact) geometry is established. We study when that distinction is *decidable* from a bounded set of observer actions." |
| Recovering the true mirror/glass surface | GIFT; Mirror3D (CVPR 2021) | **HIGH** | arXiv:2608.02068, arXiv:2106.06629 (both VERIFIED) | "We adopt contact geometry as an output; recovering it is not our contribution." |
| Explicit reflection/transmission decomposition | GLINT (CVPR 2026 Oral, Award Candidate) | **HIGH** | arXiv:2603.26181 + CVPR oral page (VERIFIED) | "GLINT decomposes a scene assumed transparent. We do not assume which optical decomposition is correct; we ask whether the evidence can decide." |
| Camera motion as a mirror cue | CVPR 2024 inconsistent-motion video mirror detection; MVMD (WACV 2025); MiD-VMD (AAAI 2026) | **HIGH** | Cited in the source document; NOT individually re-verified in this audit | "Motion-based mirror detection is established. Our contribution is not the cue but the *identifiability calculus* over competing explanations." |
| Active vision / next-best-view | NBV for reflective objects (arXiv:2202.13263); classic active-vision hypothesis verification | **HIGH** | arXiv:2202.13263 VERIFIED (builds depth/reflection hypotheses, computes information gain, selects a view) | "We do not claim information-gain viewpoint selection. We ask, *before* selecting, whether any allowed action can separate the hypotheses at all." |
| Counterfactual / geometry world models | CWM (2023); CWM for physical dynamics (ECCV 2024); 3WM; P3Sim; **VGGT-World** | **HIGH** | VGGT-World VERIFIED (arXiv:2603.12655) — forecasts frozen GFM features rather than RGB, exactly our architectural choice | "The world model is an enabling mechanism, not a contribution. Our use of it is to predict the outcome of a visual experiment under a *specific optical hypothesis*." |
| Causal identifiability from interventions | CITRIS (ICML 2022) | **HIGH** | PMLR page VERIFIED | "Causal identifiability under intervention is established. We specialise it to *image-formation explanations of apparent geometry* under a bounded observer action set." |

---

## B. Claims that remain plausibly defensible

| Claim | Nearest existing work | Degree of overlap | Evidence | Safe wording |
|---|---|---|---|---|
| **Action-set-dependent identifiability of competing optical explanations** — `I_A(H_i,H_j) = max_{a∈A} Δ_ij(a)` as the object of inference | CITRIS (general causal identifiability); NBV (chooses a view assuming a state exists) | **LOW–MEDIUM** | Targeted searches for `"action-set identifiability"` + camera motion + image-formation hypotheses returned nothing combining them | "We formulate physical-geometry inference under optical ambiguity as an action-set-dependent identifiability problem." **Add: "to our knowledge, based on a targeted audit through August 2026."** |
| **Explicit prediction of non-identifiability (abstention) rather than forced classification** | Generic uncertainty calibration; selective prediction | **LOW–MEDIUM** | No work found that distinguishes prediction uncertainty from *physical* identifiability uncertainty in this setting | "This enables a system to distinguish uncertainty caused by insufficient prediction quality from ambiguity that is fundamentally unresolved by the available visual experiments." |
| **Matched counterfactual optical benchmark with unresolvable cases** | 3D Visual Illusion (NeurIPS 2025) — categories but not matched pairs; GIFT's controlled benchmark; **The 3D Mirage** — context variation | **MEDIUM** | 3D Visual Illusion VERIFIED: 5 illusion categories, ~200k images, but no matched-at-`C_0` counterfactuals, no action set, no resolvability labels | "Existing benchmarks collect optical anomalies. Ours pairs them: the same apparent geometry produced by different mechanisms, under the same calibrated action set, with explicit resolvable / non-resolvable labels." |
| **MCRB — Minimum Causal Resolving Baseline** | No naming collision found | **LOW (name) / MEDIUM (content)** | The underlying `Δu ≈ fB/Z` parallax relation is textbook projective geometry | "We instantiate the classical parallax relation as a resolving-baseline criterion for a specific competing pair, and validate the scaling empirically (`R² = 0.92`)." **Do not present the derivation itself as new geometry.** |

---

## B2. Claims added on 2026-08-29, with their prior art

| Claim | Nearest existing work | Overlap | Evidence | Safe wording |
|---|---|---|---|---|
| **Maximin experiment selection beats the summed information objective for *decisions*** | **T-optimality (Atkinson & Fedorov 1975); Hunter–Reiner (1965); robust T-optimal discriminating designs** | **HIGH as statistics — the criterion is classical. LOW as vision.** | Verified 2026-08-29. Model-discrimination design is a mature sub-field of optimal experimental design, with its own criteria distinct from information gain. | **Do NOT claim the criterion.** Claim the transfer and the measurement: "Active vision selects viewpoints by expected information gain, a sum over hypothesis pairs. We show measurably that this can be maximised by an action leaving the deciding pair unseparated — `Δ(H_D,H_R) = 0.00` on every mirror the system got wrong — and that a model-discrimination criterion from the experimental-design literature fixes it." Cite the statistics as support, not as a competitor. |
| **Epistemic sufficiency: whether the evidence available to the observer can decide, as distinct from model uncertainty** | Selective prediction / Chow's rule (1970) and its 60-year literature; conformal prediction; aleatoric–epistemic decomposition | **LOW–MEDIUM** | No work found that separates *evidence insufficiency* from *model uncertainty* in a perception setting. The distinction is empirically demonstrated here, not merely asserted: confidence predicts failure at AUROC 0.339 (below chance) while identifiability reaches 0.632 on the same images. | "Uncertainty is conventionally decomposed into aleatoric and epistemic components, both properties of the estimator. We measure a third quantity that is a property of the *(hypotheses, action set, resolution)* triple: whether any available observation can decide. It is reducible only by acting." Qualify as ever. |
| **Calibration cannot repair a badly-ranking score** | Conformal prediction; selective classification with guarantees | **LOW as a claim, but it is a known property of conformal** stated informally elsewhere | Measured: split conformal on confidence honours coverage at 5/5 levels and still reaches 76.6 % selective risk against a 65.2 % no-abstention baseline. | "Conformal prediction constrains how many predictions are retained, never which. We give a setting where a validly calibrated confidence-based selector is worse than not abstaining." **Do not present this as a flaw in conformal** — it is a flaw in the score, and conformal behaved exactly as specified. |
| **Ground truth in existing illusion benchmarks can be unable to arbitrate the question** | No direct prior work found | **LOW** | 159 of 455 pairs in a NeurIPS 2025 benchmark have non-planar measured disparity where the physical surface is a flat panel, because stereo matching locks onto displayed content. | "Evaluating apparent-versus-physical geometry requires ground truth that is itself immune to the illusion." State as a methodological caution, not as criticism of the dataset's authors. |

**Standing rule for all four:** the first row's criterion is *not* ours. Any
sentence implying we invented maximin or minimax experiment design must be cut.

---

## C. Active risks

### R1 — "This is just active perception applied to mirror/glass classification." **[HIGHEST]**

*Response (the one the source document prescribes, and the codebase now supports
with measurements):* traditional NBV solves `argmax_a I(X;O'|a)`, assuming a
state `X` exists to be estimated. We first ask whether
`sup_{a∈A} D[p(O'|H_i,a), p(O'|H_j,a)] > ε` at all, and abstain when it is not.
The `intervene3d_no_hypothesis_conditioning` ablation makes this concrete: with
the causal conditioning removed, every pairwise separability is identically zero
and the system abstains on 100 % of scenes. A generic NBV proxy reaches CEA
`0.368` against `0.557` (11 seeds). **Do not answer "but ours uses a world
model."**

*Strengthened 2026-08-29.* The reply is no longer only a distinction. The summed
information objective — what NBV actually optimises — was measured selecting
actions that score `Δ(H_D,H_R) = 0.00` on every mirror the system got wrong,
while actions worth 4–12 px-equivalent were available. Under execution noise the
hypothesis-blind proxy loses **48.4 %** of its accuracy against **2.3 %** here
(paired, 10 seeds). That is a measured failure of the standard objective, not a
claim about ours.

### R2 — "GIFT already distinguishes physical geometry from reflections." **[HIGH]**

*Response:* GIFT solves `I → D_physical`. We solve
`(I, A) → [p(H), I_A, a*, D_contact]`, including the case where no output should
be produced. GIFT always outputs geometry; we can decline.

### R3 — "GLINT already models interface / reflection / transmission." **[HIGH]**

*Response:* GLINT assumes the scene is transparent and reconstructs it. We do not
assume which optical decomposition holds; the competing decompositions are the
inference target.

### R4 — **The 3D Mirage (arXiv:2512.15423)** — a threat the source document does not list. **[MEDIUM–HIGH]**

Found during this audit. Probes and *resolves* 3D hallucination on planar,
perceptually ambiguous inputs with a benchmark, metrics (DCS, CCS) and a training
strategy. Closest work to our motivation.
*Response:* it addresses whether a model hallucinates and how to stop it; it does
not, on the evidence of its abstract, address camera actions, action-set-dependent
identifiability, or abstention. **Action required: cite it, and sharpen the
framing away from "detecting hallucinated 3D structure" toward "deciding whether
the cause is determinable".**

### R5 — Misuse of "causal". **[MEDIUM]**

The source document warns that using "causal" merely because frames are
sequential would damage the paper. *Mitigation, already in the code:* the
intervention is an explicit `do(C_{t+1} = C_0 · a)` with `a ∈ SE(3)` drawn from a
bounded, serialised action set (`interventions/action_space.py`), and hypotheses
are explicit image-formation mechanisms. The preferred term is **Interventional
Identifiability**, not "Causal Geometry".

### R6 — The benchmark is synthetic and shares its optics with the model. **[MEDIUM]**

The simulator and the analytical transition use the same forward optics, so
oracle-encoder results are an upper bound. *Mitigation:* stated in every run's
`summary.md`, in `docs/RESEARCH_SPEC_AUDIT.md` §9, and in the README. The mirror
and display transitions **are** independently cross-validated (virtual camera;
plane-induced homography), so the optics are checked even though the renderer is
not.

*Partly discharged 2026-08-29.* The identifiability claim now has independent
support: on 296 real stereo pairs from a published benchmark, with a released
depth checkpoint and nothing tuned, identifiability predicts failure at AUROC
0.632 against 0.339 for confidence. That does **not** validate the synthetic
benchmark itself, and every synthetic number in this repository remains subject
to the shared-optics caveat.

### R7 — A concurrent CVPR 2027 submission. **[UNQUANTIFIABLE]**

The area is moving fast. The audit covers work up to 2026-08-28 only. Re-run the
queries in `docs/LITERATURE_CROSS_RESEARCH.md` §4 before any submission.

---

## D. The claim as it should currently be worded

> We formulate physical-geometry inference under optical ambiguity as an
> action-set-dependent identifiability problem. Rather than forcing a unique 3-D
> interpretation, the framework maintains competing image-formation hypotheses,
> predicts how each should respond to controlled observer interventions, selects
> the experiment that best separates them, and **explicitly abstains when the
> available visual evidence cannot establish physicality**. This distinguishes
> uncertainty caused by insufficient prediction quality from ambiguity that is
> fundamentally unresolved by the available visual experiments.

Qualify every novelty statement with *"to our knowledge, based on a targeted
literature audit through August 2026"*.

**Not to be claimed:** causality, identifiability, active vision, light transport,
world models, or depth ambiguity as such. The contribution is the specific vision
task formed at their intersection.
