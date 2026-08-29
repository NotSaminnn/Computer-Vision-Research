# Development Roadmap

Gates, not a linear plan. **Do not pass a gate until the previous one holds.**
The purpose is to avoid spending weeks on a sophisticated architecture before the
central experimental premise is shown to be measurable.

---

## Gate 1 — Synthetic environment and smoke test · **PASSED**

`bash scripts/run_smoke_test.sh` exits 0 from a clean checkout. All 17 required
steps pass; 22 figures are produced and verified non-empty; the run manifest,
metrics, predictions and tables all exist.

## Gate 2 — Analytical transitions validated · **PASSED**

Mirror, display and direct transitions are validated by unit tests, and two of
them against **independent derivations** rather than against themselves:

- mirror: virtual-point vs virtual-camera formulations agree to `1e-9`;
- static display: screen-point construction vs plane-induced homography agree to `1e-6` px.

Plus: matched counterfactuals hold exactly (`0.000e+00` deviation); a view-tracked
display is provably non-separable from a direct scene; separability is exactly
zero under the null action; separability increases monotonically with baseline.
Visualised in figures 04, 05, 10, 17, 18.

## Gate 3 — Phase 1 runs end to end · **PASSED**

288-variant benchmark, 9 methods, now **11 seeds**, aggregated with confidence
intervals into `results/phase1_problem_existence/`. ~28 s per run on CPU.

## Gate 4 — Does intervention add measurable information? · **PASSED**

At 11 seeds (`results/phase1_problem_existence/aggregate.json`): single-frame
`0.333 ± 0.000` (exactly chance) < passive multi-view `0.621 ± 0.046` <
intervention-aware, forced-choice `0.722 ± 0.029` — and strictly increasing in
**each of the 11 seeds individually**, not merely in the mean. FPCR
`0.831 → 0.000` with abstention, and `0.000` in every seed. MCRB scaling law
`R² = 0.818 – 0.950` across the 11 seeds, over 13–26 applicable scenes each.
`docs/EXPERIMENT_PLAN.md` E2 still quotes the original 3-seed reading; where the
two differ, `results/` is authoritative.

**The central premise is measurable. Proceed.**

---

## Gate 5 — External datasets · **PASSED** (2026-08-29)

Acquired and readable: **four datasets, eight variants, ~57 GB**, every file
verified against the publisher's own SHA-256 and pinned to an immutable upstream
revision recorded in a per-variant `manifest.json`.

| Dataset | Variants acquired | Licence |
|---|---|---|
| LayeredDepth | `validation`, `test` | CC0-1.0 |
| LayeredDepth-Syn | `validation`, `train` | BSD-3-Clause |
| TransPhy3D | `sample`, `test` | Apache-2.0 |
| 3D Visual Illusion | `real`, `virtual_meta` | Apache-2.0 |

The prerequisites were met rather than waived: sources re-verified, licences read
and recorded verbatim in `configs/datasets/external.yaml`, byte counts reported
before any transfer (`data/external/fetchers.py` refuses an unverified licence,
refuses without `automated_download_permitted`, and holds any transfer above
1 GB until `--yes`), and one reader per publisher format in
`data/external/loaders.py`. The registry's remaining entries stay
**unacquired**, each for a stated reason the fetcher enforces: `Mirror3D`,
`Booster`, `MD3K`, `GIFT-Benchmark`, `MVMD`, `MAGD`, `DepthFocus-Synth` and
`PDI` have an unverified licence or unverified access; `ClearPose` and `DREDS`
have verified licences but do not grant automated download, so acquiring them
means following the publisher's own route by hand.

**The blocking issue still blocks.** External data cannot supply resolvability
labels or matched counterfactuals. It has been used for *external* tests of
apparent-versus-physical geometry (Gate 6) and for transition training (Gate 7);
it is not a substitute for Intervene3D-Bench, and no CEA, FPCR, regret or MCRB
number has an external counterpart.

**A new limitation, found by using the data.** 159 of 455 stereo pairs in
3D Visual Illusion carry ground truth that cannot arbitrate the question being
asked: stereo matching on a display locks onto the *displayed content* rather
than the panel, so the measured disparity is not planar where a physical panel
is. Before filtering, the `video_monitor` category reported a 1 % failure rate —
the exact inverse of the truth. Published benchmarks do not guarantee ground
truth immune to the illusion they contain.

## Gate 6 — Real geometry foundation encoder · **PASSED** (2026-08-29) — first encoder only

`models/foundation_encoders.py::MonocularDepthEncoder` is the first encoder in
this repository that looks at a pixel. Depth-Anything-V2-Small (Apache-2.0), run
through `transformers`, **24 ms/image**, 0.35 GB VRAM, used exactly as released —
nothing trained or tuned. Selectable as
`model.geometry_encoder.name: depth_anything_v2`.

Sanity-checked against **TransPhy3D**'s rendered depth: **r² = 0.9797** after the
scale-and-shift alignment its relative output requires — the encoder is reading
the scene, not producing noise. Then run in anger on the 455 real stereo pairs of
3D Visual Illusion (`scripts/evaluate_external.py`,
`scripts/evaluate_identifiability.py`), which is what Gate 6 existed for: an
encoder whose optics are not this repository's simulator.

What the gate's own conditions bought:

- the checkpoint identifier and its **licence** land in every run manifest — the
  default is Apache-2.0, the Base and Large checkpoints are CC-BY-NC-4.0 and
  selecting one records the non-commercial restriction;
- `encode()` **raises when the observation carries no image** rather than falling
  back to the oracle, so a run cannot claim a foundation encoder it never ran;
- metric depth is never claimed: the network emits *relative inverse depth* and
  metric values appear only through `align_scale_shift`, which reports its
  residual.

`ground_truth` / `mock` remain the defaults, so this is not a hard dependency.

**What is still open.** `moge` and `vggt_like` remain **NOT IMPLEMENTED** and
still raise with instructions. One encoder is not the encoder study. And the
expected effect has *not* been obtained: the synthetic identifiability AUROC and
`MAE_MCRB` are still `1.000` and `≈0`, because **no synthetic experiment has been
re-run with this encoder** — the benchmark stores landmark arrays and only a
handful of preview renders, so most synthetic scenes cannot be encoded from
pixels at all. Making those two metrics non-degenerate needs an image-carrying
benchmark, not just an encoder.

## Gate 7 — Scale the learned world model · **RUN, NEGATIVE RESULT** (2026-08-29)

Implemented and executed, and the answer was no. Recording it as a completed
gate with a negative outcome, not as a success.

`models/torch_transition.py` is a PyTorch residual transition trained on
TransPhy3D `test` — 2,750 frame pairs at stride 5, 11.26 M supervised samples,
6 of 25 **sequences** held out (splits are sequence-wise, never frame-wise), 124 s
on an RTX 5070 (`scripts/train_transition.py`). The target is the residual
between the observed next frame and exact rigid reprojection of the reference
depth, so predicting zero is exactly the `H_D` hypothesis and the learned
residual is the departure from it. Occlusion is separated with a z-buffer and
reported separately, because disoccluded pixels carry orders of magnitude more
residual than optics does.

| subset | rows | rigid (`H_D`) baseline MSE | model MSE | ratio |
|---|---|---|---|---|
| geometrically consistent | 2,416,960 (97.9 %) | 7.74e-07 | 2.29e-03 | **2956.7×** |
| occlusion-affected | 52,928 (2.1 %) | 1.228 | 0.778 | 0.634× |
| *pooled — **not** the result* | | 0.0263 | 0.0189 | 0.719× |

**Why it failed, and it is not a training failure.** On the geometrically
consistent pixels — where transparency and reflection live, and the only place
the optics claim could be tested — the rigid baseline's MSE is `7.7e-07`, i.e. an
RMSE of about `9e-04`. That is the **16-bit depth quantisation floor** of the
source, not a physical residual: TransPhy3D stores *rendered geometric depth*, so
the depth of a transparent or reflective surface is the depth of that surface,
not of what is seen through or in it. The optical effect this project is about
never enters the depth channel. There was nothing to learn, and the model was
**~3000× worse than predicting zero** exactly where the signal was supposed to
be. What it did learn is occlusion, where it beats the rigid baseline 0.63×.

The pooled ratio of `0.719×` looks like a win and is dominated entirely by the
2.1 % occlusion rows. It is labelled `_POOLED` in the metrics and must never be
quoted as the result.

**What this closes and what it does not.** It closes "train the residual on
external data and see whether hypothesis-free transition learning picks up
optics": on this source it cannot, for a reason in the data rather than the
method. It does **not** test hypothesis-conditioned transition learning — this
network takes no hypothesis input and is therefore not the mechanism the project
claims. A real test needs a source whose depth channel records the *physical*
contact surface while the image shows the apparent one, which rendered geometric
depth by construction does not.

`models/learned_transition.py` (the NumPy MLP) remains what it always was: real,
tested, deterministic, and a placeholder that operates on landmarks rather than
on a geometry-foundation-model latent. On the synthetic benchmark its residual
target is identically zero, so `hybrid` equals `analytical` there and the
ablation is uninformative.

Standing constraint, unchanged: re-read `docs/NOVELTY_RISK_REGISTER.md` R1 —
VGGT-World already forecasts frozen GFM features, so **the world model must never
be presented as the contribution**.

---

## Immediate next milestone (recommended)

The previous milestone list is discharged: Phase 2 benchmark core has been run
(`docs/EXPERIMENT_PLAN.md` E4, 3 seeds — the graded ambiguity ladder held, with
non-identifiability rising from ~36 % to ~70 % once glass is added), the
action-noise generalisation study has been run (E8, 10 seeds paired), and
`The 3D Mirage` (arXiv:2512.15423) is in `docs/LITERATURE_CROSS_RESEARCH.md` §2.1
and `docs/NOVELTY_RISK_REGISTER.md` R4. Gates 5, 6 and 7 have all been attempted.

What is still open, in the order it should be taken:

1. **Validate `H_M` against an independent formulation, or drop it from the
   paper.** Unchanged from the previous list. It is still PRELIMINARY and still
   excluded from every experiment.
2. **Implement the RQ3 comparison properly: motion to resolution**, restricted to
   scenes each method actually resolves. Unchanged. The raw motion-cost
   comparison is not the claim the research question makes.
3. **Raise the seed count everywhere the paper quotes a number.**
   `phase1_problem_existence` and `phase1_selector_study` are at 11 seeds,
   `phase1_action_noise` at 10, but `phase2_benchmark_core` is at **3** and every
   external result is at **one seed with one checkpoint**.
4. **Multi-step intervention.** The engine supports `max_steps > 1`; nothing has
   ever run at more than one step.
5. **An image-carrying synthetic benchmark**, without which Gate 6's encoder
   cannot be used on Intervene3D-Bench at all (see Gate 6).
6. **A data source for Gate 7 whose depth channel records the physical contact
   surface**, without which the learned-transition question stays unanswerable
   (see Gate 7).

---

## Known limitations, ranked by how much they threaten a conclusion

1. **The simulator and the analytical transition share the same forward optics.**
   Oracle-encoder results are an upper bound. Gates 5 and 6 relieve this only
   *outside* the benchmark: the external evaluations use real photographs and a
   published encoder, but **every synthetic number in this repository is still
   produced by `ground_truth` or `mock`**, so the upper-bound caveat stands
   unchanged for all of them.
2. **Interface parameters (plane pose, aperture, slab thickness) are assumed
   known.** In a real system they would come from a plane detector, and estimation
   error would propagate into every prediction. Nothing in the current results
   measures that.
3. **The observation model is landmark-based, not dense.** Real encoders produce
   dense point maps; occlusion and matching failures are not modelled.
4. **`H_M` is unvalidated.**
5. **RGB rendering is a splat renderer**, adequate for figures, not physically based.
6. **Real-world data now exists but is thin.** One dataset (3D Visual Illusion,
   real stereo), one checkpoint, one seed, 296 usable images — and 159 of the 455
   had to be excluded because their ground truth cannot arbitrate the question.
   No external result is multi-seed and none has a second dataset behind it.
7. **Multi-step intervention is implemented but untested at `max_steps > 1`.**
8. **The learned transition is not hypothesis-conditioned.** Neither the NumPy
   residual nor the PyTorch one takes a hypothesis index, so neither is evidence
   for or against the project's central mechanism.
