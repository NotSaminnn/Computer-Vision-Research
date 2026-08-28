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

288-variant benchmark, 9 methods, 3 seeds, aggregated with confidence intervals.
~28 s per run on CPU.

## Gate 4 — Does intervention add measurable information? · **PASSED**

single-frame `0.333` (chance) < passive multi-view `0.621 ± 0.051` <
intervention-aware `0.714 ± 0.017`, strictly increasing in every seed; FPCR
`0.810 → 0.000` with abstention; MCRB scaling law `R² = 0.83–0.91`. Details in `docs/EXPERIMENT_PLAN.md` E2.

**The central premise is measurable. Proceed.**

---

## Gate 5 — External datasets · **NOT STARTED**

Prerequisites, per dataset, before any download:

1. Re-verify the official source page (they move).
2. Record the licence **verbatim** in `configs/datasets/external.yaml` and set
   `licence_status: verified` only if it was actually read.
3. Confirm storage requirements and any authentication.
4. Write an adapter under `data/external/` producing the shared
   `GeometryFeature` landmark layout.
5. Write a `manifest.json` with per-file SHA-256 checksums, source URL and version.
6. Keep several benchmarks **out of training** so a zero-shot claim is possible.

Order (from `docs/DATASET_MATRIX.md`): LayeredDepth → TransPhy3D →
3D Visual Illusion → Mirror3D → ClearPose/DREDS.

**Blocking issue:** external data cannot supply resolvability labels or matched
counterfactuals. It can only be used for *external* tests of contact depth and
mechanism stability, never as a substitute for Intervene3D-Bench.

## Gate 6 — Real geometry foundation encoder · **NOT STARTED**

`moge` and `vggt_like` adapters are registered and raise `NotImplementedError`
with instructions. Before implementing either:

1. verify the current official repository and its licence;
2. verify checkpoint availability and terms;
3. verify Python / PyTorch compatibility;
4. verify GPU memory requirements;
5. verify the inference command on one image;
6. re-run the literature check `latest geometry foundation models 2027`.

Then implement `encode()` returning the shared landmark layout, and record the
checkpoint identifier in the run config so it lands in every run manifest.

**The preliminary pipeline must keep working with `ground_truth` / `mock`**, so
this integration can never become a hard dependency.

Expected effect: this is what makes identifiability AUROC and `MAE_MCRB`
non-degenerate. Today they are `1.000` and `≈0` because the model's optics are
the simulator's optics.

## Gate 7 — Scale the learned world model · **PLACEHOLDER EXISTS**

`models/learned_transition.py` is a two-hidden-layer NumPy MLP predicting a
per-landmark residual on top of the analytical prediction. It is real, tested and
deterministic, but it is a placeholder: it operates on landmarks, not on a
geometry-foundation-model latent, and is not a world model.

Before scaling: re-run `latest geometry world models 2027`, and re-read
`docs/NOVELTY_RISK_REGISTER.md` R1 — VGGT-World already forecasts frozen GFM
features, so **the world model must never be presented as the contribution**.

---

## Immediate next milestone (recommended)

**Phase 2 benchmark core**, in this order:

1. Run `configs/synthetic/benchmark_core.yaml` (adds `H_T`; 4 mechanisms,
   192 base scenes) and check that glass produces the expected graded ambiguity
   ladder — resolvable only at large baselines.
2. Validate `H_M` against an independent formulation, or drop it from the paper.
3. Implement the RQ3 comparison properly: **motion to resolution**, restricted to
   scenes each method actually resolves. The current raw motion-cost comparison is
   not the claim the research question makes.
4. Run the action-noise generalisation study (`action_noise.enabled: true`) —
   already implemented, never executed.
5. Add `The 3D Mirage` (arXiv:2512.15423) to related work and re-check that the
   novelty framing still holds against it.

Only then move to Gate 5.

---

## Known limitations, ranked by how much they threaten a conclusion

1. **The simulator and the analytical transition share the same forward optics.**
   Oracle-encoder results are an upper bound. → Gates 5 and 6.
2. **Interface parameters (plane pose, aperture, slab thickness) are assumed
   known.** In a real system they would come from a plane detector, and estimation
   error would propagate into every prediction. Nothing in the current results
   measures that.
3. **The observation model is landmark-based, not dense.** Real encoders produce
   dense point maps; occlusion and matching failures are not modelled.
4. **`H_M` is unvalidated.**
5. **RGB rendering is a splat renderer**, adequate for figures, not physically based.
6. **No real-world data.**
7. **Multi-step intervention is implemented but untested at `max_steps > 1`.**
