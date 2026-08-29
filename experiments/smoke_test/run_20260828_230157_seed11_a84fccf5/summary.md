# smoke_test -- run summary

- seed: `11`
- evaluation split: `test` (15 scene variants over 5 base scenes)
- resolvable fraction (ground truth): **0.867** (2 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.333
- action set: 15 candidate interventions; epsilon = 1 px-equivalent; tau = 0.8
- dataset: `intervene3d_synth_smoke` (config hash `1a4c0fe1`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_frame_classifier | 15 | 0.3333 | 0.3333 | 0.0000 | 0.7308 | 0.8667 | 0.0000 | n/a | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.8000 | 0.2000 |
| passive_multiview_classifier | 15 | 0.6667 | 0.6667 | 0.0000 | 0.7692 | 0.8667 | 1.0000 | n/a | n/a | n/a | n/a | 0.1500 | 0.6000 | 1.0000 | 0.4000 |
| intervene3d_no_abstention | 15 | 0.9333 | 0.9333 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0215 | 0.0300 | 0.0145 | 0.1881 | 0.3000 | 1.0000 | 1.0000 | 0.8000 |
| intervene3d | 15 | 0.8667 | 1.0000 | 0.1333 | 1.0000 | 1.0000 | 0.0000 | 0.0215 | 0.0300 | 0.0145 | 0.1881 | 0.3000 | 0.8000 | 1.0000 | 0.8000 |


## Phase 1 verdict

Basis: CEA with abstention counted as incorrect; the intervention-aware tier uses the forced-choice variant 'intervene3d_no_abstention' so all three tiers are forced-choice.

- single-frame CEA        : **0.333** (chance = 0.333)
- passive multi-view CEA  : **0.667**
- intervention-aware CEA  : **0.933**

Ordering `single-frame <= passive <= intervention-aware` holds: **True** (strictly increasing: True).

With abstention enabled the full method reports CEA(all) = 0.867, CEA(committed) = 1.000 at an abstention rate of 0.133. The gap between those two numbers is the abstention mechanism declining to name a mechanism on cases no allowed action can resolve.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1559, intercept = 2.4074, **R^2 = 0.9935** (n = 3)
- the theory predicts linearity; the slope is not expected to equal `f/delta` because the derivation uses the extremal depth pair while the measurement is an RMS residual.

## Interpretation limits (read before quoting any number)

- The causal variants are **pixel-identical at the reference view by construction**, so a near-chance single-frame result is a property of the benchmark design, not a discovery.
- Methods using the `ground_truth` encoder together with the `analytical` transition share the simulator's optics exactly. Their results are an **upper bound and a pipeline check**, not evidence about real imagery. Compare against the noisy-encoder condition for a non-degenerate reading.
- `MAE_MCRB` is near zero for oracle-encoder methods for the same reason.
- No external dataset was used in this run; see `docs/DATASET_MATRIX.md` for their status.

## Artefacts

- metrics: `metrics/metrics.json`
- figure inputs: `metrics/figure_data.json`
- predictions: `predictions/predictions.csv`
- tables: `experiments\smoke_test\run_20260828_230157_seed11_a84fccf5\tables\method_comparison.csv`, `experiments\smoke_test\run_20260828_230157_seed11_a84fccf5\tables\method_comparison.md`

- figures (44 files):
  - `fig01_pipeline_overview.pdf`
  - `fig02_initial_view_similarity.pdf`
  - `fig03_matched_variants.pdf`
  - `fig04_separability_vs_baseline.pdf`
  - `fig05_separability_matrix.pdf`
  - `fig06_action_utility.pdf`
  - `fig07_camera_trajectory.pdf`
  - `fig08_hypothesis_probabilities.pdf`
  - `fig09_predicted_vs_observed.pdf`
  - `fig10_landmark_views.pdf`
  - `fig11_explanation_accuracy.pdf`
  - `fig12_identifiability_roc.pdf`
  - `fig13_fpcr.pdf`
  - `fig14_resolvability_distribution.pdf`
  - `fig15_uncertainty_decomposition.pdf`
  - `fig16_contact_depth_error.pdf`
  - `fig17_contact_vs_apparent.pdf`
  - `fig18_mcrb_theory_validation.pdf`
  - `fig19_mcrb_error.pdf`
  - `fig20_intervention_regret.pdf`
  - `fig21_motion_cost.pdf`
  - `fig22_metric_summary.pdf`
