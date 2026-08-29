# phase1_selector_study -- run summary

- seed: `42`
- evaluation split: `test` (78 scene variants over 26 base scenes)
- resolvable fraction (ground truth): **0.705** (23 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.333
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_phase1_s42` (config hash `ae814265`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passive_multiview_classifier | 78 | 0.6667 | 0.6667 | 0.0000 | 0.8577 | 0.7051 | 0.8261 | n/a | n/a | n/a | n/a | 0.2000 | 0.9231 | 0.8462 | 0.2308 |
| max_baseline_intervention | 78 | 0.5641 | 0.8462 | 0.3333 | 0.9727 | 0.9615 | 0.0000 | 0.1565 | 0.2731 | 0.0033 | 0.3590 | 0.3000 | 0.5769 | 0.8462 | 0.2692 |
| intervene3d_sum | 78 | 0.5897 | 0.8364 | 0.2949 | 1.0000 | 1.0000 | 0.0000 | 0.1544 | 0.2732 | 0.0000 | 0.2863 | 0.2750 | 0.5769 | 0.8846 | 0.3077 |
| intervene3d_maxmin | 78 | 0.6154 | 1.0000 | 0.3846 | 0.9364 | 0.9103 | 0.0000 | 0.1077 | 0.1799 | 0.0033 | 0.2418 | 0.1827 | 0.5769 | 0.6923 | 0.5769 |
| intervene3d_maxmin_forced | 78 | 0.8205 | 0.8205 | 0.0000 | 0.9364 | 0.9103 | 0.6087 | 0.1077 | 0.1799 | 0.0033 | 0.2418 | 0.1827 | 1.0000 | 0.8846 | 0.5769 |


## Phase 1 verdict

Ordering NOT EVALUATED: the three tiers were not all present in `methods`.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1441, intercept = 1.8121, **R^2 = 0.8436** (n = 18)
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
- tables: `experiments\phase1_selector_study\run_20260829_013128_seed42_e188f489\tables\method_comparison.csv`, `experiments\phase1_selector_study\run_20260829_013128_seed42_e188f489\tables\method_comparison.md`

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
