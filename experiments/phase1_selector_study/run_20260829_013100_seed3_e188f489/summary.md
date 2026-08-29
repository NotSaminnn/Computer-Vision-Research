# phase1_selector_study -- run summary

- seed: `3`
- evaluation split: `test` (87 scene variants over 29 base scenes)
- resolvable fraction (ground truth): **0.632** (32 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.333
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_phase1_s3` (config hash `2fd7b0f4`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passive_multiview_classifier | 87 | 0.5632 | 0.5632 | 0.0000 | 0.7449 | 0.6322 | 0.6562 | n/a | n/a | n/a | n/a | 0.2000 | 0.6897 | 0.6897 | 0.3103 |
| max_baseline_intervention | 87 | 0.4713 | 0.8200 | 0.4253 | 0.9545 | 0.9425 | 0.0000 | 0.2186 | 0.3880 | 0.0000 | 0.3386 | 0.3000 | 0.4828 | 0.6897 | 0.2414 |
| intervene3d_sum | 87 | 0.5287 | 0.8364 | 0.3678 | 1.0000 | 1.0000 | 0.0000 | 0.1912 | 0.3243 | 0.0000 | 0.2187 | 0.2466 | 0.4828 | 0.6897 | 0.4138 |
| intervene3d_maxmin | 87 | 0.5057 | 1.0000 | 0.4943 | 0.9000 | 0.8736 | 0.0000 | 0.1752 | 0.2930 | 0.0000 | 0.2237 | 0.1481 | 0.4828 | 0.5517 | 0.4828 |
| intervene3d_maxmin_forced | 87 | 0.7241 | 0.7241 | 0.0000 | 0.9000 | 0.8736 | 0.2500 | 0.1752 | 0.2930 | 0.0000 | 0.2237 | 0.1481 | 1.0000 | 0.6897 | 0.4828 |


## Phase 1 verdict

Ordering NOT EVALUATED: the three tiers were not all present in `methods`.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1657, intercept = 0.7230, **R^2 = 0.9073** (n = 17)
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
- tables: `experiments\phase1_selector_study\run_20260829_013100_seed3_e188f489\tables\method_comparison.csv`, `experiments\phase1_selector_study\run_20260829_013100_seed3_e188f489\tables\method_comparison.md`

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
