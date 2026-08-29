# phase1_selector_study -- run summary

- seed: `1`
- evaluation split: `test` (90 scene variants over 30 base scenes)
- resolvable fraction (ground truth): **0.633** (33 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.333
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_phase1_s1` (config hash `441c0e74`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passive_multiview_classifier | 90 | 0.6444 | 0.6444 | 0.0000 | 0.6560 | 0.6333 | 0.9394 | n/a | n/a | n/a | n/a | 0.2000 | 0.6667 | 0.7667 | 0.5000 |
| max_baseline_intervention | 90 | 0.5111 | 0.8364 | 0.3889 | 0.9825 | 0.9778 | 0.0000 | 0.2058 | 0.3934 | 0.0022 | 0.3174 | 0.3000 | 0.5000 | 0.7667 | 0.2667 |
| intervene3d_sum | 90 | 0.5667 | 0.8947 | 0.3667 | 1.0000 | 1.0000 | 0.0000 | 0.1835 | 0.3352 | 0.0022 | 0.1712 | 0.2533 | 0.5000 | 0.7667 | 0.4333 |
| intervene3d_maxmin | 90 | 0.5778 | 1.0000 | 0.4222 | 0.9561 | 0.9444 | 0.0000 | 0.1620 | 0.2991 | 0.0022 | 0.2143 | 0.1750 | 0.5000 | 0.7000 | 0.5333 |
| intervene3d_maxmin_forced | 90 | 0.7667 | 0.7667 | 0.0000 | 0.9561 | 0.9444 | 0.5455 | 0.1620 | 0.2991 | 0.0022 | 0.2143 | 0.1750 | 1.0000 | 0.7667 | 0.5333 |


## Phase 1 verdict

Ordering NOT EVALUATED: the three tiers were not all present in `methods`.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.2103, intercept = -0.8735, **R^2 = 0.8512** (n = 22)
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
- tables: `experiments\phase1_selector_study\run_20260829_013001_seed1_e188f489\tables\method_comparison.csv`, `experiments\phase1_selector_study\run_20260829_013001_seed1_e188f489\tables\method_comparison.md`

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
