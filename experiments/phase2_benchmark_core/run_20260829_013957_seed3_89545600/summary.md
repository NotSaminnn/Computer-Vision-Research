# phase2_benchmark_core -- run summary

- seed: `3`
- evaluation split: `test` (256 scene variants over 64 base scenes)
- resolvable fraction (ground truth): **0.426** (147 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.250
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_benchmark_core_s3` (config hash `eff1d712`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection | CEA_transmission |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_frame_classifier | 256 | 0.2500 | 0.2500 | 0.0000 | 0.5019 | 0.4258 | 0.0000 | n/a | n/a | n/a | n/a | 0.0000 | 0.2812 | 0.0000 | 0.2188 | 0.5000 |
| passive_multiview_classifier | 256 | 0.5195 | 0.5195 | 0.0000 | 0.8336 | 0.4258 | 0.0204 | n/a | n/a | n/a | n/a | 0.2000 | 0.4219 | 0.7969 | 0.3125 | 0.5469 |
| max_baseline_intervention | 256 | 0.3086 | 0.9753 | 0.6836 | 0.8944 | 0.8906 | 0.0000 | 0.1377 | 0.2564 | 0.0000 | 0.2159 | 0.3000 | 0.0469 | 0.7812 | 0.3594 | 0.0469 |
| intervene3d_sum | 256 | 0.3359 | 0.9773 | 0.6562 | 0.9530 | 0.9180 | 0.0000 | 0.1274 | 0.2273 | 0.0000 | 0.1526 | 0.2913 | 0.0469 | 0.7812 | 0.4219 | 0.0938 |
| intervene3d_maxmin | 256 | 0.3359 | 1.0000 | 0.6641 | 0.8612 | 0.9102 | 0.0000 | 0.0971 | 0.1826 | 0.0000 | 0.2930 | 0.1919 | 0.0469 | 0.6406 | 0.6094 | 0.0469 |
| intervene3d_maxmin_forced | 256 | 0.8477 | 0.8477 | 0.0000 | 0.8612 | 0.9102 | 0.4490 | 0.0971 | 0.1826 | 0.0000 | 0.2930 | 0.1919 | 1.0000 | 0.7812 | 0.6094 | 1.0000 |


## Phase 1 verdict

Basis: CEA with abstention counted as incorrect; the intervention-aware tier uses the forced-choice variant 'intervene3d_sum' so all three tiers are forced-choice.

- single-frame CEA        : **0.250** (chance = 0.250)
- passive multi-view CEA  : **0.520**
- intervention-aware CEA  : **0.336**

Ordering `single-frame <= passive <= intervention-aware` holds: **False** (strictly increasing: False).

With abstention enabled the full method reports CEA(all) = 0.336, CEA(committed) = 0.977 at an abstention rate of 0.656. The gap between those two numbers is the abstention mechanism declining to name a mechanism on cases no allowed action can resolve.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1584, intercept = 1.3227, **R^2 = 0.7752** (n = 41)
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
- tables: `experiments\phase2_benchmark_core\run_20260829_013957_seed3_89545600\tables\method_comparison.csv`, `experiments\phase2_benchmark_core\run_20260829_013957_seed3_89545600\tables\method_comparison.md`

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
