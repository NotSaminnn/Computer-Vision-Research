# phase2_benchmark_core -- run summary

- seed: `1`
- evaluation split: `test` (224 scene variants over 56 base scenes)
- resolvable fraction (ground truth): **0.344** (147 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.250
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_benchmark_core_s1` (config hash `a40991ce`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection | CEA_transmission |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_frame_classifier | 224 | 0.2500 | 0.2500 | 0.0000 | 0.5603 | 0.3438 | 0.0000 | n/a | n/a | n/a | n/a | 0.0000 | 0.5000 | 0.0179 | 0.4821 | 0.0000 |
| passive_multiview_classifier | 224 | 0.4643 | 0.4643 | 0.0000 | 0.7428 | 0.3438 | 0.0136 | n/a | n/a | n/a | n/a | 0.2000 | 0.5536 | 0.6250 | 0.0893 | 0.5893 |
| max_baseline_intervention | 224 | 0.2143 | 0.9600 | 0.7768 | 0.8726 | 0.8795 | 0.0000 | 0.1920 | 0.3567 | 0.0000 | 0.1868 | 0.3000 | 0.0357 | 0.6071 | 0.1786 | 0.0357 |
| intervene3d_sum | 224 | 0.2723 | 0.9839 | 0.7232 | 0.9591 | 0.9330 | 0.0000 | 0.1593 | 0.2887 | 0.0000 | 0.1097 | 0.2898 | 0.0357 | 0.6071 | 0.3750 | 0.0714 |
| intervene3d_maxmin | 224 | 0.2455 | 1.0000 | 0.7545 | 0.8254 | 0.9018 | 0.0000 | 0.1519 | 0.2831 | 0.0000 | 0.3238 | 0.1533 | 0.0357 | 0.4821 | 0.4286 | 0.0357 |
| intervene3d_maxmin_forced | 224 | 0.7634 | 0.7634 | 0.0000 | 0.8254 | 0.9018 | 0.2993 | 0.1519 | 0.2831 | 0.0000 | 0.3238 | 0.1533 | 1.0000 | 0.6250 | 0.4286 | 1.0000 |


## Phase 1 verdict

Basis: CEA with abstention counted as incorrect; the intervention-aware tier uses the forced-choice variant 'intervene3d_sum' so all three tiers are forced-choice.

- single-frame CEA        : **0.250** (chance = 0.250)
- passive multi-view CEA  : **0.464**
- intervention-aware CEA  : **0.272**

Ordering `single-frame <= passive <= intervention-aware` holds: **False** (strictly increasing: False).

With abstention enabled the full method reports CEA(all) = 0.272, CEA(committed) = 0.984 at an abstention rate of 0.723. The gap between those two numbers is the abstention mechanism declining to name a mechanism on cases no allowed action can resolve.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1438, intercept = 1.5481, **R^2 = 0.8534** (n = 29)
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
- tables: `experiments\phase2_benchmark_core\run_20260829_013513_seed1_89545600\tables\method_comparison.csv`, `experiments\phase2_benchmark_core\run_20260829_013513_seed1_89545600\tables\method_comparison.md`

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
