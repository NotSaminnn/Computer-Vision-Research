# phase2_benchmark_core -- run summary

- seed: `2`
- evaluation split: `test` (256 scene variants over 64 base scenes)
- resolvable fraction (ground truth): **0.422** (148 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.250
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_benchmark_core_s2` (config hash `634f5eb6`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection | CEA_transmission |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_frame_classifier | 256 | 0.2500 | 0.2500 | 0.0000 | 0.5040 | 0.4219 | 0.0000 | n/a | n/a | n/a | n/a | 0.0000 | 0.0156 | 0.1094 | 0.0000 | 0.8750 |
| passive_multiview_classifier | 256 | 0.4961 | 0.4961 | 0.0000 | 0.7885 | 0.4219 | 0.0000 | n/a | n/a | n/a | n/a | 0.2000 | 0.3438 | 0.7500 | 0.3750 | 0.5156 |
| max_baseline_intervention | 256 | 0.2969 | 0.9744 | 0.6953 | 0.8757 | 0.8828 | 0.0000 | 0.1253 | 0.2194 | 0.0000 | 0.2132 | 0.3000 | 0.0625 | 0.7344 | 0.3125 | 0.0781 |
| intervene3d_sum | 256 | 0.3086 | 0.9753 | 0.6836 | 0.9243 | 0.8945 | 0.0000 | 0.1259 | 0.2211 | 0.0003 | 0.1607 | 0.2853 | 0.0625 | 0.7344 | 0.3125 | 0.1250 |
| intervene3d_maxmin | 256 | 0.3086 | 1.0000 | 0.6914 | 0.8325 | 0.8867 | 0.0000 | 0.0841 | 0.1495 | 0.0000 | 0.3072 | 0.1621 | 0.0625 | 0.5938 | 0.5156 | 0.0625 |
| intervene3d_maxmin_forced | 256 | 0.8125 | 0.8125 | 0.0000 | 0.8325 | 0.8867 | 0.3378 | 0.0841 | 0.1495 | 0.0000 | 0.3072 | 0.1621 | 1.0000 | 0.7344 | 0.5156 | 1.0000 |


## Phase 1 verdict

Basis: CEA with abstention counted as incorrect; the intervention-aware tier uses the forced-choice variant 'intervene3d_sum' so all three tiers are forced-choice.

- single-frame CEA        : **0.250** (chance = 0.250)
- passive multi-view CEA  : **0.496**
- intervention-aware CEA  : **0.309**

Ordering `single-frame <= passive <= intervention-aware` holds: **False** (strictly increasing: False).

With abstention enabled the full method reports CEA(all) = 0.309, CEA(committed) = 0.975 at an abstention rate of 0.684. The gap between those two numbers is the abstention mechanism declining to name a mechanism on cases no allowed action can resolve.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1873, intercept = 0.2481, **R^2 = 0.8418** (n = 39)
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
- tables: `experiments\phase2_benchmark_core\run_20260829_013731_seed2_89545600\tables\method_comparison.csv`, `experiments\phase2_benchmark_core\run_20260829_013731_seed2_89545600\tables\method_comparison.md`

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
