# phase1_problem_existence -- run summary

- seed: `2`
- evaluation split: `test` (96 scene variants over 32 base scenes)
- resolvable fraction (ground truth): **0.719** (27 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.333
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_phase1_s2` (config hash `9982e1e2`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_frame_classifier | 96 | 0.3333 | 0.3333 | 0.0000 | 0.4750 | 0.7188 | 0.0000 | n/a | n/a | n/a | n/a | 0.0000 | 0.2500 | 0.3438 | 0.4062 |
| passive_multiview_classifier | 96 | 0.6562 | 0.6562 | 0.0000 | 0.6951 | 0.7188 | 0.9259 | n/a | n/a | n/a | n/a | 0.2000 | 0.8125 | 0.8438 | 0.3125 |
| random_intervention | 96 | 0.4688 | 0.7500 | 0.3750 | 0.9348 | 0.9062 | 0.0000 | 0.2101 | 0.3703 | 0.0000 | 0.5860 | 0.1534 | 0.5938 | 0.6875 | 0.1250 |
| max_baseline_intervention | 96 | 0.5312 | 0.7846 | 0.3229 | 0.9710 | 0.9583 | 0.0000 | 0.2089 | 0.3670 | 0.0043 | 0.4235 | 0.3000 | 0.5938 | 0.8438 | 0.1562 |
| entropy_nbv | 96 | 0.4167 | 0.7018 | 0.4062 | 0.9130 | 0.8750 | 0.0000 | 0.2229 | 0.3884 | 0.0028 | 0.6979 | 0.0873 | 0.5938 | 0.5938 | 0.0625 |
| intervene3d_no_hypothesis_conditioning | 96 | 0.0000 | n/a | 1.0000 | 0.5000 | 0.2812 | 0.0000 | 0.3971 | 0.7114 | n/a | 0.6559 | 0.0500 | 0.0000 | 0.0000 | 0.0000 |
| intervene3d_no_abstention | 96 | 0.7083 | 0.7083 | 0.0000 | 1.0000 | 1.0000 | 0.8889 | 0.1848 | 0.3180 | 0.0000 | 0.3155 | 0.2766 | 1.0000 | 0.8438 | 0.2812 |
| intervene3d_noisy_encoder | 96 | 0.5729 | 0.7971 | 0.2812 | 1.0000 | 1.0000 | 0.0000 | 0.1902 | 0.3377 | 0.0001 | 0.3159 | 0.2776 | 0.5938 | 0.8438 | 0.2812 |
| intervene3d | 96 | 0.5729 | 0.7971 | 0.2812 | 1.0000 | 1.0000 | 0.0000 | 0.1848 | 0.3180 | 0.0000 | 0.3155 | 0.2766 | 0.5938 | 0.8438 | 0.2812 |


## Phase 1 verdict

Basis: CEA with abstention counted as incorrect; the intervention-aware tier uses the forced-choice variant 'intervene3d_no_abstention' so all three tiers are forced-choice.

- single-frame CEA        : **0.333** (chance = 0.333)
- passive multi-view CEA  : **0.656**
- intervention-aware CEA  : **0.708**

Ordering `single-frame <= passive <= intervention-aware` holds: **True** (strictly increasing: True).

With abstention enabled the full method reports CEA(all) = 0.573, CEA(committed) = 0.797 at an abstention rate of 0.281. The gap between those two numbers is the abstention mechanism declining to name a mechanism on cases no allowed action can resolve.

## MCRB theory validation

- measurement: homography-compensated lateral resolving baseline
- linear fit of `1/B_min` on `f|1/Z1 - 1/Z2|`: slope = 0.1477, intercept = 1.6862, **R^2 = 0.8334** (n = 22)
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
- tables: `experiments\phase1_problem_existence\run_20260828_231735_seed2_42bc7de9\tables\method_comparison.csv`, `experiments\phase1_problem_existence\run_20260828_231735_seed2_42bc7de9\tables\method_comparison.md`

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
