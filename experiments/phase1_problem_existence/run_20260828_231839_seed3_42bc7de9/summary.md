# phase1_problem_existence -- run summary

- seed: `3`
- evaluation split: `test` (87 scene variants over 29 base scenes)
- resolvable fraction (ground truth): **0.632** (32 scenes are non-identifiable under the allowed action set)
- chance-level accuracy: 0.333
- action set: 33 candidate interventions; epsilon = 1 px-equivalent; tau = 0.45
- dataset: `intervene3d_synth_phase1_s3` (config hash `2fd7b0f4`)

## Method comparison

| method | n | CEA | CEA_committed | abstention_rate | identifiability_auroc | resolvability_accuracy | FPCR | AbsRel_contact | RMSE_contact | MAE_MCRB | normalised_regret | motion_cost | CEA_direct | CEA_emissive | CEA_reflection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_frame_classifier | 87 | 0.3333 | 0.3333 | 0.0000 | 0.4932 | 0.6322 | 0.0000 | n/a | n/a | n/a | n/a | 0.0000 | 0.4138 | 0.1724 | 0.4138 |
| passive_multiview_classifier | 87 | 0.5632 | 0.5632 | 0.0000 | 0.7449 | 0.6322 | 0.6562 | n/a | n/a | n/a | n/a | 0.2000 | 0.6897 | 0.6897 | 0.3103 |
| random_intervention | 87 | 0.3678 | 0.7111 | 0.4828 | 0.9091 | 0.8851 | 0.0000 | 0.2604 | 0.4562 | 0.0000 | 0.5264 | 0.1487 | 0.4828 | 0.5862 | 0.0345 |
| max_baseline_intervention | 87 | 0.4713 | 0.8200 | 0.4253 | 0.9545 | 0.9425 | 0.0000 | 0.2186 | 0.3880 | 0.0000 | 0.3386 | 0.3000 | 0.4828 | 0.6897 | 0.2414 |
| entropy_nbv | 87 | 0.3333 | 0.6744 | 0.5057 | 0.8909 | 0.8621 | 0.0000 | 0.2576 | 0.4487 | 0.0000 | 0.6207 | 0.0946 | 0.4828 | 0.4828 | 0.0345 |
| intervene3d_no_hypothesis_conditioning | 87 | 0.0000 | n/a | 1.0000 | 0.5000 | 0.3678 | 0.0000 | 0.3910 | 0.6984 | n/a | 0.6026 | 0.0500 | 0.0000 | 0.0000 | 0.0000 |
| intervene3d_no_abstention | 87 | 0.7011 | 0.7011 | 0.0000 | 1.0000 | 1.0000 | 0.8125 | 0.1912 | 0.3243 | 0.0000 | 0.2187 | 0.2466 | 1.0000 | 0.6897 | 0.4138 |
| intervene3d_noisy_encoder | 87 | 0.5172 | 0.8333 | 0.3793 | 0.9909 | 0.9885 | 0.0000 | 0.1970 | 0.3456 | 0.0002 | 0.2332 | 0.2500 | 0.4828 | 0.6552 | 0.4138 |
| intervene3d | 87 | 0.5287 | 0.8364 | 0.3678 | 1.0000 | 1.0000 | 0.0000 | 0.1912 | 0.3243 | 0.0000 | 0.2187 | 0.2466 | 0.4828 | 0.6897 | 0.4138 |


## Phase 1 verdict

Basis: CEA with abstention counted as incorrect; the intervention-aware tier uses the forced-choice variant 'intervene3d_no_abstention' so all three tiers are forced-choice.

- single-frame CEA        : **0.333** (chance = 0.333)
- passive multi-view CEA  : **0.563**
- intervention-aware CEA  : **0.701**

Ordering `single-frame <= passive <= intervention-aware` holds: **True** (strictly increasing: True).

With abstention enabled the full method reports CEA(all) = 0.529, CEA(committed) = 0.836 at an abstention rate of 0.368. The gap between those two numbers is the abstention mechanism declining to name a mechanism on cases no allowed action can resolve.

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
- tables: `experiments\phase1_problem_existence\run_20260828_231839_seed3_42bc7de9\tables\method_comparison.csv`, `experiments\phase1_problem_existence\run_20260828_231839_seed3_42bc7de9\tables\method_comparison.md`

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
