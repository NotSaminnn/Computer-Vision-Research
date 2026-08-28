# Claude Code Prompt — Intervene3D Preliminary Codebase + Reproducible Experimental Framework

## Role

You are the lead research engineer responsible for turning the attached research specification into a **reproducible, executable preliminary research codebase**.

The source research document is:

- `research(1).md`
- Research concept: **Intervene3D: Interventional Identifiability of Physical Geometry under Optical Ambiguity**

You must read the entire `research(1).md` before making implementation decisions. Do not rely only on a summary or the first part of the file.

The research document is the conceptual source of truth for the project. However, it was written as a research proposal/literature synthesis rather than as an implementation specification. Your job is to convert it into a technically coherent, modular, reproducible codebase.

---

# 1. Primary Objective

Build the **preliminary Intervene3D codebase first**.

Do NOT immediately attempt to build the complete final CVPR-scale system.

The first milestone is a clean, executable research repository that can:

1. create the environment;
2. download/prepare the required datasets where licensing/access permits;
3. validate datasets;
4. generate a controlled synthetic Intervene3D-style benchmark;
5. represent optical hypotheses and camera interventions;
6. implement the analytical optical transition layer;
7. implement a minimal hypothesis-conditioned geometry transition model;
8. implement separability/identifiability calculations;
9. implement belief updating;
10. implement intervention selection;
11. implement the abstention/non-identifiability mechanism;
12. run a complete smoke test from a clean environment;
13. run at least one tiny end-to-end experiment;
14. save every artifact reproducibly;
15. generate publication-quality IEEE-style figures;
16. make all experiment outputs traceable to configuration, code version, seed, and dataset version.

The initial codebase must be designed so that larger models, datasets, and experiments can later be plugged in without restructuring the repository.

---

# 2. IMPORTANT: Multi-Agent / Multi-Subagent Workflow

Use multiple specialized subagents whenever the environment supports them.

Do not delegate everything to one generic agent.

Create a research-engineering workflow approximately like this:

### Agent A — Research Specification Auditor

Responsibilities:

- read the entire `research(1).md`;
- extract every proposed equation, metric, dataset, baseline, ablation, experiment, and implementation dependency;
- identify ambiguities and contradictions;
- produce a structured implementation requirements list;
- identify which claims are conceptual versus experimentally testable.

Deliverable:

`docs/RESEARCH_SPEC_AUDIT.md`

---

### Agent B — Literature / Cross-Research Auditor

Perform fresh cross-research using authoritative sources.

The research file contains literature claims up to August 2026. **Do not blindly trust them.**

Verify:

- paper titles;
- authors;
- venue/year;
- official paper pages;
- repositories;
- official project pages;
- dataset availability;
- dataset licenses;
- code availability;
- pretrained checkpoint availability;
- computational requirements;
- whether a cited method is actually usable as a baseline.

Search for work published or updated after the original research document's cutoff if available.

Especially audit novelty threats involving:

- multilayer depth;
- mirror/glass physical depth;
- optical ambiguity;
- visual illusions;
- active vision / next-best-view;
- intervention-based hypothesis verification;
- world models;
- geometry world models;
- counterfactual visual prediction;
- causal/interventional identifiability;
- transparent/reflective scene reconstruction.

Do not conclude "novel" merely because no obvious paper was found.

Deliverables:

`docs/LITERATURE_CROSS_RESEARCH.md`

`docs/NOVELTY_RISK_REGISTER.md`

The risk register must contain:

| Claim | Existing work | Degree of overlap | Evidence | Safe wording |
|---|---|---|---|---|

---

### Agent C — Dataset / Data Engineering Auditor

For every proposed dataset:

- verify official source;
- determine download mechanism;
- determine license;
- determine whether automated downloading is permitted;
- determine required preprocessing;
- determine expected directory structure;
- determine metadata/annotation format;
- identify storage requirements;
- identify whether authentication/manual download is necessary;
- design checksum validation;
- design dataset manifests.

Priority datasets from the research document:

1. TransPhy3D
2. LayeredDepth
3. 3D Visual Illusion dataset
4. Mirror3D
5. ClearPose / DREDS
6. MAGD
7. MVMD
8. MultiDepth-3k / MD-3k
9. DepthFocus data if legally/technically accessible
10. Booster
11. GIFT benchmark if accessible
12. PDI-Dataset
13. optional generic geometry/video datasets

Do not download large datasets automatically during the first smoke test.

The code must support:

- tiny/sample mode;
- full mode;
- resume;
- checksum validation;
- clear failure messages;
- manual-download instructions when automated download is not allowed.

Deliverable:

`docs/DATASET_MATRIX.md`

---

### Agent D — Software Architecture Agent

Design the repository architecture.

Focus on:

- package boundaries;
- configuration system;
- dataset interfaces;
- model interfaces;
- optical hypothesis abstractions;
- intervention abstractions;
- metric interfaces;
- experiment runners;
- artifact logging;
- visualization interfaces;
- testing strategy.

Deliverable:

`docs/SOFTWARE_ARCHITECTURE.md`

---

### Agent E — Experimental Methodology Agent

Convert the research proposal into executable experiments.

Design:

- smoke test;
- sanity checks;
- Phase 1 problem-existence experiment;
- benchmark-core experiments;
- analytical optics validation;
- world-model experiments;
- intervention-selection experiments;
- abstention experiments;
- generalization experiments;
- ablations;
- external benchmark evaluation.

For every experiment define:

- research question;
- hypothesis;
- inputs;
- outputs;
- baseline;
- model;
- dataset;
- configuration;
- metrics;
- expected artifact files;
- required plots;
- success/failure criteria.

Deliverable:

`docs/EXPERIMENT_PLAN.md`

---

### Agent F — Reproducibility / MLOps Agent

Design:

- seed handling;
- deterministic settings where possible;
- environment locking;
- configuration hashing;
- Git commit capture;
- dataset manifest capture;
- command logging;
- hardware/software metadata;
- run manifests;
- checkpoints;
- metrics;
- figure generation;
- result aggregation.

Deliverable:

`docs/REPRODUCIBILITY.md`

---

### Agent G — Visualization / Publication Agent

Design all plots before experiments are implemented.

Every important result must have a relevant visualization.

Use an **IEEE-style scientific plotting system** throughout.

Prefer a dedicated plotting module with a centralized publication configuration.

Figures should support:

- single-column and double-column IEEE layouts;
- readable fonts;
- vector PDF output where practical;
- high-resolution raster output when needed;
- consistent typography;
- mathematical notation;
- grayscale-safe/readable presentation;
- no unnecessary decoration;
- reproducible figure generation from saved result files.

Do not manually create plots inside individual experiments.

Deliverable:

`docs/VISUALIZATION_PLAN.md`

---

# 3. Required Development Order

Follow this exact high-level order.

## Phase 0 — Read and audit

Before coding:

1. read the complete `research(1).md`;
2. produce the research audit;
3. perform cross-research;
4. produce the dataset matrix;
5. design the software architecture;
6. design the experiment plan;
7. design reproducibility and visualization systems.

Do not make unsupported scientific assumptions.

---

# 4. Phase 1 — Build the Preliminary Codebase

After the audit, immediately build the preliminary repository.

The repository should resemble:

```text
Intervene3D/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml
├── configs/
│   ├── default.yaml
│   ├── smoke_test.yaml
│   ├── synthetic/
│   ├── experiments/
│   ├── datasets/
│   └── models/
├── scripts/
│   ├── setup_environment.sh
│   ├── download_datasets.sh
│   ├── validate_datasets.py
│   ├── generate_synthetic_data.py
│   ├── run_smoke_test.sh
│   ├── run_experiment.py
│   ├── aggregate_results.py
│   └── generate_all_figures.py
├── src/
│   └── intervene3d/
│       ├── __init__.py
│       ├── config/
│       ├── data/
│       ├── geometry/
│       ├── optics/
│       ├── hypotheses/
│       ├── interventions/
│       ├── models/
│       ├── inference/
│       ├── metrics/
│       ├── experiments/
│       ├── visualization/
│       ├── reproducibility/
│       └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── manifests/
├── checkpoints/
├── experiments/
├── figures/
├── results/
├── logs/
├── docs/
└── notebooks/
```

You may improve this structure if there is a technically superior alternative, but preserve the principles.

---

# 5. Experiment Directory Rule — MANDATORY

**ALL experiments must live under ONE top-level experiment directory.**

Use:

```text
experiments/
```

Every individual run must have its own immutable run directory.

For example:

```text
experiments/
└── phase1_problem_existence/
    ├── run_20260828_230000_seed42_a1b2c3/
    │   ├── config.yaml
    │   ├── command.txt
    │   ├── git_commit.txt
    │   ├── environment.txt
    │   ├── dataset_manifest.json
    │   ├── run_manifest.json
    │   ├── logs/
    │   ├── checkpoints/
    │   ├── predictions/
    │   ├── metrics/
    │   ├── tables/
    │   └── figures/
    └── ...
```

Never overwrite an old experiment run.

A run directory must contain enough information for another researcher to understand exactly how it was produced.

---

# 6. Reproducibility Requirements — MANDATORY

Every experiment must record:

- random seed;
- Python version;
- OS;
- GPU;
- CUDA version;
- PyTorch version;
- installed package versions;
- Git commit;
- configuration;
- configuration hash;
- dataset manifest;
- dataset checksums where possible;
- model checkpoint identifiers;
- command used;
- timestamp;
- hostname or machine metadata where appropriate;
- relevant environment variables;
- deterministic settings.

Use a single reproducibility utility rather than duplicating logic.

Provide:

```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

and make it automatically create a unique run directory.

---

# 7. Configuration System

Do not hard-code experiment parameters.

Use YAML or another transparent configuration format.

Every experiment must be configuration-driven.

Example:

```yaml
experiment:
  name: phase1_problem_existence
  seed: 42

data:
  dataset: synthetic_intervene3d
  split: test
  num_scenes: 32

camera:
  intrinsics:
    fx: 500
    fy: 500
    cx: 320
    cy: 240

interventions:
  candidate_actions:
    - ...

model:
  geometry_encoder: mock
  transition_model: analytical
  hidden_dim: 128

metrics:
  - causal_explanation_accuracy
  - identifiability_auroc
  - fpcr

visualization:
  ieee: true
  formats:
    - pdf
    - png
```

The exact schema may be improved.

---

# 8. Preliminary Synthetic Benchmark Must Work Without External Datasets

This is critical.

Before depending on any external dataset, implement a tiny synthetic Intervene3D environment.

Start with the simplest scientifically meaningful cases:

1. Direct physical 3D scene
2. Static planar display
3. Planar mirror

Then add:

4. transmission through planar glass
5. mixed reflection/transmission

The initial synthetic benchmark should contain:

- scene ID;
- hypothesis label;
- camera poses;
- camera intrinsics;
- intervention actions;
- RGB observations where feasible;
- contact geometry;
- apparent geometry;
- optical mechanism;
- masks;
- resolvability label;
- ground-truth minimum resolving baseline where analytically defined.

The benchmark should be deterministic under a seed.

---

# 9. Analytical Optical Transition Layer

Implement the simplest physically interpretable transitions first.

## Direct geometry

Use the camera transformation:

```text
X' = R X + t
```

with appropriate coordinate conventions.

## Planar mirror

Implement a virtual reflected camera using a calibrated plane.

## Static planar display

Implement a screen-plane/projective transformation.

## Transmission

Begin with a controlled simplified planar transmission model.

## Mixed optics

Implement only after the simpler mechanisms are validated.

The architecture should expose something like:

```python
transition = optical_model.predict(
    state=geometry_state,
    hypothesis=hypothesis,
    action=action,
)
```

Do not hide all physics inside an opaque neural network.

---

# 10. Hypothesis Representation

Use an explicit hypothesis abstraction.

Initial hypothesis family:

```text
H_D = direct physical geometry
H_R = reflection
H_T = transmission/refraction
H_E = emissive/display-induced geometry
H_M = mixed optical mechanism
```

Do NOT treat "unidentifiable" as a physical hypothesis.

Unidentifiable is an inference outcome.

---

# 11. Intervention Representation

Represent camera actions explicitly.

An action should encode a controlled observer/camera intervention such as:

```text
ΔC ∈ SE(3)
```

Support:

- lateral translation;
- forward/backward translation;
- yaw;
- pitch;
- optional roll;
- bounded motion magnitude.

The action space must be configurable.

Do not assume arbitrary actions.

---

# 12. Identifiability and Separability

Implement:

```text
Δ_ij(a) =
D(
    p(O' | H_i, a),
    p(O' | H_j, a)
)
```

and action-set identifiability:

```text
I_A(H_i, H_j) =
max_{a ∈ A} Δ_ij(a)
```

Implement an epsilon-based decision:

```text
I_A < ε
    -> unresolved / non-identifiable
I_A >= ε
    -> potentially resolvable
```

Make the distance function modular.

Start with deterministic geometry-space distances where possible.

Later support:

```text
D =
λg D_geometry
+ λf D_feature
+ λm D_motion
+ λo D_occlusion
```

Do not pretend all terms are meaningful before they are validated.

---

# 13. Belief Updating

Implement:

```text
e_k =
D(F_{t+1}, F_hat_{t+1}^{k,a})
```

and a configurable belief update such as:

```text
p_{t+1}(H_k)
∝
exp(-β e_k) p_t(H_k)
```

Validate the implementation with unit tests.

Include numerical stability protections.

---

# 14. Abstention / Non-Identifiability

This is a core feature, not an optional classifier threshold.

The system must distinguish:

1. prediction uncertainty;
2. physical identifiability uncertainty.

Implement an explicit output object containing at least:

```text
hypothesis_probabilities
identifiability_score
resolvable
selected_action
contact_geometry
abstained
```

Example outcome:

```text
physical explanation unresolved under available visual evidence
```

Do not force a physical label when the action set cannot distinguish hypotheses.

---

# 15. Minimum Causal Resolving Baseline

Implement the theory-backed metric for the simple planar display vs real 3D case.

Use:

```text
Δu ≈ fB/Z
```

and:

```text
B_min >
δ /
[
f |1/Z1 - 1/Z2|
]
```

The implementation must document:

- assumptions;
- units;
- camera convention;
- perceptual threshold δ;
- when the equation is applicable;
- when it is not.

Do not generalize this equation beyond its assumptions.

Call the metric:

**MCRB — Minimum Causal Resolving Baseline**

unless the literature audit finds a naming collision requiring a safer name.

---

# 16. Metrics

Implement the research document's main metrics as reusable metric classes/functions.

Required:

### Causal Explanation Accuracy

```text
CEA = P(H_hat = H*)
```

Report overall and by mechanism.

### Physical Contact Depth Error

At minimum:

- AbsRel_contact
- RMSE_contact

### Identifiability AUROC

Predict whether the ambiguity is resolvable under the allowed action set.

### MCRB Error

```text
MAE_MCRB = |B_hat_min - B_GT_min|
```

### False Physical Certainty Rate

For non-identifiable cases:

```text
FPCR =
P(max_k p(H_k) > τ | y_id = 0)
```

Also implement optional:

### Intervention Regret

```text
Regret = Δ(a*) - Δ(a_hat)
```

Every metric must have tests.

---

# 17. Phase 1 Problem-Existence Experiment

This experiment must be implemented before attempting the complete model.

Compare:

```text
single-frame classifier
<
passive multi-view/video baseline
<
intervention-aware model
```

The objective is not to achieve impressive numbers.

The objective is to determine whether the proposed research problem is experimentally observable.

The experiment should test:

- appearance-matched direct/display/mirror cases;
- fixed reference viewpoint;
- controlled interventions;
- separability;
- explanation accuracy;
- false certainty;
- intervention effectiveness.

Generate figures showing:

1. initial-view similarity;
2. hypothesis separability vs baseline;
3. explanation accuracy;
4. identifiability prediction;
5. FPCR;
6. selected intervention;
7. predicted vs observed geometry response.

---

# 18. Smoke Test — ABSOLUTELY MANDATORY

Create:

```bash
scripts/run_smoke_test.sh
```

The smoke test must run from a clean environment and verify the complete pipeline.

The smoke test should be tiny and CPU-compatible if possible.

It must:

1. import the package;
2. validate configuration;
3. create a synthetic scene;
4. generate at least two competing hypotheses;
5. generate candidate interventions;
6. produce predictions;
7. calculate separability;
8. select an intervention;
9. simulate/observe the next view;
10. update belief;
11. calculate at least several metrics;
12. save a run directory;
13. generate at least one figure;
14. verify that the figure exists;
15. verify metrics exist;
16. verify the run manifest exists;
17. exit non-zero if any step fails.

The smoke test should be runnable with one command:

```bash
bash scripts/run_smoke_test.sh
```

The final README must clearly document expected smoke-test output.

---

# 19. Unit and Integration Tests

At minimum test:

### Geometry

- SE(3) transforms;
- inverse transforms;
- projection;
- coordinate conventions.

### Optics

- mirror plane transformation;
- virtual camera;
- display homography;
- simplified transmission.

### Hypotheses

- serialization;
- equality;
- labels;
- validation.

### Interventions

- valid/invalid poses;
- motion bounds;
- candidate generation.

### Identifiability

- known separable cases;
- known non-separable cases;
- epsilon behavior.

### Belief update

- normalization;
- numerical stability;
- correct posterior ordering.

### Metrics

- perfect prediction;
- wrong prediction;
- abstention;
- edge cases.

### Reproducibility

- same seed produces same synthetic sample;
- same configuration produces same deterministic result where expected.

---

# 20. Dataset Downloading and Preparation

Create robust scripts.

Examples:

```bash
bash scripts/download_datasets.sh --dataset transphy3d
bash scripts/download_datasets.sh --dataset layereddepth
bash scripts/download_datasets.sh --dataset mirror3d
```

Also support:

```bash
python scripts/validate_datasets.py --dataset transphy3d
```

and:

```bash
python scripts/validate_datasets.py --all
```

Rules:

- never silently download a huge dataset;
- show expected storage;
- support `--dry-run`;
- support `--sample`;
- support resume;
- validate checksums where available;
- preserve original files;
- write dataset manifests;
- record source URL and version;
- respect licenses;
- if a dataset requires manual download, clearly say so and provide exact instructions in documentation.

Do not bypass access restrictions.

---

# 21. Environment Setup

Provide both:

```text
pyproject.toml
requirements.txt
environment.yml
```

if appropriate.

Provide:

```bash
scripts/setup_environment.sh
```

It should:

1. create/describe the environment;
2. install dependencies;
3. verify PyTorch;
4. verify CUDA when available;
5. verify rendering dependencies;
6. verify package import;
7. run a small validation.

Document CPU-only and GPU-capable paths.

Avoid unnecessary dependencies.

---

# 22. Model Architecture for the Preliminary Version

Do not start with a giant transformer.

Implement modular interfaces:

```text
GeometryEncoder
TransitionModel
SeparabilityEstimator
BeliefUpdater
InterventionSelector
IdentifiabilityEstimator
```

The first implementation may use:

```text
GeometryEncoder:
    synthetic ground-truth / simple feature representation

TransitionModel:
    analytical optical model

SeparabilityEstimator:
    deterministic geometry distance

BeliefUpdater:
    likelihood-based update

InterventionSelector:
    maximum predicted hypothesis separability
```

Then create a minimal learned transition model:

```text
F_hat_{t+1} =
W_theta(F_t, H_k, a)
```

The learned model should initially be deliberately small.

Only after the pipeline is correct should you integrate a real geometry foundation model such as a suitable MoGe/VGGT-style representation.

---

# 23. Real Geometry Foundation Model Integration

Create an adapter interface.

Do not hard-code the whole repository to one external model.

For example:

```python
encoder = build_geometry_encoder(config)
features = encoder.encode(images, camera=camera)
```

Support:

```text
mock
ground_truth
moge
vggt_like
```

where realistically available.

Before selecting a specific foundation model:

- verify the current official repository;
- verify license;
- verify checkpoint availability;
- verify Python/PyTorch compatibility;
- verify GPU memory;
- verify inference commands.

If the chosen model cannot be installed reliably, the preliminary codebase must still work using the mock/analytical encoder.

---

# 24. Visualization — IEEE Scientific Style

Create one central plotting module.

For example:

```text
src/intervene3d/visualization/
├── ieee_style.py
├── geometry_plots.py
├── ambiguity_plots.py
├── intervention_plots.py
├── metric_plots.py
├── ablation_plots.py
└── export.py
```

Do not scatter matplotlib configuration across experiments.

Every figure should be reproducible from saved result files.

Required plot families include:

### Geometry

- depth maps;
- contact vs apparent geometry;
- error maps;
- point clouds where useful.

### Ambiguity

- hypothesis probability;
- pairwise separability matrix;
- ambiguity/resolvability distribution;
- initial-view similarity.

### Intervention

- camera trajectory;
- action utility;
- selected action;
- intervention regret;
- separability as a function of baseline.

### Metrics

- CEA;
- AbsRel/RMSE;
- Identifiability AUROC;
- MCRB MAE;
- FPCR.

### Ablations

- intervention strategy;
- hypothesis conditioning;
- analytical vs learned;
- geometry feature vs RGB;
- abstention enabled/disabled.

### Generalization

- unseen materials;
- unseen scenes;
- unseen optical compositions;
- camera/action noise.

At least one figure should visually communicate:

```text
same apparent geometry
        ↓
multiple hypotheses
        ↓
controlled intervention
        ↓
different predicted consequences
        ↓
observation
        ↓
belief update
        ↓
resolved OR abstained
```

---

# 25. Results Must Never Be Stored Only in Logs

Every experiment must produce machine-readable results.

Prefer:

```text
metrics.json
predictions.parquet / csv
summary.json
```

and human-readable:

```text
summary.md
```

Figures should be generated from those files.

Do not manually copy numbers into plots.

---

# 26. Experiment Registry

Implement a lightweight experiment registry.

Each run should have:

```json
{
  "experiment_name": "...",
  "run_id": "...",
  "seed": 42,
  "config_hash": "...",
  "git_commit": "...",
  "dataset_manifest": "...",
  "status": "success",
  "metrics_file": "...",
  "figures": [...]
}
```

The registry should make it possible to aggregate multiple runs.

---

# 27. Multi-Seed Evaluation

Any meaningful experiment must support:

```bash
--seed 1
--seed 2
--seed 3
```

and aggregate:

- mean;
- standard deviation;
- confidence intervals where appropriate.

Do not report one lucky seed as the final result.

The smoke test can use one seed.

---

# 28. Reproduction Command

Every successful experiment must print a reproduction command.

Example:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/phase1_problem_existence.yaml \
  --seed 42
```

The exact command must also be saved to:

```text
command.txt
```

---

# 29. Research Integrity Rules

These rules are mandatory.

### Never fabricate results.

If an experiment has not been run, label it:

```text
NOT RUN
```

### Never fabricate dataset availability.

If access cannot be verified:

```text
ACCESS UNVERIFIED
```

### Never fabricate a paper.

Every literature claim must be tied to an actual source.

### Never silently change the research question.

If implementation constraints require changing a component, document the change.

### Do not overclaim novelty.

The source research explicitly states that the following are NOT independently novel:

- multilayer depth;
- physical-vs-apparent depth;
- mirror/glass correction;
- reflection/transmission decomposition;
- camera motion as a cue;
- active next-best-view;
- counterfactual world models;
- causal identifiability generally.

Keep the novelty centered on:

**action-set-dependent identifiability of competing optical explanations + explicit non-identifiability prediction + matched counterfactual benchmark.**

---

# 30. Dataset Leakage Prevention

This is critical.

The new benchmark must not be evaluated using scenes that appeared in training.

For matched counterfactual data:

- split by underlying base scene;
- not merely by frame;
- not merely by camera pose;
- not merely by rendered image.

Document the split policy.

If synthetic variants of one underlying scene exist across hypotheses, all variants must remain in the same split.

---

# 31. Synthetic Data Generator

Build:

```text
src/intervene3d/data/synthetic/
```

with:

```text
scene_generator.py
camera_generator.py
optical_variants.py
trajectory_generator.py
ground_truth.py
dataset_writer.py
validator.py
```

Generate metadata such as:

```json
{
  "scene_id": "...",
  "base_scene_id": "...",
  "hypothesis": "reflection",
  "camera_pose": [...],
  "intrinsics": [...],
  "contact_geometry": "...",
  "apparent_geometry": "...",
  "action_set": [...],
  "resolvable": true,
  "mcrb": 0.12
}
```

Start small.

The generator must be able to produce:

```bash
python scripts/generate_synthetic_data.py \
    --config configs/synthetic/smoke.yaml
```

---

# 32. Benchmark Design

Eventually implement Intervene3D-Bench around:

```text
matched causal variants
+
controlled camera interventions
+
physical contact geometry
+
causal image-formation labels
+
resolvable/non-resolvable labels
+
minimum resolving motion
```

The benchmark must contain explicit non-identifiable examples.

Do not make every case solvable.

A benchmark in which every ambiguity can always be resolved would undermine the core scientific question.

---

# 33. Baseline Framework

Create a common baseline API.

Initial baseline families:

1. Static image classifier
2. Passive multi-view/video model
3. Random intervention
4. Maximum-baseline intervention
5. Generic entropy/NBV-style intervention
6. No hypothesis conditioning
7. Analytical-only
8. Learned-only
9. Full Intervene3D preliminary method

Do not implement every external paper immediately.

First implement the conceptual baselines that test the research hypothesis.

External published methods should be integrated only after their official implementations and licenses are verified.

---

# 34. Ablation Framework

Implement configurable ablations:

- single image only;
- random camera movement;
- maximum-baseline movement;
- generic uncertainty/NBV;
- no hypothesis conditioning;
- RGB instead of geometry features;
- learned-only transition;
- analytical-only transition;
- no abstention;
- no matched-counterfactual training;
- noisy camera actions.

A single configuration should be enough to activate/deactivate these components.

---

# 35. Generalization Framework

Prepare experiments for:

### Unseen materials

Train on common mirror/glass.

Test on:

- tinted glass;
- curved mirrors;
- polished metal.

### Unseen scene categories

Train indoor.

Test:

- laboratory;
- office;
- outdoor storefront;
- other held-out categories.

### Unseen optical compositions

Train individual mechanisms.

Test mixed mechanisms.

### Action noise

Inject pose/action noise:

```text
ΔC + ε
```

### Unseen cameras

Vary:

- focal length;
- resolution;
- sensor parameters.

---

# 36. Recommended Initial Milestones

Do not proceed linearly into a huge implementation.

Use these gates.

## Gate 1

The synthetic environment and smoke test work.

If not, stop and fix.

## Gate 2

The analytical mirror/display/direct transitions are validated with unit tests and visualizations.

## Gate 3

The Phase 1 problem-existence experiment can run end-to-end.

## Gate 4

The results demonstrate whether intervention adds measurable information.

If not, investigate scientifically before expanding the architecture.

## Gate 5

Only then integrate external datasets.

## Gate 6

Only then integrate a real geometry foundation encoder.

## Gate 7

Only then scale the learned world model.

This prevents spending weeks implementing a sophisticated architecture before proving that the central experimental premise is measurable.

---

# 37. Commands That Must Exist

At minimum:

```bash
# environment
bash scripts/setup_environment.sh

# smoke test
bash scripts/run_smoke_test.sh

# generate tiny synthetic data
python scripts/generate_synthetic_data.py \
    --config configs/synthetic/smoke.yaml

# validate datasets
python scripts/validate_datasets.py --all

# run experiment
python scripts/run_experiment.py \
    --config configs/experiments/phase1_problem_existence.yaml \
    --seed 42

# aggregate runs
python scripts/aggregate_results.py \
    --experiment phase1_problem_existence

# regenerate figures
python scripts/generate_all_figures.py
```

Make the commands actually work.

Do not merely document commands that do not exist.

---

# 38. Documentation Requirements

Create at least:

```text
README.md
docs/
├── RESEARCH_SPEC_AUDIT.md
├── LITERATURE_CROSS_RESEARCH.md
├── NOVELTY_RISK_REGISTER.md
├── DATASET_MATRIX.md
├── SOFTWARE_ARCHITECTURE.md
├── EXPERIMENT_PLAN.md
├── REPRODUCIBILITY.md
├── VISUALIZATION_PLAN.md
├── DEVELOPMENT_ROADMAP.md
└── TROUBLESHOOTING.md
```

README must include:

- project motivation;
- research question;
- architecture;
- repository structure;
- installation;
- smoke test;
- synthetic benchmark;
- dataset acquisition;
- experiment execution;
- result locations;
- figure generation;
- reproducibility;
- known limitations.

---

# 39. Cross-Research Must Continue During Implementation

Do not treat literature research as a one-time step.

Before integrating a major component, perform a focused literature/code audit.

Examples:

Before choosing a geometry foundation model:

```text
search latest geometry foundation models 2026
```

Before implementing world modeling:

```text
search latest geometry world models 2026
```

Before claiming intervention novelty:

```text
search active vision hypothesis verification optical ambiguity
```

Before claiming benchmark novelty:

```text
search optical ambiguity benchmark physical apparent geometry intervention
```

Record important findings in:

```text
docs/LITERATURE_CROSS_RESEARCH.md
```

If new literature materially threatens the research idea, stop and report it rather than silently proceeding.

---

# 40. Final Deliverables for This First Task

The first task is complete only when all of the following exist:

### Research

- complete research specification audit;
- fresh literature cross-research;
- novelty risk register;
- dataset matrix.

### Codebase

- preliminary package;
- environment setup;
- configuration system;
- synthetic benchmark;
- optical hypothesis abstractions;
- intervention system;
- analytical transitions;
- separability;
- belief update;
- identifiability;
- abstention;
- metrics;
- visualization;
- experiment runner.

### Tests

- unit tests;
- integration tests;
- **smoke test**;
- smoke test passes.

### Reproducibility

- seeds;
- manifests;
- config hashes;
- Git metadata;
- environment metadata;
- run directories;
- reproducible commands.

### Documentation

- README;
- architecture;
- experiment plan;
- dataset instructions;
- reproduction instructions.

### First experiment

At least one tiny Phase 1 experiment must execute end-to-end.

### Figures

At least several meaningful IEEE-style figures must be generated automatically from saved results.

---

# 41. Definition of Done

Do NOT say "done" merely because files were created.

The project is considered complete for this preliminary milestone only if:

```text
clean environment
      ↓
setup script
      ↓
package installation
      ↓
synthetic data generation
      ↓
smoke test
      ↓
experiment runner
      ↓
unique experiment directory
      ↓
metrics
      ↓
plots
      ↓
reproduction metadata
```

works end-to-end.

Run the complete smoke test yourself.

Then run the first tiny research experiment yourself.

Inspect:

- logs;
- metrics;
- generated figures;
- metadata;
- reproducibility information.

Fix errors before reporting completion.

---

# 42. What NOT to Do in the First Implementation

Do not:

- build a giant diffusion model;
- build a generic video generator;
- build a full navigation system;
- build SLAM;
- add VLM reasoning unless experimentally necessary;
- add arbitrary object actions;
- implement every cited paper;
- download hundreds of gigabytes before validating the pipeline;
- hard-code dataset paths;
- hard-code GPU assumptions;
- put all code in notebooks;
- store experiments in random directories;
- overwrite results;
- manually edit result tables;
- claim performance without executing experiments;
- claim novelty without current literature verification.

The first goal is **scientific infrastructure + a minimal falsifiable prototype**, not maximum model complexity.

---

# 43. Expected First Prototype

The first working system should conceptually implement:

```text
Synthetic Scene
      ↓
Reference Observation I0
      ↓
Candidate Hypotheses
{Direct, Mirror, Display}
      ↓
Candidate Camera Actions
      ↓
Analytical Hypothesis-Conditioned Prediction
      ↓
Pairwise Separability
      ↓
Choose Best Intervention
      ↓
Generate/Observe Next View
      ↓
Compare Observation With Predictions
      ↓
Belief Update
      ↓
Identifiability Decision
      ├── Resolved → physical explanation + contact geometry
      └── Unresolved → explicit abstention
```

This is the minimum scientific loop.

---

# 44. Scientific Validation Philosophy

The implementation must be designed to answer:

### RQ1

Can competing physical explanations be distinguished through controlled observer motion?

### RQ2

Can the system recognize when no allowed intervention is sufficient?

### RQ3

Can intervention-aware hypothesis separation reduce motion compared with generic strategies?

### RQ4

Can identifiability-aware inference reduce confidently incorrect physical geometry predictions?

The first prototype does not need to answer all four completely.

It must, however, create the infrastructure required to answer them rigorously.

---

# 45. Output Format From You

At the end of the implementation session, provide a concise report containing:

```text
1. What was implemented
2. Repository structure
3. Environment/setup commands
4. Dataset status
5. Smoke-test command
6. Smoke-test result
7. First experiment command
8. First experiment result
9. Generated figures
10. Known limitations
11. Unimplemented components
12. Recommended next milestone
```

Do not claim a component works unless you actually executed it.

---

# 46. Priority Order

If time or compute is limited, use this exact priority:

```text
P0:
research audit
cross research
software architecture
environment
synthetic benchmark
smoke test

P1:
analytical optics
hypothesis representation
interventions
separability
belief update
abstention
metrics
IEEE visualization

P2:
Phase 1 experiment
baseline framework
reproducibility
multi-seed runs

P3:
dataset download/preprocessing
real benchmark adapters
geometry foundation model

P4:
learned transition/world model

P5:
large-scale benchmark
large experiments
full ablations
generalization
real-world controlled dataset
```

Never sacrifice P0/P1 correctness for P4/P5 complexity.

---

# 47. Final Instruction

Start now.

**First read the complete `research(1).md`.**

Then use multiple specialized subagents for the research, dataset, architecture, experiment, reproducibility, and visualization audits.

Then build the **preliminary codebase**.

Do not merely produce a conceptual plan.

Actually create the repository structure and implementation.

The first executable milestone is:

```bash
bash scripts/run_smoke_test.sh
```

and it MUST PASS.

Immediately after that, run the smallest meaningful Phase 1 experiment.

All experiments must live under:

```text
experiments/
```

with a separate immutable directory for every run.

All results must be reproducible.

All important results must be visualized.

All scientific figures must be generated through the centralized IEEE-style plotting system.

Perform fresh cross-research throughout the implementation and document anything that changes the novelty or experimental design.

The guiding principle is:

> **Build the smallest rigorous system that can falsify or support the Intervene3D hypothesis before building the large system.**
