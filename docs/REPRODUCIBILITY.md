# Reproducibility

One utility, `intervene3d.reproducibility`, used everywhere. No experiment
duplicates any of this logic.

---

## 1. What every run records

Written automatically by `create_run_directory` before the experiment starts, so
a crash cannot lose it:

| Item | Where |
|---|---|
| random seed | run directory name, `run_manifest.json`, registry |
| seed report (what was actually seeded, incl. torch if present) | `run_manifest.json.seed_report` |
| Python version, implementation, executable | `environment.{txt,json}` |
| OS and release | `environment.{txt,json}` |
| CPU, machine, core count, hostname | `environment.json.hardware` |
| GPU, CUDA version, CUDA availability, MPS | `environment.json.hardware` |
| PyTorch version (or `"not installed"`) | `environment.json.hardware` |
| tracked package versions | `environment.json.packages` |
| **full `pip freeze`-equivalent** | `environment.txt` |
| Git commit, branch, dirty flag, dirty file list, remote, describe | `git_commit.txt`, `environment.json.git` |
| configuration, verbatim | `config.yaml` |
| configuration hash (short + full) | run directory name, `run_manifest.json` |
| dataset manifest, statistics and config hash | `dataset_manifest.json` |
| dataset per-file SHA-256 checksums | the dataset's own `manifest.json` |
| the exact command | `command.txt` |
| the **reproduction** command | `command.txt`, `run_manifest.json`, and printed on success |
| UTC timestamp (start and finish), duration | `run_manifest.json` |
| relevant environment variables | `environment.json.environment_variables` |
| deterministic settings applied | `run_manifest.json.seed_report` |
| status (`success` / `failed`) and, on failure, the traceback | `run_manifest.json` |

The tracked environment variables are those that can change numerical results:
`PYTHONHASHSEED`, the four BLAS thread-count variables, `CUDA_VISIBLE_DEVICES`,
`CUBLAS_WORKSPACE_CONFIG`, `MPLBACKEND`, `INTERVENE3D_DATA_ROOT`.

---

## 2. Determinism

The preliminary pipeline is pure NumPy, so determinism comes from seeding rather
than from disabling nondeterministic kernels.

- `set_global_seed(seed)` seeds `random`, `numpy`, `PYTHONHASHSEED`, and — if
  PyTorch is installed — `torch`, `cudnn.deterministic`, `use_deterministic_algorithms`
  and `CUBLAS_WORKSPACE_CONFIG`. It **returns a report of what it configured**,
  which is archived, so "deterministic" is never an unverified assertion.
- `rng_for(seed, *stream)` gives named sub-streams via
  `default_rng([seed, *stream])`. This matters: deriving generators sequentially
  would mean that adding a new consumer perturbs the draws of every existing one,
  and a run's results would drift as the code grows. Covered by
  `test_rng_streams_are_independent`.

**Verified determinism guarantees** (all in `tests/unit/test_reproducibility.py`
and `tests/integration/test_pipeline.py`):

- the same seed produces the same synthetic scene (points, colours, markers, pose);
- regenerating a dataset with the same config and seed produces **byte-identical
  files** (checked via the manifest's per-file SHA-256);
- the full oracle separability computation is a pure function of (config, seed);
- a scene's split assignment is stable when the dataset grows.

---

## 3. Configuration hashing

`config_hash` is a SHA-256 over the canonical JSON with `_`-prefixed transient
keys and **`experiment.seed`** removed.

Excluding the seed is deliberate and was a bug once: with the seed in the hash,
three seeds of the same experiment produced three different hashes, and
`aggregate_results.py` — which by design refuses to average across differing
configurations — would have silently reported a single run as if it were the whole
study. The seed is recorded in three other places. Locked in by
`test_config_hash_is_seed_independent`.

---

## 4. Immutable run directories

```
experiments/<experiment_name>/run_<UTC>_seed<N>_<confighash8>/
```

**All experiments live under one top-level `experiments/` directory, and an
existing run is never overwritten.** If a directory of the same name exists, a
numeric suffix is appended rather than any file being replaced
(`test_runs_are_never_overwritten`).

Failures are recorded, not lost: a crashed run leaves `status: "failed"` and the
traceback in `run_manifest.json` and a row in the registry
(`test_failed_runs_are_recorded_not_silently_lost`).

---

## 5. Experiment registry

`experiments/registry.jsonl` — append-only, one JSON object per run:

```json
{"experiment_name": "...", "run_id": "...", "run_path": "...", "seed": 1,
 "config_hash": "...", "git_commit": "...", "dataset_manifest": "...",
 "status": "success", "metrics_file": "...", "figures": ["..."],
 "created_utc": "...", "finished_utc": "...", "duration_seconds": 27.9}
```

Append-only means aggregation is a plain file read and a crash leaves a legible
record rather than a hole.

---

## 6. Multi-seed evaluation

```bash
for s in 1 2 3; do
  python scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed $s
done
python scripts/aggregate_results.py --experiment phase1_problem_existence
```

Reports mean, standard deviation and a 95 % confidence interval using a
**Student-t** critical value, so three seeds are not reported as if they were
Gaussian. A single seed reports `ci95: NaN` with an explicit note rather than a
fabricated interval.

**Two seeding protocols, and the difference matters:**

| `data.reseed_with_experiment_seed` | What varies | What the error bars mean |
|---|---|---|
| `false` (default) | model-side stochasticity only | Deterministic methods show **zero** variance. That is a determinism check, not an error bar. |
| `true` (used by Phase 1) | **the benchmark is redrawn per seed** | Variation over the data-generating process. These are the error bars worth reporting. |

The first protocol was what the Phase 1 config used initially, and it produced
`± 0.000` on almost every metric — technically true, scientifically empty. The
config now uses the second.

Aggregation refuses to average across differing config hashes unless
`--all-hashes` is passed explicitly.

---

## 7. Dataset reproducibility

Every generated dataset writes a `manifest.json` containing the full generator
config, its hash, the seed, generation timestamp, the action space, the
identifiability settings, split statistics, benchmark statistics, and a
**SHA-256 for every file**.

`validate_dataset` re-checks the checksums plus five scientific invariants:
matched counterfactuals (max reference-view pixel deviation must be `0`), no split
leakage, presence of non-identifiable cases, presence of resolvable cases, and
array shape integrity.

---

## 8. Figures are reproducible from files

Figures are drawn *only* from `metrics/figure_data.json`. `generate_figures`
never sees a live object, so:

```bash
python scripts/generate_all_figures.py --run <run_dir>
```

rebuilds the whole figure set without re-running anything. No number is ever
transcribed into a plot by hand. Which style path was used (SciencePlots vs the
built-in fallback) is recorded in `metrics.json.style`, so a figure's appearance
is never a mystery.

---

## 9. Environment

| Path | Command | Notes |
|---|---|---|
| **CPU (default, fully supported)** | `bash scripts/setup_environment.sh` | NumPy, Matplotlib, PyYAML, SciencePlots. Everything in this milestone runs here. |
| **GPU (optional)** | `bash scripts/setup_environment.sh --gpu` | Adds PyTorch. Needed only for Gate 6 / Gate 7. |
| **Conda** | `conda env create -f environment.yml` | |
| **Verify only** | `bash scripts/setup_environment.sh --check-only` | |

The setup script verifies the interpreter, core dependencies, headless Matplotlib
rendering, the IEEE style (including whether LaTeX would be required — it is not),
PyTorch and CUDA if present, the package import, and finally runs a small
end-to-end validation of the analytical optics and the identifiability
computation.

Pins live in `requirements.txt` (runtime), `requirements-dev.txt` (tests, lint)
and `requirements-gpu.txt` (optional). Every run archives a full frozen
environment listing regardless.

---

## 10. Reproducing the reported Phase 1 results

```bash
bash scripts/setup_environment.sh
.venv/bin/python scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml
for s in 1 2 3; do
  .venv/bin/python scripts/run_experiment.py \
    --config configs/experiments/phase1_problem_existence.yaml --seed $s
done
.venv/bin/python scripts/aggregate_results.py --experiment phase1_problem_existence
```

Note that with `reseed_with_experiment_seed: true`, each seed generates its own
benchmark (`intervene3d_synth_phase1_s1`, `_s2`, `_s3`) — about 12 s each.
