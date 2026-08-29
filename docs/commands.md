# Commands

Every experiment in this repository and the command that runs it. Timings
measured 2026-08-29 on Windows 11 / Python 3.10.11 / Ryzen 9 9900X / RTX 5070.

Status of each experiment lives in `docs/EXPERIMENT_PLAN.md`; what is and is not
claimed lives in `README.md`. This file is commands only.

## The interpreter

`$PY` below is the virtualenv's Python. `make` and the shell scripts resolve it
themselves; when typing by hand use `.venv/bin/python` (POSIX) or
`.venv/Scripts/python.exe` (Windows). `PYTHON=/path/to/python` overrides
everywhere.

---

## 1. Setup

| Command | Does | Time |
|---|---|---|
| `bash scripts/setup_environment.sh` | create `.venv`, install, verify deps + headless render + IEEE style + package import | ~40 s |
| `bash scripts/setup_environment.sh --check-only` | verify an existing environment | ~10 s |
| `$PY -m pip install -e ".[data]"` | `pyarrow` + `Pillow`, needed only to **read** external datasets | ~10 s |
| `$PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` | GPU extras. **cu128 is required for Blackwell / sm_120** (RTX 50-series) | ~3 min |
| `conda env create -f environment.yml` | conda alternative, CPU only | — |

Confirm the GPU can actually execute, not merely that it is detected:

```bash
$PY -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name,f'sm_{p.major}{p.minor}',torch.cuda.get_arch_list())"
```

`sm_120` must appear in the arch list or every kernel fails at launch.

---

## 2. Tests

| Command | Does | Time |
|---|---|---|
| `make test` | full suite — **276 tests** | ~35 s |
| `make unit` / `make integration` | one tier | ~25 s / ~5 s |
| `bash scripts/run_smoke_test.sh` | the mandatory 16-step end-to-end gate; non-zero exit if any step fails | ~40 s |
| `make lint` | `ruff check src scripts tests` | ~2 s |

On Windows add `-p no:cacheprovider` to silence a `.pytest_cache` permission warning.

---

## 3. Synthetic benchmark

Generated, not downloaded; deterministic under `dataset.seed`.

| Command | Produces | Time |
|---|---|---|
| `$PY scripts/generate_synthetic_data.py --config configs/synthetic/smoke.yaml` | 36 variants / 12 base scenes | ~2 s |
| `$PY scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml` | **288 variants / 96 base scenes, 103 non-identifiable** | ~22 s |
| `$PY scripts/generate_synthetic_data.py --config configs/synthetic/benchmark_core.yaml` | 768 variants / 192 base scenes, adds `transmission` | ~60 s |

Flags: `--seed N`, `--dry-run`, `--force`, `--no-validate`, `--output-root DIR`,
`--set dotted.key=value`.

Larger draw (generation is single-threaded; scale is linear):

```bash
$PY scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml \
  --set dataset.num_base_scenes=1024 --set dataset.name=intervene3d_synth_phase1_xl
```

---

## 4. Dataset validation

| Command | Does |
|---|---|
| `$PY scripts/validate_datasets.py --list` | registry: status, licence, download permission, local presence |
| `$PY scripts/validate_datasets.py --all` | validate every synthetic dataset and report every external one |
| `$PY scripts/validate_datasets.py --synthetic <dir>` | one synthetic dataset |
| `$PY scripts/validate_datasets.py --dataset <key>` | one external dataset against its `manifest.json` |
| `$PY scripts/validate_datasets.py --all --report out.json` | machine-readable |

Add `--no-checksums` to skip hashing.

---

## 5. External dataset acquisition

```bash
bash scripts/download_datasets.sh --list                                # registry
bash scripts/download_datasets.sh --dataset <key> --dry-run             # files + exact bytes
bash scripts/download_datasets.sh --dataset <key> --variant <v> --yes   # fetch
```

Transfers above 1 GB require `--yes`. `--workers N` spreads spare workers across
byte ranges of a single large file. Each fetch writes `manifest.json` pinning the
resolved upstream commit.

| Dataset | Variant | Command suffix | Size | Licence |
|---|---|---|---|---|
| LayeredDepth | `validation` | `--dataset layereddepth --variant validation --yes` | 4.13 GB · 300 ex. | CC0-1.0 |
| LayeredDepth | `test` | `--dataset layereddepth --variant test --yes` | 15.15 GB · 1,200 ex., images only | CC0-1.0 |
| LayeredDepth-Syn | `validation` | `--dataset layereddepth_syn --variant validation` | 0.80 GB · 500 ex. | BSD-3-Clause |
| LayeredDepth-Syn | `train` | `--dataset layereddepth_syn --variant train --yes` | 23.95 GB · 14,800 ex. | BSD-3-Clause |
| 3D Visual Illusion | `real` | `--dataset visual_illusion_3d --variant real --yes` | 8.10 GB · 455 stereo pairs | Apache-2.0 |
| TransPhy3D | `sample` | `--dataset transphy3d --variant sample` | 0.72 GB · 5 sequences | Apache-2.0 |
| TransPhy3D | `test` | `--dataset transphy3d --variant test --yes` | 4.12 GB · 28 sequences | Apache-2.0 |

`transphy3d/sample` is a strict **subset** of `transphy3d/test`; never combine them.

Reading the acquired data:

```python
from intervene3d.data.external.loaders import get_reader
reader = get_reader("transphy3d", variant="test")
reader.verify(full=True)      # checksums against the manifest
reader.provenance()           # repo, revision, licence -- what a result must cite
for sample in reader: ...
for a, b, T_rel in reader.iter_pairs(stride=5): ...   # (F_t, a) -> F_t+1
```

---

## 6. Experiments

### E1 — Smoke

```bash
$PY scripts/run_experiment.py --config configs/smoke_test.yaml --seed 0
```

4 methods, 15 eval scenes, 44 figure files. ~7 s.

### E2 — Phase 1, problem existence *(the headline experiment)*

```bash
$PY scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed 42
```

9 methods over 78 eval scenes. ~64 s per seed, including the per-seed benchmark
redraw (`data.reseed_with_experiment_seed: true`).

Multi-seed, which is what the reported error bars require:

```bash
for s in 1 2 3; do
  $PY scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed $s
done
$PY scripts/aggregate_results.py --experiment phase1_problem_existence
```

~3.5 min. Seeds may be run concurrently in separate shells — run directories are
immutable and the registry is append-only.

The nine methods, all in one run:

| Method | Kind |
|---|---|
| `single_frame_classifier` | trained softmax, reference view only |
| `passive_multiview_classifier` | trained softmax + one fixed unchosen action |
| `random_intervention` | engine, arbitrary allowed action |
| `max_baseline_intervention` | engine, largest allowed translation |
| `entropy_nbv` | engine, hypothesis-blind next-best-view proxy |
| `intervene3d_no_hypothesis_conditioning` | ablation |
| `intervene3d_no_abstention` | ablation, forced choice |
| `intervene3d_noisy_encoder` | 0.6 px + 2 % depth noise |
| `intervene3d` | the full method |

### E3 — Learned transition ablation (synthetic)

```bash
$PY scripts/run_experiment.py --config <config with transition: hybrid or learned_only> --seed 0
```

The residual training stage runs automatically when a method requests `hybrid` or
`learned_only`, and is tuned by an optional `learned_transition:` block
(`epochs`, `hidden_dim`, `learning_rate`, `batch_size`, `max_actions_per_scene`).

### E4 — Transition-model training on external sequences (GPU)

```bash
$PY scripts/train_transition.py --dataset transphy3d --variant test \
    --max-pairs 2800 --max-points 4096 --stride 5 \
    --epochs 80 --hidden-dim 256 --depth 3 --val-fraction 0.25 --seed 0
```

~4 min to build supervision (CPU, single-threaded) + ~2 min on the GPU.
`--dry-run` builds and reports supervision without training. `--verify`
recomputes dataset checksums first. `--device cpu` forces CPU.

Validation holds out whole **sequences**. Metrics are reported separately on the
geometrically consistent and occlusion-affected subsets; the pooled ratio is
labelled `_POOLED` and is not the result.

### E5 — Real geometry encoder (GPU)

```python
from intervene3d.models.foundation_encoders import MonocularDepthEncoder
enc = MonocularDepthEncoder()          # Depth-Anything-V2-Small, Apache-2.0
inv_depth = enc.predict_inverse_depth(image)   # relative, NOT metric
```

~24 ms/image, 0.35 GB VRAM. The Base and Large checkpoints are **CC-BY-NC-4.0**;
selecting one records the non-commercial restriction in the run manifest. Also
selectable as `model.geometry_encoder.name: depth_anything_v2`, which requires
`Observation.image` and raises rather than falling back to the oracle.

### Config overrides

Any dotted key, repeatable:

```bash
--set model.abstention.tau=0.9
--set model.identifiability.epsilon_px=2.0
--set classifier_epochs=1200
--set data.split=val
--set action_noise.enabled=true --set action_noise.translation_std=0.01
```

---

## 7. Aggregation and figures

| Command | Does |
|---|---|
| `$PY scripts/aggregate_results.py --experiment <name>` | mean / std / Student-t 95 % CI across seeds |
| `$PY scripts/aggregate_results.py --list` | every experiment in the registry |
| `$PY scripts/aggregate_results.py --experiment <name> --config-hash <h>` | pin one configuration |
| `$PY scripts/generate_all_figures.py` | rebuild every figure for every successful run |
| `$PY scripts/generate_all_figures.py --experiment <name> --latest` | newest run only |
| `$PY scripts/generate_all_figures.py --run <dir> --out figures/paper --formats pdf --dpi 600` | publication export |

Aggregation refuses to average across differing config hashes unless
`--all-hashes` is passed. Figures are drawn only from `metrics/figure_data.json`,
so the whole set rebuilds without re-running anything.

### Where output lives

```
experiments/<name>/run_<UTC>_seed<N>_<confighash>/
├── config.yaml  command.txt  git_commit.txt  environment.{txt,json}
├── dataset_manifest.json  run_manifest.json  summary.md
├── metrics/{metrics.json, summary.json, figure_data.json}
├── predictions/predictions.csv
├── tables/method_comparison.{csv,md}
└── figures/*.{pdf,png}
results/<name>/aggregate.{json,csv}
experiments/registry.jsonl       append-only, one line per run
```

---

## 8. Full sequence from a clean checkout

```bash
bash scripts/setup_environment.sh
PY=.venv/bin/python                       # Windows: .venv/Scripts/python.exe

$PY -m pytest tests -q                    # 276 passed
bash scripts/run_smoke_test.sh

$PY scripts/generate_synthetic_data.py --config configs/synthetic/phase1.yaml
$PY scripts/validate_datasets.py --all

for s in 1 2 3; do
  $PY scripts/run_experiment.py --config configs/experiments/phase1_problem_existence.yaml --seed $s
done
$PY scripts/aggregate_results.py --experiment phase1_problem_existence
$PY scripts/generate_all_figures.py --experiment phase1_problem_existence
```

~6 minutes, CPU-only, no network.

---

## 9. Make targets

| Target | Equivalent |
|---|---|
| `make setup` | `bash scripts/setup_environment.sh` |
| `make install` | editable install into the active interpreter |
| `make data` | generate the smoke benchmark |
| `make smoke` | the smoke test |
| `make test` / `make unit` / `make integration` | pytest tiers |
| `make experiment SEED=42` | Phase 1 at one seed |
| `make aggregate` | aggregate Phase 1 |
| `make figures` | regenerate all figures |
| `make validate-datasets` | `validate_datasets.py --all` |
| `make lint` | ruff |
| `make help` | list targets |
