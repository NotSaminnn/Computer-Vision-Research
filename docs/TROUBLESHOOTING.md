# Troubleshooting

## Installation

**`ModuleNotFoundError: No module named 'intervene3d'`**
Scripts bootstrap `src/` onto `sys.path` themselves, so `python scripts/...` works
from a clean checkout. If you are importing from your own code, either install the
package (`pip install -e .`) or add `src/` to `sys.path`.

**`python3` is Python 3.9 (macOS system Python)**
The package needs ≥ 3.10. `scripts/setup_environment.sh` creates a `.venv` with a
suitable interpreter; use `.venv/bin/python` afterwards. To point the script at a
specific interpreter: `PYTHON=/path/to/python3.11 bash scripts/setup_environment.sh`.

**SciencePlots is missing, or LaTeX errors while saving a figure**
Figure generation still works: `ieee_style.py` falls back to a self-contained
IEEE-like style. It never enables `text.usetex`, so no LaTeX installation is
required. Check which path was used:
`python -c "import sys;sys.path.insert(0,'src');from intervene3d.visualization import style_provenance;print(style_provenance())"`.

**PyTorch is not installed**
That is expected and fine. The preliminary pipeline, the smoke test and the
Phase 1 experiment are pure NumPy and CPU-only. `environment.txt` will record
`torch: not installed`, which is a correct record, not a warning.

**Windows: `.venv/bin/python: No such file or directory`**
A Windows virtualenv puts the interpreter in `.venv/Scripts/python.exe`. The
scripts and the `Makefile` detect this themselves, so run them unchanged from Git
Bash. If you are typing a command out of the docs by hand, substitute
`.venv/Scripts/python.exe`, or set `PYTHON=` once. `python3` also does not exist
on a stock Windows install — `setup_environment.sh` falls back to `python`.

**Windows: `PytestCacheWarning: could not create cache path ... Access is denied`**
Harmless, and the tests still pass. It happens when the repository directory
denies the rename pytest uses to create `.pytest_cache`. Silence it with
`python -m pytest tests -q -p no:cacheprovider`.

---

## Configuration

**`ConfigError: missing required configuration key ...`**
Validation names the exact key. Compare against `configs/default.yaml` and
`SYNTHETIC_DEFAULTS` / `EXPERIMENT_DEFAULTS` in `config/schema.py`.

**`ConfigError: ... lambda_feature must be 0 (D_feature is NOT IMPLEMENTED)`**
Intentional. `D_feature` needs a real geometry-foundation-model feature space; a
non-zero weight would let a config claim a term the code does not compute.

**`ConfigError: splits.policy must be 'base_scene'`**
Also intentional. Splitting by frame, pose or rendering would separate matched
causal variants of the same scene across splits — the exact leakage this benchmark
must avoid. See `docs/DATASET_MATRIX.md` §4.

**`ConfigError: experiment.name must be set to a descriptive, filesystem-safe name`**
The name becomes a directory. Alphanumerics, `_` and `-` only.

**An override has no effect**
`--set` takes a full dotted path: `--set model.abstention.tau=0.9`, not
`--set tau=0.9`. Values are parsed as YAML, so quote strings that look numeric.

---

## Data generation

**`dataset already exists at ... -- pass --force to regenerate`**
Regeneration is deliberately not automatic. Pass `--force`, or delete the
directory. Regenerating with the same config and seed produces byte-identical
files.

**`contains_non_identifiable_cases` fails validation**
The benchmark must contain cases no allowed action can resolve; a benchmark where
everything resolves would undermine the research question. If this fires, the
scene parameters have drifted. Widen `observer_markers.rig_offset_lateral`
(pushes mirror virtual images out of the aperture), raise
`display.view_tracked_probability`, or narrow `scene.content_depth_spread`
(raises the MCRB beyond the action bounds).

**`matched_counterfactuals` fails validation**
Serious: the causal variants are no longer pixel-identical at `C_0`, so a
single-frame classifier could exploit the difference and the premise collapses.
Almost always a change to `reference_observation` or to the world construction in
`optics/`. `tests/unit/test_optics.py::test_all_mechanisms_are_pixel_identical_at_the_reference_view`
should catch it first.

**`checksums` fails validation**
A file changed after generation. Regenerate with `--force`.

---

## Experiments

**`no scenes in split 'test'`**
The dataset is too small for the split fractions. Increase
`dataset.num_base_scenes` or adjust `splits`.

**`no successful runs for experiment ... in experiments/registry.jsonl`**
Either nothing has run, or every run failed. Check
`experiments/<name>/run_*/run_manifest.json` for `status` and `error`, and
`logs/run.log`.

**`aggregate_results.py` reports fewer runs than expected**
It refuses to average across differing config hashes by default — averaging
different experiments would be meaningless. It prints how many runs it excluded.
Use `--config-hash <hash>` to pick one, or `--all-hashes` if you really mean it.
Note the hash **excludes** `experiment.seed`, so seeds of the same config do share
a hash.

**Every metric shows `± 0.000` across seeds**
Expected under the default seeding protocol: the benchmark is fixed and the
methods are deterministic, so the seed varies nothing that matters. That is a
determinism check, not an error bar. For meaningful confidence intervals set
`data.reseed_with_experiment_seed: true` (the Phase 1 config already does), which
redraws the benchmark per seed.

**FPCR is `0.000` for every method**
Check the `sweep` field in `metrics.json`. With a three-hypothesis family whose
residual ambiguity is two-way, a forced argmax over an unresolvable pair sits at
exactly `p = 0.5`, so any `τ ≥ 0.5` reports zero for the wrong reason. Phase 1
uses `τ = 0.45`.

**`NotImplementedError: geometry encoder 'moge' is NOT IMPLEMENTED`**
Correct behaviour — the adapter refuses rather than silently substituting a
different encoder, so no run can claim a foundation model it did not use. Use
`ground_truth` or `mock`, or complete the Gate 6 checklist in
`docs/DEVELOPMENT_ROADMAP.md`.

**A method abstains on everything**
Expected for `intervene3d_no_hypothesis_conditioning`: with the causal
conditioning removed every pairwise separability is identically zero, so nothing
is ever identifiable. That is the ablation working. If it happens to a *normal*
method, `ε` is probably too large — check `model.identifiability.epsilon_px`.

---

## Datasets

**`automated download : REFUSED`**
By design. A fetch requires **both** a verified licence and
`automated_download_permitted: true` in `configs/datasets/external.yaml`, plus a
`fetch:` block naming the remote repository. Four of the fourteen entries qualify
(2026-08-29); the rest print acquisition instructions. No external dataset is
needed for the smoke test or Phase 1.

**`HELD: N GB exceeds the 1.0 GB confirmation threshold`**
Also by design — nothing large is transferred without an explicit `--yes`. The
message prints the exact command to re-run. Use `--dry-run` first to see the file
list and byte total without transferring.

**`insufficient free space: X GB available, Y GB needed`**
The fetcher refuses below the transfer size + 10 % headroom. Pick a smaller
`--variant` (`bash scripts/download_datasets.sh --dataset <key> --dry-run` lists
them with sizes) or free space.

**`checksum mismatch for <file>`**
The bytes received do not match the publisher's own SHA-256 (Hugging Face
`lfs.oid`), so the file was deleted rather than kept. Re-run; the transfer resumes
rather than restarting. Persistent mismatches mean the upstream file changed —
compare `revision_resolved` in `manifest.json` against the current `main`.

**A download stalls on a single large file**
Some variants are one large tarball. Raise `--workers`: spare workers are spread
across byte ranges of the same file. `--workers 8` gives 8 segments on a
single-file variant.

**`ACCESS UNVERIFIED` next to a dataset**
Means the download mechanism, licence or terms were not confirmed during the
audit. It does **not** mean the dataset is unavailable. Verify it yourself and
update the registry — and check the *machine-readable* source first
(`/api/datasets/<id>` for a Hugging Face entry), which is what the 2026-08-29
re-verification found the 2026-08-28 audit had missed.

**A verified licence but `Auto-download: no`**
Not a contradiction. ClearPose (MIT) and DREDS (CC BY-NC 4.0) are freely usable
but are hosted where no stable, checksummable URL exists, so an automated fetch
could not be reproducible. Permission is about mechanism as much as licence.

---

## Figures

**`no runs found` from `generate_all_figures.py`**
It reads `experiments/registry.jsonl`. Run an experiment first, or point it at a
directory with `--run <path>`.

**A figure looks wrong after changing plotting code**
Regenerate from saved data — no experiment re-run needed:
`python scripts/generate_all_figures.py --run <run_dir>`.

**Figures are slow to save**
PDF font subsetting. Use `--formats png` while iterating.

---

## Smoke test

**It fails at a specific step**
The banner names the step. Steps 1–11 are a single self-contained Python block
you can run in isolation to get the full traceback. Re-run after clearing state:
`rm -rf experiments/smoke_test data/processed/intervene3d_synth_smoke`.

**It prints `SMOKE TEST FAILED` after passing**
Fixed in the current version — an earlier `trap` misread the exit status. If you
see it, you are on an old checkout.
