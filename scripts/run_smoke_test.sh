#!/usr/bin/env bash
# Intervene3D -- MANDATORY end-to-end smoke test.
#
#   bash scripts/run_smoke_test.sh
#
# Tiny, CPU-only, no external data, no network.  Exits non-zero if ANY step fails.
#
# Steps (research plan section 18):
#    1 import the package                    10 update belief
#    2 validate configuration                11 calculate several metrics
#    3 create a synthetic scene              12 save a run directory
#    4 at least two competing hypotheses     13 generate at least one figure
#    5 generate candidate interventions      14 verify the figures exist
#    6 produce predictions                   15 verify the metrics exist
#    7 calculate separability                16 verify the run manifest exists
#    8 select an intervention                17 exit non-zero on any failure
#    9 simulate/observe the next view
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi

SYNTH_CONFIG="configs/synthetic/smoke.yaml"
EXP_CONFIG="configs/smoke_test.yaml"
SEED=0
STEP=0
SUCCEEDED=0

banner() { printf '\n\033[1;34m[%02d] %s\033[0m\n' "$1" "$2"; }
ok()     { printf '     \033[0;32mPASS\033[0m %s\n' "$*"; }
die()    { printf '     \033[1;31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

# Any exit before the final line is a failure, including an unexpected one.
trap '[ "$SUCCEEDED" -eq 1 ] || printf "\n\033[1;31mSMOKE TEST FAILED\033[0m at step %s\n" "$STEP" >&2' EXIT

printf '\033[1m================ Intervene3D smoke test ================\033[0m\n'
printf 'python : %s\n' "$("$PY" -V 2>&1)"
printf 'repo   : %s\n' "$REPO_ROOT"

# ---------------------------------------------------------------- steps 1-11
STEP=1
banner 1 "Steps 1-11: package, config, scene, hypotheses, actions, predictions, separability, selection, observation, belief, metrics"
"$PY" - "$SYNTH_CONFIG" <<'PYCORE' || die "core pipeline check failed"
import sys
sys.path.insert(0, "src")
import numpy as np

# 1. import the package
import intervene3d
from intervene3d.config import ConfigError, load_config, validate_synthetic_config
print(f"     intervene3d {intervene3d.__version__} imported")

# 2. validate configuration (and prove that validation actually rejects bad input)
cfg = validate_synthetic_config(load_config(sys.argv[1]))
print(f"     config validated: {cfg['dataset']['name']}, mechanisms={cfg['mechanisms']}")
try:
    validate_synthetic_config({"splits": {"train": 0.9, "val": 0.2, "test": 0.3, "policy": "base_scene"}})
    raise AssertionError("validation accepted a config whose splits do not sum to 1")
except ConfigError:
    print("     config validation correctly rejects an invalid config")

# 3. create a synthetic scene
from intervene3d.data.synthetic import (
    action_space_from_config, build_hypothesis_set, generate_base_scene, reference_observation, simulate,
)
base = generate_base_scene(np.random.default_rng([0, 0]), cfg, 0)
reference = reference_observation(base.content, None) if False else None
print(f"     scene created: {base.content.n_content} content landmarks, "
      f"{base.content.n_markers} observer markers, interface at {base.interface_distance:.2f} m")

# 4. at least two competing hypotheses
hyps = build_hypothesis_set(base.content.interface, cfg, np.random.default_rng([0, 0, 7]))
assert len(hyps) >= 2, "need at least two competing hypotheses"
print(f"     hypotheses: {[h.symbol for h in hyps]} ({[h.label for h in hyps]})")

# matched-counterfactual property
refs = [reference_observation(base.content, h) for h in hyps]
f0 = refs[0].feature
for r in refs[1:]:
    assert np.array_equal(r.feature.visible, f0.visible), "variants differ in visibility at C_0"
dev = max(float(np.nanmax(np.abs(r.feature.uv[f0.visible] - f0.uv[f0.visible]))) for r in refs)
assert dev < 1e-9, f"variants are not matched at C_0 (max deviation {dev} px)"
print(f"     matched at C_0: max deviation {dev:.2e} px across all variants")

# 5. candidate interventions
actions = action_space_from_config(cfg["action_space"])
assert len(actions) >= 2, "need candidate interventions"
print(f"     action set |A| = {len(actions)}, bounds: "
      f"<= {actions.config.max_translation} m / {actions.config.max_rotation_deg} deg")

# 6-7. predictions and separability
from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
from intervene3d.models.transition import AnalyticalTransitionModel
weights = DistanceWeights.from_dict(cfg["identifiability"]["distance"])
est = GeometrySeparabilityEstimator(AnalyticalTransitionModel(), weights)
markers = base.content.observer_markers_cam
sep = est.pairwise_over_actions(f0, hyps, actions, markers_cam=markers)
assert sep.shape == (len(actions), len(hyps), len(hyps))
print(f"     predicted {len(actions) * len(hyps)} consequences; separability tensor {sep.shape}")

# 8. select an intervention
from intervene3d.inference.engine import AbstentionPolicy, Intervene3DEngine
from intervene3d.models.belief import LikelihoodBeliefUpdater
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.selector import MaxSeparabilitySelector
eps = float(cfg["identifiability"]["epsilon_px"])
engine = Intervene3DEngine(
    estimator=est, selector=MaxSeparabilitySelector(est),
    belief=LikelihoodBeliefUpdater(beta=1.0),
    identifiability=EpsilonIdentifiabilityEstimator(epsilon=eps),
    abstention=AbstentionPolicy(enabled=True, tau=0.8),
)

# 9-10. observe the next view and update belief, for every true mechanism
from intervene3d.metrics.classification import causal_explanation_accuracy, false_physical_certainty_rate
from intervene3d.metrics.depth import contact_depth_metrics
from intervene3d.metrics.identifiability import auroc
from intervene3d.data.types import CHANNEL_CONTENT

predicted, truth, abstained, scores, labels, maxp = [], [], [], [], [], []
for t, h in enumerate(hyps):
    ref = reference_observation(base.content, h)
    def observe(action, index, _h=h):
        return simulate(base.content, _h, action, reference_feature=ref.feature)
    result = engine.run(f"smoke_{h.symbol}", ref.feature, hyps, actions, observe, markers_cam=markers)
    assert result.belief_trajectory.shape[0] >= 2, "belief was not updated"
    assert abs(float(result.hypothesis_probabilities.sum()) - 1.0) < 1e-9, "posterior is not normalised"
    predicted.append(result.predicted_mechanism)
    truth.append(h.mechanism.value)
    abstained.append(result.abstained)
    scores.append(result.identifiability_score)
    labels.append(result.resolvable)
    maxp.append(result.max_probability)
    mask = (ref.feature.channel == CHANNEL_CONTENT) & ref.feature.visible
    depth = contact_depth_metrics(result.contact_geometry, ref.contact_depth, mask)
    print(f"     {h.symbol}: action={result.selected_action:>22s}  "
          f"p_max={result.max_probability:.3f}  I_A={result.identifiability_score:6.2f}  "
          f"{'ABSTAIN' if result.abstained else 'resolved -> ' + result.predicted_mechanism}  "
          f"AbsRel_contact={depth['abs_rel_contact']:.4f}")

# 11. several metrics
cea = causal_explanation_accuracy(predicted, truth, abstained=abstained)
fpcr = false_physical_certainty_rate(maxp, labels, tau=0.8, abstained=abstained)
print(f"     metrics: CEA(all)={cea['cea_all']:.3f}  CEA(committed)={cea['cea_committed']:.3f}  "
      f"abstention={cea['abstention_rate']:.3f}  FPCR={fpcr['fpcr']}")
print("     steps 1-11 OK")
PYCORE
ok "package, configuration, scene, hypotheses, actions, predictions, separability, selection, observation, belief update and metrics"

# ------------------------------------------------------- step: synthetic dataset
STEP=2
banner 2 "Generating the tiny synthetic benchmark"
"$PY" scripts/generate_synthetic_data.py --config "$SYNTH_CONFIG" || die "synthetic data generation failed"
ok "synthetic benchmark generated and validated"

# ----------------------------------------------------- steps 12-13: full run
STEP=3
banner 3 "Steps 12-13: full experiment run (unique run directory + figures)"
"$PY" scripts/run_experiment.py --config "$EXP_CONFIG" --seed "$SEED" || die "experiment run failed"
ok "experiment completed"

# ------------------------------------------------- steps 14-16: verify artefacts
STEP=4
banner 4 "Steps 14-16: verifying figures, metrics and the run manifest"
"$PY" - <<'PYVERIFY' || die "artefact verification failed"
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from intervene3d.reproducibility.manifest import runs_for_experiment

runs = runs_for_experiment("smoke_test", root="experiments", status="success")
assert runs, "no successful smoke_test run in experiments/registry.jsonl"
run = Path(runs[-1]["run_path"])
print(f"     run directory: {run}")

# 16. run manifest
manifest_path = run / "run_manifest.json"
assert manifest_path.exists(), f"missing {manifest_path}"
manifest = json.loads(manifest_path.read_text())
assert manifest["status"] == "success", f"run status is {manifest['status']}"
for key in ("seed", "config_hash", "git_commit", "created_utc", "command", "reproduction_command"):
    assert manifest.get(key) is not None, f"run manifest is missing {key!r}"
print(f"     run manifest OK (seed={manifest['seed']}, config_hash={manifest['config_hash']}, "
      f"git={str(manifest['git_commit'])[:10]})")

# reproducibility side-cars
for name in ("config.yaml", "command.txt", "git_commit.txt", "environment.txt",
             "dataset_manifest.json", "summary.md"):
    assert (run / name).exists() and (run / name).stat().st_size > 0, f"missing or empty {name}"
print("     config / command / git / environment / dataset manifest / summary all present")

# 15. metrics
metrics_path = run / "metrics" / "metrics.json"
assert metrics_path.exists(), f"missing {metrics_path}"
metrics = json.loads(metrics_path.read_text())
assert metrics["methods"], "metrics contain no methods"
for name, m in metrics["methods"].items():
    for key in ("cea", "fpcr", "identifiability", "contact_depth", "mcrb"):
        assert key in m, f"method {name} is missing metric group {key!r}"
print(f"     metrics OK ({len(metrics['methods'])} methods, {metrics['n_eval_scenes']} eval scenes)")
assert (run / "metrics" / "figure_data.json").exists(), "missing figure_data.json"
assert (run / "predictions" / "predictions.csv").exists(), "missing predictions.csv"
assert (run / "tables" / "method_comparison.csv").exists(), "missing method_comparison.csv"

# 14. figures
pdfs = sorted((run / "figures").glob("*.pdf"))
pngs = sorted((run / "figures").glob("*.png"))
assert pdfs, "no PDF figures were generated"
for f in pdfs + pngs:
    assert f.stat().st_size > 500, f"figure {f.name} is suspiciously small ({f.stat().st_size} bytes)"
print(f"     figures OK ({len(pdfs)} PDF + {len(pngs)} PNG, all non-empty)")

print(f"     RUN_DIR={run}")
PYVERIFY
ok "figures, metrics, predictions, tables and run manifest all verified"

# ------------------------------------------- step: figures regenerate from files
STEP=5
banner 5 "Regenerating figures from saved result files only"
"$PY" scripts/generate_all_figures.py --experiment smoke_test --latest >/dev/null || die "figure regeneration failed"
ok "every figure is reproducible from metrics/figure_data.json"

SUCCEEDED=1
printf '\n\033[1;32m================ SMOKE TEST PASSED ================\033[0m\n'
printf 'Inspect the run with:\n'
printf '  cat  $(ls -dt experiments/smoke_test/run_* | head -1)/summary.md\n'
printf '  open $(ls -dt experiments/smoke_test/run_* | head -1)/figures/\n\n'
exit 0
