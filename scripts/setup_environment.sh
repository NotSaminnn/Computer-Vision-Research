#!/usr/bin/env bash
# Intervene3D -- environment setup and verification.
#
#   bash scripts/setup_environment.sh                 # create .venv and install
#   bash scripts/setup_environment.sh --gpu           # also install the optional torch extras
#   bash scripts/setup_environment.sh --check-only    # verify an already-active environment
#
# The preliminary pipeline is CPU-only and needs nothing beyond NumPy,
# Matplotlib, PyYAML and SciencePlots.  PyTorch is OPTIONAL and is required only
# for a real geometry foundation encoder (Gate 6) or a scaled world model
# (Gate 7).  See docs/REPRODUCIBILITY.md for the CPU and GPU paths.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
# POSIX venvs put the interpreter in bin/, Windows venvs (incl. Git Bash / MSYS)
# in Scripts/.  Resolve the layout once, after the venv exists.
venv_python() {
  if [ -x "$1/bin/python" ]; then printf '%s/bin/python' "$1"
  elif [ -x "$1/Scripts/python.exe" ]; then printf '%s/Scripts/python.exe' "$1"
  else printf '%s/bin/python' "$1"; fi
}
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) VENV_BIN="Scripts" ;;
  *)                    VENV_BIN="bin" ;;
esac
WITH_GPU=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --gpu)        WITH_GPU=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    -h|--help)    sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mFAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. environment
if [ "$CHECK_ONLY" -eq 0 ]; then
  log "Creating the Python environment in $VENV_DIR"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV_DIR" >/dev/null
    PY="$(venv_python "$VENV_DIR")"
    log "Installing dependencies with uv"
    uv pip install --python "$PY" -r requirements-dev.txt >/dev/null
    [ "$WITH_GPU" -eq 1 ] && uv pip install --python "$PY" -r requirements-gpu.txt >/dev/null
    uv pip install --python "$PY" -e . >/dev/null
  else
    BASE_PY="${PYTHON:-}"
    if [ -z "$BASE_PY" ]; then
      if command -v python3 >/dev/null 2>&1; then BASE_PY="python3"; else BASE_PY="python"; fi
    fi
    command -v "$BASE_PY" >/dev/null || fail "no Python 3 found; set PYTHON=/path/to/python"
    "$BASE_PY" -m venv "$VENV_DIR"
    PY="$(venv_python "$VENV_DIR")"
    log "Installing dependencies with pip (install 'uv' for a much faster path)"
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install -r requirements-dev.txt >/dev/null
    [ "$WITH_GPU" -eq 1 ] && "$PY" -m pip install -r requirements-gpu.txt >/dev/null
    "$PY" -m pip install -e . >/dev/null
  fi
else
  PY="${PYTHON:-$(venv_python "$VENV_DIR")}"
  [ -x "$PY" ] || PY="${PYTHON:-python3}"
  log "Check-only mode: using $PY"
fi

log "Python: $("$PY" -V 2>&1)"

# ---------------------------------------------------- 2-6. verify the installation
log "Verifying core dependencies, PyTorch/CUDA, rendering and package import"
"$PY" - <<'PYCHECK'
import sys, importlib, platform
ok = True

def need(name, label=None):
    global ok
    try:
        m = importlib.import_module(name)
        print(f"  [ok]   {label or name:16s} {getattr(m, '__version__', '')}")
        return m
    except Exception as exc:
        ok = False
        print(f"  [FAIL] {label or name:16s} {type(exc).__name__}: {exc}")
        return None

print(f"  python           {platform.python_version()} ({platform.system()} {platform.machine()})")
need("numpy"); need("yaml", "pyyaml")

# rendering: Matplotlib must work headless
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1, 1)); ax.plot([0, 1], [0, 1]); plt.close(fig)
    print(f"  [ok]   matplotlib       {matplotlib.__version__} (Agg backend renders)")
except Exception as exc:
    ok = False
    print(f"  [FAIL] matplotlib       {type(exc).__name__}: {exc}")

# IEEE style
try:
    import scienceplots  # noqa: F401
    styles = [s for s in ("science", "ieee", "no-latex") if s in plt.style.available]
    if len(styles) == 3:
        print("  [ok]   scienceplots     science+ieee+no-latex available (no LaTeX needed)")
    else:
        print(f"  [warn] scienceplots     only {styles}; the built-in IEEE fallback will be used")
except Exception:
    print("  [warn] scienceplots     not installed; the built-in IEEE fallback will be used")

# PyTorch / CUDA are OPTIONAL
try:
    import torch
    print(f"  [ok]   torch            {torch.__version__}")
    print(f"         cuda available  : {torch.cuda.is_available()}  (version {torch.version.cuda or 'n/a'})")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"         gpu[{i}]         : {torch.cuda.get_device_name(i)}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        print("         mps available   : True")
except ImportError:
    print("  [ok]   torch            not installed -- OPTIONAL; the preliminary pipeline is CPU/NumPy only")

# the package itself
try:
    sys.path.insert(0, "src")
    import intervene3d
    from intervene3d.config import validate_experiment_config  # noqa: F401
    from intervene3d.optics import get_transition  # noqa: F401
    print(f"  [ok]   intervene3d      {intervene3d.__version__}")
except Exception as exc:
    ok = False
    print(f"  [FAIL] intervene3d      {type(exc).__name__}: {exc}")

sys.exit(0 if ok else 1)
PYCHECK

# ------------------------------------------------------------ 7. small validation
log "Running a small end-to-end validation (analytical optics + identifiability)"
"$PY" - <<'PYVALIDATE'
import sys; sys.path.insert(0, "src")
import numpy as np
from intervene3d.config import validate_synthetic_config
from intervene3d.data.synthetic import (
    action_space_from_config, build_hypothesis_set, generate_base_scene, reference_observation,
)
from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
from intervene3d.models.transition import AnalyticalTransitionModel

cfg = validate_synthetic_config({"dataset": {"num_base_scenes": 1}, "scene": {"num_content_landmarks": 12}})
base = generate_base_scene(np.random.default_rng([0, 0]), cfg, 0)
hyps = build_hypothesis_set(base.content.interface, cfg, np.random.default_rng([0, 0, 7]))
refs = [reference_observation(base.content, h) for h in hyps]
f0 = refs[0].feature
assert all(np.array_equal(r.feature.visible, f0.visible) for r in refs), "variants are not matched at C_0"
dev = max(float(np.nanmax(np.abs(r.feature.uv[f0.visible] - f0.uv[f0.visible]))) for r in refs)
assert dev < 1e-9, f"reference views differ by {dev} px"

est = GeometrySeparabilityEstimator(AnalyticalTransitionModel(),
                                    DistanceWeights.from_dict(cfg["identifiability"]["distance"]))
sep = est.pairwise_over_actions(f0, hyps, action_space_from_config(cfg["action_space"]),
                                markers_cam=base.content.observer_markers_cam)
I = sep.max(axis=0)
print(f"  matched at C_0            : max deviation {dev:.2e} px  [ok]")
print(f"  hypotheses                : {hyps.names}")
print(f"  action-set identifiability: {np.round(I[np.triu_indices(len(hyps), 1)], 2).tolist()}")
print("  [ok]   analytical optics and identifiability behave as expected")
PYVALIDATE

if [ "$VENV_BIN" = "Scripts" ]; then
  ACTIVATE="source $VENV_DIR/Scripts/activate   (PowerShell: .\\$VENV_DIR\\Scripts\\Activate.ps1)"
else
  ACTIVATE="source $VENV_DIR/bin/activate"
fi
cat <<DONE

Environment ready.

  Activate      : $ACTIVATE
  Smoke test    : bash scripts/run_smoke_test.sh
  Tests         : $PY -m pytest tests -q
  Phase 1       : $PY scripts/run_experiment.py \\
                    --config configs/experiments/phase1_problem_existence.yaml --seed 42

CPU-only is fully supported and is the default. Install the GPU extras with
\`bash scripts/setup_environment.sh --gpu\` only when integrating a real geometry
foundation encoder (see docs/DEVELOPMENT_ROADMAP.md, Gate 6).
DONE
