#!/usr/bin/env bash
# Intervene3D -- external dataset acquisition.
#
#   bash scripts/download_datasets.sh --list
#   bash scripts/download_datasets.sh --dataset layereddepth                     # plan only
#   bash scripts/download_datasets.sh --dataset layereddepth --variant train     # < 1 GB, fetches
#   bash scripts/download_datasets.sh --dataset layereddepth --variant benchmark --yes
#   bash scripts/download_datasets.sh --all --dry-run
#
# POLICY (research plan sections 20 and 29):
#   * nothing large is ever downloaded silently -- the exact byte count is listed
#     from the remote host first, and anything over 1 GB needs --yes;
#   * a download is attempted ONLY when both the licence AND the download
#     permission are recorded as `verified` / `true` in
#     configs/datasets/external.yaml;
#   * access restrictions are never bypassed and no credential is ever created;
#   * when manual acquisition is required, exact instructions are printed;
#   * availability is never fabricated -- unconfirmed datasets are reported as
#     ACCESS UNVERIFIED.
#
# Every completed fetch writes manifest.json (per-file SHA-256, resolved commit
# revision, source URL, recorded licence) so the result is verifiable later with
#   python scripts/validate_datasets.py --dataset <key>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  # POSIX venvs put the interpreter in bin/, Windows venvs in Scripts/.
  if   [ -x ".venv/bin/python" ];         then PY=".venv/bin/python"
  elif [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"
  elif command -v python3 >/dev/null 2>&1; then PY="python3"
  else PY="python"
  fi
fi

DATASET=""
VARIANT=""
DRY_RUN=0
ALL=0
LIST=0
YES=0
WORKERS=4

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --variant) VARIANT="${2:-}"; shift 2 ;;
    --workers) WORKERS="${2:-4}"; shift 2 ;;
    --all)     ALL=1; shift ;;
    --list)    LIST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y)  YES=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$LIST" -eq 1 ]; then
  exec "$PY" scripts/validate_datasets.py --list
fi

if [ -z "$DATASET" ] && [ "$ALL" -eq 0 ]; then
  sed -n '2,25p' "$0"
  echo
  echo "Nothing to do. Use --list, --dataset <key> or --all."
  exit 2
fi

export INTERVENE3D_DL_DATASET="$DATASET"
export INTERVENE3D_DL_VARIANT="$VARIANT"
export INTERVENE3D_DL_DRY_RUN="$DRY_RUN"
export INTERVENE3D_DL_ALL="$ALL"
export INTERVENE3D_DL_YES="$YES"
export INTERVENE3D_DL_WORKERS="$WORKERS"

"$PY" - <<'PYDL'
import os, sys
sys.path.insert(0, "src")
from intervene3d.data.external.registry import ExternalRegistry, validate_external_dataset
from intervene3d.data.external.fetchers import (
    CONFIRM_THRESHOLD_GB, FetchFailed, FetchRefused,
    execute_fetch, plan_fetch, resolve_plan, variants_for,
)

registry = ExternalRegistry()
dry_run = os.environ["INTERVENE3D_DL_DRY_RUN"] == "1"
confirmed = os.environ["INTERVENE3D_DL_YES"] == "1"
workers = max(1, int(os.environ["INTERVENE3D_DL_WORKERS"] or 4))
key = os.environ["INTERVENE3D_DL_DATASET"]
variant = os.environ["INTERVENE3D_DL_VARIANT"] or None
do_all = os.environ["INTERVENE3D_DL_ALL"] == "1"

if key and key not in registry:
    print(f"unknown dataset {key!r}; registered: {registry.keys()}", file=sys.stderr)
    raise SystemExit(2)
if variant and do_all:
    print("--variant applies to a single --dataset, not --all", file=sys.stderr)
    raise SystemExit(2)

targets = registry.by_priority() if do_all else [registry[key]]
blocked = fetched = failed = 0

for ds in targets:
    print("=" * 78)
    print(ds.describe())
    report = validate_external_dataset(ds)
    print(f"  local status          : {report['status']}  ({report['root']})")
    known = variants_for(ds)
    if known:
        print("  fetchable variants    : " + ", ".join(
            f"{n} (~{(v.get('approx_size_gb') or 0):.2f} GB)" for n, v in known.items()))

    # ---------------------------------------------------------------- refuse?
    try:
        plan = plan_fetch(ds, variant)
    except FetchRefused as exc:
        blocked += 1
        print("  automated download    : REFUSED")
        print(f"    reason: {exc}")
        print("    Acquire it manually:")
        for line in ds.instructions().splitlines():
            print(f"      {line}")
        print(f"    Then place it under: {report['root']}")
        print(f"    And validate with  : python scripts/validate_datasets.py --dataset {ds.key}")
        print()
        continue

    print("  automated download    : PERMITTED")
    print(plan.describe())

    # ------------------------------------------------- list before you fetch
    try:
        plan = resolve_plan(plan)
    except Exception as exc:
        failed += 1
        print(f"  could not list the remote repository: {type(exc).__name__}: {exc}")
        print()
        continue
    print(f"  remote listing        : {len(plan.files)} files, {plan.total_gb:.3f} GB "
          f"({plan.total_bytes:,} bytes)")

    if dry_run:
        print("  --dry-run: nothing was fetched.")
        print()
        continue
    if plan.total_gb > CONFIRM_THRESHOLD_GB and not confirmed:
        blocked += 1
        print(f"  HELD: {plan.total_gb:.2f} GB exceeds the {CONFIRM_THRESHOLD_GB:.1f} GB confirmation "
              "threshold.")
        print(f"    Re-run with --yes to proceed:")
        print(f"      bash scripts/download_datasets.sh --dataset {ds.key} "
              f"--variant {plan.variant} --yes")
        print()
        continue

    print(f"  downloading to {plan.dest} ...")
    try:
        manifest = execute_fetch(plan, confirmed=True, workers=workers)
    except (FetchRefused, FetchFailed) as exc:
        failed += 1
        print(f"  FAILED: {exc}")
        print()
        continue
    fetched += 1
    print(f"  DONE: {manifest['n_files']} files, {manifest['total_gb']:.3f} GB in "
          f"{manifest['duration_s']:.0f} s")
    print(f"    revision  : {manifest['source']['revision_resolved']}")
    print(f"    manifest  : {plan.dest / 'manifest.json'}")
    print(f"    validate  : python scripts/validate_datasets.py --dataset {ds.key}")
    print()

print("=" * 78)
print(f"{len(targets)} dataset(s) considered; {fetched} fetched, {blocked} not fetched, {failed} failed.")
print("No external dataset is required by the smoke test or the Phase 1 experiment.")
raise SystemExit(1 if failed else 0)
PYDL
