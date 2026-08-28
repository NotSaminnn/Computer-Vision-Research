#!/usr/bin/env bash
# Intervene3D -- external dataset acquisition.
#
#   bash scripts/download_datasets.sh --list
#   bash scripts/download_datasets.sh --dataset transphy3d
#   bash scripts/download_datasets.sh --dataset layereddepth --dry-run
#   bash scripts/download_datasets.sh --all --dry-run
#
# POLICY (research plan sections 20 and 29):
#   * nothing large is ever downloaded silently;
#   * an automated download is attempted ONLY when both the licence AND the
#     download permission are recorded as `verified` in
#     configs/datasets/external.yaml;
#   * access restrictions are never bypassed;
#   * when manual acquisition is required, exact instructions are printed;
#   * availability is never fabricated -- unconfirmed datasets are reported as
#     ACCESS UNVERIFIED.
#
# As of the 2026-08-28 audit, NO registered dataset satisfies the automated-download
# condition, so this script currently prints instructions for every dataset.
# That is the correct, honest behaviour, not a missing feature.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
fi

DATASET=""
DRY_RUN=0
ALL=0
LIST=0
SAMPLE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --all)     ALL=1; shift ;;
    --list)    LIST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --sample)  SAMPLE=1; shift ;;
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
export INTERVENE3D_DL_DRY_RUN="$DRY_RUN"
export INTERVENE3D_DL_ALL="$ALL"
export INTERVENE3D_DL_SAMPLE="$SAMPLE"

"$PY" - <<'PYDL'
import os, sys
sys.path.insert(0, "src")
from intervene3d.data.external.registry import ExternalRegistry, validate_external_dataset

registry = ExternalRegistry()
dry_run = os.environ["INTERVENE3D_DL_DRY_RUN"] == "1"
sample = os.environ["INTERVENE3D_DL_SAMPLE"] == "1"
key = os.environ["INTERVENE3D_DL_DATASET"]
do_all = os.environ["INTERVENE3D_DL_ALL"] == "1"

if key and key not in registry:
    print(f"unknown dataset {key!r}; registered: {registry.keys()}", file=sys.stderr)
    raise SystemExit(2)

targets = registry.by_priority() if do_all else [registry[key]]
blocked = 0
for ds in targets:
    print("=" * 78)
    print(ds.describe())
    report = validate_external_dataset(ds)
    print(f"  local status          : {report['status']}  ({report['root']})")
    if sample:
        print("  --sample              : the registry records no separate sample split for this dataset; "
              "use the synthetic benchmark for a tiny end-to-end run")
    if report["present"]:
        print("  already present -- nothing to do. Validate with:")
        print(f"    python scripts/validate_datasets.py --dataset {ds.key}")
        print()
        continue
    if not ds.may_auto_download:
        blocked += 1
        print("  automated download    : REFUSED")
        print(f"    reason: licence_status={ds.licence_status!r}, "
              f"automated_download_permitted={ds.payload.get('automated_download_permitted')!r}")
        print("    Automated download requires BOTH to be verified. Acquire it manually:")
        for line in ds.instructions().splitlines():
            print(f"      {line}")
        print(f"    Then place it under: {report['root']}")
        print(f"    And validate with  : python scripts/validate_datasets.py --dataset {ds.key}")
        print()
        continue
    print("  automated download    : PERMITTED")
    if dry_run:
        print("  --dry-run: would download now; nothing was fetched.")
    else:
        print("  NOT IMPLEMENTED: no registered dataset currently satisfies the automated-download")
        print("  condition, so no fetcher has been written. Add one alongside the registry entry that")
        print("  first records a verified licence and a verified download permission.")
    print()

print("=" * 78)
print(f"{len(targets)} dataset(s) considered; {blocked} require manual acquisition.")
print("No data was downloaded. External datasets are not needed for the smoke test or Phase 1.")
PYDL
