#!/usr/bin/env python3
"""Validate synthetic and external datasets.

    python scripts/validate_datasets.py --all
    python scripts/validate_datasets.py --dataset transphy3d
    python scripts/validate_datasets.py --synthetic data/processed/intervene3d_synth_smoke
    python scripts/validate_datasets.py --list

Synthetic datasets are checked for checksum integrity, the matched-counterfactual
property, split leakage, the presence of genuinely non-identifiable cases, the
reproducibility metadata and array shapes.  External datasets are reported
honestly as PRESENT or NOT DOWNLOADED -- they are never fetched automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from intervene3d.config.loader import repo_root
from intervene3d.data.external.registry import ExternalRegistry, validate_external_dataset
from intervene3d.data.synthetic.validator import validate_dataset
from intervene3d.utils.io import dump_json
from intervene3d.utils.logging import setup_logging


def _discover_synthetic(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p.parent for p in root.glob("*/manifest.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="validate every synthetic and external dataset")
    parser.add_argument("--dataset", help="validate one registered external dataset by key")
    parser.add_argument("--synthetic", help="validate one generated synthetic dataset directory")
    parser.add_argument("--list", action="store_true", help="list the external dataset registry and exit")
    parser.add_argument("--registry", default="configs/datasets/external.yaml")
    parser.add_argument("--no-checksums", action="store_true", help="skip checksum verification (faster)")
    parser.add_argument("--report", help="write the full report to this JSON path")
    args = parser.parse_args(argv)

    setup_logging()
    registry = ExternalRegistry(args.registry)
    results: dict[str, object] = {}
    ok = True

    if args.list:
        print(f"External dataset registry ({registry.path})")
        print(f"verified on: {registry.payload.get('verified_on')}\n")
        for row in registry.summary_rows():
            print(
                f"  [{row['priority']}/5] {row['key']:20s} {row['status']:18s} "
                f"licence={row['licence_status']:18s} auto={'yes' if row['auto_download'] else 'no ':3s} "
                f"local={'present' if row['present_locally'] else 'absent'}"
            )
        print("\nNo external dataset is required by the smoke test or the Phase 1 experiment.")
        return 0

    if args.synthetic or args.all:
        roots = [Path(args.synthetic)] if args.synthetic else _discover_synthetic(repo_root() / "data" / "processed")
        if not roots:
            print("no synthetic datasets found under data/processed/. Generate one with:")
            print("  python scripts/generate_synthetic_data.py --config configs/synthetic/smoke.yaml")
        for root in roots:
            report = validate_dataset(root, verify_checksums=not args.no_checksums)
            print(report.render())
            print()
            results[f"synthetic:{root.name}"] = report.to_dict()
            ok = ok and report.passed

    if args.dataset:
        if args.dataset not in registry:
            print(f"unknown dataset {args.dataset!r}; registered: {registry.keys()}", file=sys.stderr)
            return 2
        ds = registry[args.dataset]
        print(ds.describe())
        report = validate_external_dataset(ds)
        print(f"\n  local status : {report['status']}")
        if not report["present"]:
            print("\nHow to obtain it:\n" + "\n".join("  " + line for line in ds.instructions().splitlines()))
        else:
            print(f"  files        : {report['n_files']}  ({report['size_gb']} GB)")
            print(f"  checksums    : {report['checksums']}")
        results[f"external:{ds.key}"] = report

    elif args.all:
        print("External datasets")
        print("=" * 60)
        for ds in registry.by_priority():
            report = validate_external_dataset(ds)
            mark = "PRESENT" if report["present"] else "NOT DOWNLOADED"
            print(f"  [{ds.payload.get('priority', 0)}/5] {ds.key:20s} {mark:15s} "
                  f"status={ds.status:18s} licence={ds.licence_status}")
            results[f"external:{ds.key}"] = report
        print(
            "\nNone of these are required by the preliminary pipeline. "
            "Run with --dataset <key> for acquisition instructions."
        )

    if not (args.all or args.dataset or args.synthetic):
        parser.print_help()
        return 2

    if args.report:
        dump_json(args.report, results)
        print(f"\nreport written to {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
