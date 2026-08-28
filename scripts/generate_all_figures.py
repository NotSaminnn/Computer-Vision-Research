#!/usr/bin/env python3
"""Regenerate every figure from saved result files.

    python scripts/generate_all_figures.py                       # every successful run
    python scripts/generate_all_figures.py --experiment phase1_problem_existence --latest
    python scripts/generate_all_figures.py --run experiments/smoke_test/run_.../
    python scripts/generate_all_figures.py --out figures/paper --run <run>

No experiment is re-executed and no metric is recomputed: figures are drawn from
each run's ``metrics/figure_data.json``.  That is what makes every figure in the
repository reproducible from files alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from intervene3d.experiments.figures import generate_figures
from intervene3d.experiments.registry import runs_for_experiment, summarise_registry
from intervene3d.utils.io import load_json
from intervene3d.utils.logging import setup_logging


def _run_dirs(args) -> list[Path]:
    if args.run:
        return [Path(args.run)]
    names = [args.experiment] if args.experiment else [r["experiment_name"] for r in summarise_registry(args.root)]
    out: list[Path] = []
    for name in names:
        runs = runs_for_experiment(name, root=args.root, status="success")
        if not runs:
            continue
        out.extend([Path(runs[-1]["run_path"])] if args.latest else [Path(r["run_path"]) for r in runs])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="a single run directory")
    parser.add_argument("--experiment", help="regenerate figures for this experiment only")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--latest", action="store_true", help="only the most recent run per experiment")
    parser.add_argument("--out", default=None, help="output directory (default: <run>/figures)")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args(argv)

    setup_logging()
    runs = _run_dirs(args)
    if not runs:
        print("no runs found. Run an experiment first:\n"
              "  python scripts/run_experiment.py --config configs/smoke_test.yaml", file=sys.stderr)
        return 1

    total = 0
    for run in runs:
        data_path = run / "metrics" / "figure_data.json"
        if not data_path.exists():
            print(f"skipping {run}: no metrics/figure_data.json")
            continue
        out_dir = Path(args.out) / run.name if args.out else run / "figures"
        written = generate_figures(
            load_json(data_path), out_dir, formats=args.formats.split(","), dpi=args.dpi
        )
        total += len(written)
        print(f"{run.name}: {len(written)} files -> {out_dir}")
    print(f"\nregenerated {total} figure files from saved result data.")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
