#!/usr/bin/env python3
"""Aggregate multiple runs of one experiment across seeds.

    python scripts/aggregate_results.py --experiment phase1_problem_existence
    python scripts/aggregate_results.py --experiment phase1_problem_existence --config-hash a1b2c3d4

Reports mean, standard deviation and a 95% confidence interval (Student-t, so a
small number of seeds is not reported as if it were Gaussian).  Runs are read
from ``experiments/registry.jsonl``; nothing is recomputed and no run directory
is modified.

By default only runs sharing the newest configuration hash are aggregated:
averaging across *different* configurations would silently mix experiments.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from intervene3d.experiments.registry import (
    load_run_metrics,
    runs_for_experiment,
    summarise_registry,
)
from intervene3d.metrics.aggregate import aggregate_runs, format_pm
from intervene3d.utils.io import dump_json, write_csv
from intervene3d.utils.logging import setup_logging

HEADLINE = [
    ("cea.cea_all", "CEA"),
    ("cea.abstention_rate", "abstain"),
    ("identifiability.identifiability_auroc", "AUROC"),
    ("identifiability.resolvability_accuracy", "resolv.acc"),
    ("fpcr.fpcr", "FPCR"),
    ("contact_depth.abs_rel_contact", "AbsRel"),
    ("contact_depth.rmse_contact", "RMSE"),
    ("mcrb.mae", "MAE_MCRB"),
    ("intervention.normalised_regret", "regret"),
    ("intervention.motion_cost", "motion"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", help="experiment name to aggregate")
    parser.add_argument("--root", default="experiments", help="experiment root directory")
    parser.add_argument("--out", default=None, help="output directory (default: results/<experiment>)")
    parser.add_argument("--config-hash", default=None, help="aggregate only runs with this config hash")
    parser.add_argument("--all-hashes", action="store_true", help="aggregate across differing config hashes (unsafe)")
    parser.add_argument("--list", action="store_true", help="list experiments in the registry and exit")
    args = parser.parse_args(argv)

    setup_logging()
    if args.list or not args.experiment:
        rows = summarise_registry(args.root)
        if not rows:
            print(f"no runs recorded in {args.root}/registry.jsonl")
            return 0
        print(f"{'experiment':36s} {'runs':>5s} {'ok':>4s} {'fail':>5s}  seeds")
        for r in rows:
            print(f"{r['experiment_name']:36s} {r['runs']:5d} {r['success']:4d} {r['failed']:5d}  {r['seeds']}")
        return 0 if args.list else 2

    runs = runs_for_experiment(args.experiment, root=args.root, status="success")
    if not runs:
        print(f"no successful runs for experiment {args.experiment!r} in {args.root}/registry.jsonl", file=sys.stderr)
        return 1

    if not args.all_hashes:
        target = args.config_hash or runs[-1]["config_hash"]
        kept = [r for r in runs if r["config_hash"] == target]
        if len(kept) != len(runs):
            print(f"note: {len(runs) - len(kept)} run(s) with a different config hash were excluded; "
                  f"aggregating config_hash={target}")
        runs = kept

    loaded = [(r, load_run_metrics(r)) for r in runs]
    loaded = [(r, m) for r, m in loaded if m is not None]
    if not loaded:
        print("runs were found but none had a readable metrics file", file=sys.stderr)
        return 1

    seeds = sorted({r["seed"] for r, _ in loaded})
    methods = sorted({name for _, m in loaded for name in m["methods"]})
    print(f"experiment  : {args.experiment}")
    print(f"runs        : {len(loaded)}  seeds: {seeds}")
    print(f"config hash : {loaded[-1][0]['config_hash']}")
    if len(seeds) == 1:
        print("WARNING: only one seed -- no confidence intervals. Re-run with --seed 2 --seed 3 to aggregate.")

    per_method: dict[str, list[dict]] = defaultdict(list)
    for _, m in loaded:
        for name, payload in m["methods"].items():
            per_method[name].append(payload)

    aggregated = {name: aggregate_runs(runs_) for name, runs_ in per_method.items()}

    header = f"{'method':34s}" + "".join(f"{label:>18s}" for _, label in HEADLINE)
    print("\n" + header)
    print("-" * len(header))
    table_rows = []
    for name in methods:
        agg = aggregated.get(name, {})
        cells = [format_pm(agg.get(key, {"mean": float('nan'), "n": 0})) for key, _ in HEADLINE]
        print(f"{name:34s}" + "".join(f"{c:>18s}" for c in cells))
        row = {"method": name, "n_runs": len(per_method.get(name, [])), "seeds": str(seeds)}
        for key, label in HEADLINE:
            s = agg.get(key, {})
            row[f"{label}_mean"] = s.get("mean")
            row[f"{label}_std"] = s.get("std")
            row[f"{label}_ci95"] = s.get("ci95")
        table_rows.append(row)

    out_dir = Path(args.out or (Path("results") / args.experiment))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": args.experiment,
        "n_runs": len(loaded),
        "seeds": seeds,
        "config_hash": loaded[-1][0]["config_hash"],
        "git_commits": sorted({r["git_commit"] for r, _ in loaded}),
        "run_ids": [r["run_id"] for r, _ in loaded],
        "run_paths": [r["run_path"] for r, _ in loaded],
        "aggregate": aggregated,
        "note": (
            "single seed: confidence intervals are undefined" if len(seeds) == 1
            else "mean +/- std with 95% Student-t confidence intervals"
        ),
    }
    dump_json(out_dir / "aggregate.json", payload)
    write_csv(out_dir / "aggregate.csv", table_rows)
    print(f"\nwritten: {out_dir / 'aggregate.json'}\n         {out_dir / 'aggregate.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
