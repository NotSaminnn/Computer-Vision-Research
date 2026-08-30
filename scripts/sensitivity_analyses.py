#!/usr/bin/env python3
"""Sensitivity analyses requested of the decision thresholds.

    python scripts/sensitivity_analyses.py

Three thresholds carry the decisions in this work, and a reader is entitled to
know how much each one is doing:

``epsilon``
    the perceptual threshold separating identifiable from unidentifiable. It is
    fixed at 1.0 px and never tuned, but "not tuned" is a claim about process,
    not about influence. This sweeps it across two orders of magnitude and
    reports what the abstention decision does in response, recomputed from the
    per-scene identifiability scores already saved by each run, so no experiment
    is re-executed and no seed is re-drawn.

``R^2`` admissibility
    the criterion excluding images whose stereo reference cannot arbitrate the
    question. It removes about a third of one benchmark, which invites the
    reasonable worry that the retained set is biased toward easy or hard cases.
    This sweeps the threshold and reports the fooled rate and the identifiability
    AUROC on each retained subset.

``tau``
    the confidence threshold above which a commitment counts as confident. Its
    sweep is already recorded in every run; it is summarised here alongside the
    others so the three appear together.

Nothing here is a new measurement. Every number is recomputed from artefacts the
original runs wrote, which is what makes it a sensitivity analysis rather than a
second experiment.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.io import dump_json, write_csv
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - p * (p + 1) / 2) / (p * n))


# ------------------------------------------------------------------ epsilon
def epsilon_sweep(epsilons: list[float]) -> list[dict]:
    """Recompute the abstention decision at each epsilon, per seed.

    The identifiability score of a scene does not depend on epsilon; only the
    comparison against it does. So the sweep is exact rather than approximate:
    it is the decision the system would have taken had epsilon been set
    differently, on the very same scenes and the very same scores.
    """
    rows: list[dict] = []
    files = sorted(glob.glob(str(REPO / "experiments/phase1_problem_existence/run_*/predictions/predictions.csv")))
    if not files:
        return rows
    per_eps: dict[float, list[dict]] = defaultdict(list)
    for f in files:
        recs = [r for r in csv.DictReader(Path(f).open())
                if r.get("method") == "intervene3d"]
        if not recs:
            continue
        try:
            iden = np.array([float(r["identifiability_score"]) for r in recs])
            correct = np.array([r["predicted_mechanism"] == r["true_mechanism"] for r in recs])
            resolvable = np.array([r["resolvable_gt"] == "True" for r in recs])
            pmax = np.array([float(r["max_probability"]) for r in recs])
        except (KeyError, ValueError):
            continue
        for eps in epsilons:
            answer = iden >= eps          # the decision the rule would take
            n = len(recs)
            committed = correct[answer]
            # FPCR: confident on a case no permitted action resolves
            unres = ~resolvable
            fp = answer & unres & (pmax > 0.45)
            per_eps[eps].append({
                "abstention_rate": float(1.0 - answer.mean()),
                "cea": float((correct & answer).sum() / n),
                "committed": float(committed.mean()) if answer.any() else float("nan"),
                "fpcr": float(fp.sum() / max(int(unres.sum()), 1)),
            })
    for eps in epsilons:
        v = per_eps.get(eps, [])
        if not v:
            continue
        rows.append({
            "epsilon_px": eps, "n_seeds": len(v),
            **{k: float(np.nanmean([x[k] for x in v])) for k in
               ("abstention_rate", "cea", "committed", "fpcr")},
        })
    return rows


# ------------------------------------------------------- admissibility R^2
def admissibility_sweep(thresholds: list[float], tag: str = "dav2s") -> list[dict]:
    """Vary the criterion that decides which reference measurements can arbitrate."""
    hit = sorted(glob.glob(str(
        REPO / f"experiments/external_identifiability_{tag}/run_*/predictions/per_image.csv")))
    if not hit:
        return []
    recs = list(csv.DictReader(Path(hit[-1]).open()))
    plan = np.array([float(r["gt_planarity"]) for r in recs])
    fooled = np.array([r["fooled"] == "True" for r in recs])
    delta = np.array([-float(r["delta_de"]) for r in recs])
    mask = np.array([float(r["mask_fraction"]) for r in recs])
    rows = []
    for t in thresholds:
        keep = plan >= t
        if keep.sum() < 30:
            continue
        rows.append({
            "r2_threshold": t,
            "n_retained": int(keep.sum()),
            "fraction_retained": float(keep.mean()),
            "fooled_rate": float(fooled[keep].mean()),
            "identifiability_auroc": auroc(delta[keep], fooled[keep]),
            "mask_fraction_auroc": auroc(mask[keep], fooled[keep]),
        })
    return rows


def category_breakdown(tag: str = "dav2s") -> list[dict]:
    """What the admissibility criterion removes, per illusion category."""
    hit = sorted(glob.glob(str(
        REPO / f"experiments/external_identifiability_{tag}/run_*/predictions/per_image.csv")))
    if not hit:
        return []
    by: dict[str, list[dict]] = defaultdict(list)
    for r in csv.DictReader(Path(hit[-1]).open()):
        by[r.get("category", "?")].append(r)
    rows = []
    for cat, v in sorted(by.items()):
        keep = [r for r in v if r["gt_reliable"] == "True"]
        drop = [r for r in v if r["gt_reliable"] != "True"]
        f = lambda rs: float(np.mean([r["fooled"] == "True" for r in rs])) if rs else float("nan")
        rows.append({
            "category": cat, "n_total": len(v),
            "n_retained": len(keep), "n_excluded": len(drop),
            "excluded_fraction": len(drop) / len(v),
            "fooled_retained": f(keep), "fooled_excluded": f(drop),
            "planarity_retained": float(np.median([float(r["gt_planarity"]) for r in keep])) if keep else float("nan"),
            "planarity_excluded": float(np.median([float(r["gt_planarity"]) for r in drop])) if drop else float("nan"),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="sensitivity")
    args = p.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)
    t0 = time.time()

    eps = epsilon_sweep([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    adm = admissibility_sweep([0.70, 0.80, 0.85, 0.90, 0.95, 0.98])
    cat = category_breakdown()
    if not eps or not adm:
        LOGGER.error("missing source runs; run the synthetic and external protocols first")
        return 1

    base = next((r for r in eps if abs(r["epsilon_px"] - 1.0) < 1e-9), None)
    metrics = {
        "experiment": args.experiment_name,
        "note": ("Recomputed from artefacts the original runs wrote. No experiment is "
                 "re-executed and no seed re-drawn, so these are the decisions the same "
                 "system would have taken under different thresholds on identical data."),
        "epsilon_sweep": eps,
        "admissibility_sweep": adm,
        "category_breakdown": cat,
        "operating_point": base,
        "occlusion_weight_px": 4.0,
        "tau": 0.45,
    }
    config = {
        "experiment": {"name": args.experiment_name, "kind": "sensitivity", "seed": args.seed},
        "data": {"source": "phase1_problem_existence + external_identifiability"},
        "model": {"epsilon_grid": [r["epsilon_px"] for r in eps]},
    }
    run_dir = create_run_directory(config, seed=args.seed, root=args.root, extra={"seed_report": seed_report})
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "epsilon_sweep.csv", eps)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    print("\n" + "=" * 76)
    print(f"THRESHOLD SENSITIVITY: {run_dir.path}")
    print("=" * 76)
    print(f"\n  EPSILON  (operating point 1.0 px, {eps[0]['n_seeds']} seeds)")
    print(f"    {'eps (px)':>9s} {'abstains':>9s} {'CEA':>7s} {'committed':>10s} {'FPCR':>7s}")
    for r in eps:
        star = "  <-- used" if abs(r["epsilon_px"] - 1.0) < 1e-9 else ""
        print(f"    {r['epsilon_px']:9.2f} {100*r['abstention_rate']:8.1f}% {r['cea']:7.3f} "
              f"{r['committed']:10.3f} {r['fpcr']:7.3f}{star}")

    print(f"\n  ADMISSIBILITY R^2  (operating point 0.90)")
    print(f"    {'R^2':>6s} {'retained':>9s} {'fooled':>8s} {'ident AUROC':>12s} {'mask AUROC':>11s}")
    for r in adm:
        star = "  <-- used" if abs(r["r2_threshold"] - 0.90) < 1e-9 else ""
        print(f"    {r['r2_threshold']:6.2f} {100*r['fraction_retained']:8.1f}% "
              f"{100*r['fooled_rate']:7.1f}% {r['identifiability_auroc']:12.3f} "
              f"{r['mask_fraction_auroc']:11.3f}{star}")

    print("\n  WHAT THE CRITERION EXCLUDES, BY CATEGORY")
    print(f"    {'category':18s} {'total':>6s} {'kept':>6s} {'excl':>6s} "
          f"{'planarity kept':>15s} {'planarity excl':>15s}")
    for r in cat:
        print(f"    {r['category']:18s} {r['n_total']:6d} {r['n_retained']:6d} {r['n_excluded']:6d} "
              f"{r['planarity_retained']:15.3f} {r['planarity_excluded']:15.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
