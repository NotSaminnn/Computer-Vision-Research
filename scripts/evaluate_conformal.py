#!/usr/bin/env python3
"""Split-conformal selective prediction: does calibration rescue confidence?

    python scripts/evaluate_conformal.py
    python scripts/evaluate_conformal.py --alpha 0.2 --trials 500

Why this baseline
-----------------
Raw confidence is the weakest form of the Chow's-rule argument, and a reviewer is
right to press for the strongest one. Split conformal prediction is the current
standard for selective prediction with a *finite-sample guarantee*: calibrate a
nonconformity score on held-out data, abstain above the (1-alpha) quantile, and
coverage is provably at least 1-alpha regardless of the score's quality.

The claim under test is deliberately sharp: **calibration cannot repair a score
that ranks badly.** Conformal guarantees how OFTEN you abstain, never WHICH cases
you abstain on. If the underlying signal is anti-correlated with error -- and the
measured confidence AUROC of 0.34 says it is -- then a conformal wrapper around it
inherits that ranking exactly, achieves its stated coverage, and still keeps the
wrong images.

So this script reports two things per score, and they are not the same thing:

``empirical_coverage``
    Does the procedure honour its guarantee? Both scores should pass. A conformal
    method that failed here would simply be broken, and that is not the point.
``selective_risk``
    Among the images it chooses to keep, how often is the model wrong? This is
    what a practitioner actually cares about, and it is where the ranking shows.

Both scores go through the identical procedure, on identical splits, so the only
difference between them is the signal itself.

Reads the per-image CSV from an existing run; no model inference is repeated.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.io import dump_json
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)


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


def split_conformal(score, wrong, alpha, rng, calib_frac=0.5):
    """One split-conformal trial.

    ``score`` is a nonconformity score: larger means *more likely wrong*. The
    threshold is the empirical (1-alpha) quantile on the calibration half, with
    the standard finite-sample correction ceil((n+1)(1-alpha))/n.
    """
    n = len(score)
    idx = rng.permutation(n)
    n_cal = int(round(n * calib_frac))
    cal, test = idx[:n_cal], idx[n_cal:]
    if n_cal < 10 or len(test) < 10:
        return None

    q_level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    threshold = float(np.quantile(score[cal], q_level, method="higher"))

    keep = score[test] <= threshold
    if keep.sum() == 0:
        return None
    return {
        # Guarantee: the retained set should contain at least (1-alpha) of the
        # test points. This is what conformal promises.
        "coverage": float(keep.mean()),
        # What a practitioner cares about: error among what was kept.
        "selective_risk": float(wrong[test][keep].mean()),
        "risk_no_abstention": float(wrong[test].mean()),
        "threshold": threshold,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="an external_identifiability run directory")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5],
                   help="target abstention levels")
    p.add_argument("--trials", type=int, default=400, help="random calibration/test splits")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="external_conformal")
    args = p.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)

    run = Path(args.run) if args.run else sorted(
        Path("experiments/external_identifiability").glob("run_*"))[-1]
    csv_path = run / "predictions" / "per_image.csv"
    if not csv_path.exists():
        LOGGER.error("no per-image predictions at %s", csv_path)
        return 1
    rows = [r for r in csv.DictReader(csv_path.open()) if r["gt_reliable"] == "True"]
    LOGGER.info("source run: %s  (%d images with arbitrating ground truth)", run.name, len(rows))

    wrong = np.array([r["fooled"] == "True" for r in rows])
    # Nonconformity scores: LARGER = more likely wrong, for both.
    scores = {
        "confidence_tta": np.array([float(r["tta_disagreement"]) for r in rows]),
        "identifiability": np.array([-float(r["delta_de"]) for r in rows]),
    }
    for optional in ("ens_std", "ens_range"):
        if optional in rows[0]:
            scores[f"confidence_{optional}"] = np.array([float(r[optional]) for r in rows])

    rng = np.random.default_rng(args.seed)
    results: dict[str, dict] = {}
    for name, sc in scores.items():
        results[name] = {"auroc": auroc(sc, wrong), "by_alpha": {}}
        for alpha in args.alpha:
            trials = [t for t in (split_conformal(sc, wrong, alpha, rng) for _ in range(args.trials)) if t]
            if not trials:
                continue
            results[name]["by_alpha"][f"{alpha:.2f}"] = {
                "target_coverage": 1 - alpha,
                "empirical_coverage_mean": float(np.mean([t["coverage"] for t in trials])),
                "coverage_holds": bool(np.mean([t["coverage"] for t in trials]) >= (1 - alpha) - 0.02),
                "selective_risk_mean": float(np.mean([t["selective_risk"] for t in trials])),
                "selective_risk_std": float(np.std([t["selective_risk"] for t in trials], ddof=1)),
                "risk_no_abstention": float(np.mean([t["risk_no_abstention"] for t in trials])),
                "n_trials": len(trials),
            }

    metrics = {
        "experiment": args.experiment_name,
        "source_run": str(run),
        "n_images": len(rows),
        "base_error_rate": float(wrong.mean()),
        "trials_per_alpha": args.trials,
        "procedure": (
            "Split conformal, 50/50 calibration/test, threshold at the "
            "ceil((n+1)(1-alpha))/n empirical quantile. Identical splits and "
            "procedure for every score; only the signal differs."
        ),
        "scores": results,
        "interpretation": (
            "Coverage is a guarantee about HOW MANY images are kept, not WHICH. A score "
            "that ranks badly still meets coverage while retaining the wrong images, so "
            "selective risk -- not coverage -- is the quantity that separates the methods."
        ),
    }
    config = {
        "experiment": {"name": args.experiment_name, "kind": "conformal_selective", "seed": args.seed},
        "data": {"source_run": str(run), "n_images": len(rows)},
        "model": {"alphas": args.alpha, "trials": args.trials},
    }
    t0 = time.time()
    run_dir = create_run_directory(config, seed=args.seed, root=args.root,
                                   extra={"seed_report": seed_report})
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    print("\n" + "=" * 78)
    print(f"SPLIT-CONFORMAL SELECTIVE PREDICTION: {run_dir.path}")
    print("=" * 78)
    print(f"{len(rows)} images, base error rate {100 * wrong.mean():.1f}%, "
          f"{args.trials} random splits per level\n")
    print(f"  {'score':24s} {'AUROC':>7s}   " + "".join(f"{'a=' + f'{a:.1f}':>10s}" for a in args.alpha))
    print(f"  {'':24s} {'':>7s}   " + "".join(f"{'risk':>10s}" for _ in args.alpha))
    for name, res in results.items():
        cells = "".join(
            f"{100 * res['by_alpha'][f'{a:.2f}']['selective_risk_mean']:9.1f}%"
            if f"{a:.2f}" in res["by_alpha"] else f"{'-':>10s}" for a in args.alpha)
        print(f"  {name:24s} {res['auroc']:7.3f}   {cells}")
    print(f"\n  no abstention: {100 * wrong.mean():.1f}% error")
    print("\n  coverage guarantee honoured (should be True for every score):")
    for name, res in results.items():
        holds = [v["coverage_holds"] for v in res["by_alpha"].values()]
        print(f"    {name:24s} {sum(holds)}/{len(holds)} levels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
