#!/usr/bin/env python3
"""R1b -- nested hyperparameter selection, and does XGBoost beat the rest?

    .venv-ml/Scripts/python.exe scripts/tune_abstention.py
    .venv-ml/Scripts/python.exe scripts/tune_abstention.py --sensitivity

Why nested, and why this is not optional
----------------------------------------
The obvious way to tune is to sweep hyperparameters, take the best grouped-CV
score, and report it. That number is **not** an estimate of generalisation: the
folds used to pick the setting are the same folds used to report it, so the
selection itself leaks. With 68 scenes and a grid of a few dozen settings, the
inflation is easily worth several points of AUROC -- comfortably more than the
+0.099 margin R1 is claiming over the trivial baseline. Reporting a tuned score
that way would manufacture exactly the result we want, which is why it is worth
being careful here rather than after a reviewer asks.

So selection happens on an **inner** GroupKFold inside each **outer** training
fold, and only the outer folds -- never seen during selection -- are scored. The
reported number is therefore the performance of *the whole procedure including
tuning*, which is the thing that would actually be deployed.

The untuned R1 numbers stay in the paper alongside these. If nested tuning does
not beat them, that is worth knowing and is printed rather than buried.

Sensitivity
-----------
``--sensitivity`` sweeps R3's regularisation strength ``C`` over three decades.
That single knob is what stands between "the frozen representation carries the
evidence" and "a 768-dimensional head memorised 68 scenes". If the result is
flat across the sweep it is a property of the representation; if it peaks
sharply at the value we happened to pick, it is a property of the knob.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from learn_abstention import (  # noqa: E402
    SINGLE_FEATURES, STEREO_FEATURES, auroc, build_matrix, load_checkpoint_rows,
)

TAGS = ["dav2s", "dav2b", "dav2l", "transparent", "depthpro",
        "da3s", "da3b", "da3l", "moge2", "metric3ds", "metric3dl", "unidepth2"]


def candidates():
    """(name, factory, grid) for each family. Grids are small on purpose."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = [
        ("logistic", lambda **kw: make_pipeline(StandardScaler(),
                                                LogisticRegression(max_iter=3000, **kw)),
         [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]),
        ("hist_gbdt", lambda **kw: HistGradientBoostingClassifier(random_state=0, **kw),
         [{"max_depth": d, "max_iter": n, "learning_rate": lr,
           "min_samples_leaf": leaf, "l2_regularization": 1.0}
          for d in (2, 3, 4) for n in (150, 300) for lr in (0.03, 0.08) for leaf in (20, 40)]),
    ]
    try:
        from xgboost import XGBClassifier

        out.append((
            "xgboost",
            lambda **kw: XGBClassifier(
                eval_metric="logloss", tree_method="hist", random_state=0,
                n_jobs=4, **kw),
            [{"max_depth": d, "n_estimators": n, "learning_rate": lr,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": mcw, "reg_lambda": 1.0}
             for d in (2, 3, 4) for n in (200, 400) for lr in (0.03, 0.08) for mcw in (5, 20)],
        ))
    except ImportError:
        print("  (xgboost not installed -- skipping that family)", file=sys.stderr)
    return out


def nested_score(X, y, groups, factory, grid, outer_splits=5, inner_splits=4):
    """Outer folds score; inner folds select. Returns (auroc, chosen settings)."""
    from sklearn.model_selection import GroupKFold

    outer = GroupKFold(n_splits=min(outer_splits, len(set(groups.tolist()))))
    oof = np.full(len(y), np.nan)
    chosen = []
    for tr, te in outer.split(X, y, groups):
        gtr = groups[tr]
        inner = GroupKFold(n_splits=min(inner_splits, len(set(gtr.tolist()))))
        best, best_score = None, -np.inf
        for params in grid:
            inner_oof = np.full(len(tr), np.nan)
            for itr, ite in inner.split(X[tr], y[tr], gtr):
                m = factory(**params)
                m.fit(X[tr][itr], y[tr][itr])
                inner_oof[ite] = m.predict_proba(X[tr][ite])[:, 1]
            s = auroc(inner_oof, y[tr])
            if np.isfinite(s) and s > best_score:
                best, best_score = params, s
        m = factory(**best)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        chosen.append(best)
    return auroc(oof, y), chosen


def sensitivity(out_path: Path) -> int:
    """Sweep R3's regularisation over three decades."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cache = REPO / "data" / "interim" / "abstention_features" / "dav2s.npz"
    if not cache.exists():
        print(f"no cached features at {cache}; run learn_abstention_head.py --extract-only",
              file=sys.stderr)
        return 1
    d = np.load(cache, allow_pickle=True)
    X, y, scenes = d["X"], d["y"].astype(bool), d["scenes"]
    gkf = GroupKFold(n_splits=5)

    print("\n  R3 regularisation sensitivity (frozen 768-d features, grouped 5-fold)")
    print(f"    {'C':>8s} {'AUROC':>8s}")
    curve = {}
    for C in (0.001, 0.005, 0.02, 0.1, 0.5, 2.0, 10.0):
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, scenes):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        a = auroc(oof, y)
        curve[str(C)] = a
        print(f"    {C:8.3f} {a:8.3f}")
    vals = np.array(list(curve.values()))
    spread = float(vals.max() - vals.min())
    verdict = ("FLAT across three decades -- the result is a property of the representation, "
               "not of the regularisation constant."
               if spread < 0.05 else
               f"PEAKED (range {spread:.3f}) -- the reported figure depends on the choice of C "
               "and must be reported as a tuned value with this sweep beside it.")
    print(f"\n    range over the sweep: {spread:.3f}\n    {verdict}")
    out_path.write_text(json.dumps(
        {"experiment": "r3_regularisation_sensitivity", "curve": curve,
         "range": spread, "verdict": verdict}, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sensitivity", action="store_true", help="run the R3 C-sweep as well")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=4)
    args = p.parse_args(argv)

    outdir = REPO / "results" / "learned_abstention"
    outdir.mkdir(parents=True, exist_ok=True)

    rows, present = [], []
    for t in TAGS:
        got = load_checkpoint_rows(t)
        if got:
            rows.extend(got)
            present.append(t)
    if not rows:
        print("no data", file=sys.stderr)
        return 1
    features = STEREO_FEATURES + SINGLE_FEATURES
    X, y, groups, models = build_matrix(rows, features)

    print("=" * 78)
    print("R1b  NESTED HYPERPARAMETER SELECTION")
    print("=" * 78)
    print(f"  rows {len(rows)}   checkpoints {len(present)}   scenes {len(set(groups.tolist()))}")
    print(f"  base failure rate {100 * y.mean():.1f}%")

    j = features.index("mask_fraction")
    sign = 1.0 if auroc(X[:, j], y) >= auroc(-X[:, j], y) else -1.0
    base = auroc(sign * X[:, j], y)
    print(f"  trivial baseline (mask_fraction): {base:.3f}\n")

    print(f"  nested CV: outer {args.outer} folds score, inner {args.inner} folds select")
    print(f"    {'family':12s} {'grid':>5s} {'nested AUROC':>13s} {'vs baseline':>12s}")
    results = {}
    for name, factory, grid in candidates():
        score, chosen = nested_score(X, y, groups, factory, grid, args.outer, args.inner)
        results[name] = {"nested_auroc": score, "grid_size": len(grid),
                         "selected_per_fold": chosen}
        print(f"    {name:12s} {len(grid):5d} {score:13.3f} {score - base:+12.3f}")

    best = max(results, key=lambda k: results[k]["nested_auroc"])
    print(f"\n  best family: {best} at {results[best]['nested_auroc']:.3f}")

    untuned = outdir / "metrics_gbdt.json"
    if untuned.exists():
        prev = json.loads(untuned.read_text())["grouped_cv"]["auroc"]
        delta = results[best]["nested_auroc"] - prev
        print(f"  untuned R1 (HistGBDT, fixed settings, same protocol): {prev:.3f}")
        print(f"  nested tuning changes it by {delta:+.3f}")
        if delta <= 0:
            print("  --> tuning does NOT help. The untuned settings stand, and the honest")
            print("      reading is that the grid was not the limiting factor.")

    payload = {
        "experiment": "nested_hyperparameter_selection",
        "n_rows": len(rows), "checkpoints": present,
        "n_scenes": int(len(set(groups.tolist()))),
        "baseline_mask_fraction": base,
        "families": {k: {"nested_auroc": v["nested_auroc"], "grid_size": v["grid_size"],
                         "selected_per_fold": v["selected_per_fold"]}
                     for k, v in results.items()},
        "best_family": best,
        "protocol_note": (
            "Nested: settings are chosen on inner GroupKFold folds within each outer "
            "training fold, and scored only on outer folds never seen during selection. "
            "A single-level sweep reporting its own best score would be optimistically "
            "biased by more than the margin being claimed."
        ),
    }
    (outdir / "nested_tuning.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  written: {outdir / 'nested_tuning.json'}")

    if args.sensitivity:
        return sensitivity(outdir / "r3_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
