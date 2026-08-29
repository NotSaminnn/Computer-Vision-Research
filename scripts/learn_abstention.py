#!/usr/bin/env python3
"""R1 -- learn when to abstain, and test whether the policy transfers.

    .venv-ml/Scripts/python.exe scripts/learn_abstention.py
    .venv-ml/Scripts/python.exe scripts/learn_abstention.py --model gbdt

The question
------------
Every signal this project has measured on real data is a *single hand-designed
number*: a planarity deviation, a photometric separability, the model's own
confidence. None of them predicts failure well, and the strongest of them is
embarrassing -- ``mask_fraction``, which is simply how much of the frame the
ambiguous region covers, and which needs no model, no action and no theory.

So: does a *combination* of these features do better than the best one alone?
And -- the part that decides whether this is a finding or a curve fit -- does a
policy learned on some depth models predict the failures of a model it has
never seen?

Two protocols, and the second is the one that matters
-----------------------------------------------------
``grouped``
    5-fold cross-validation with folds split by **base scene**. The benchmark
    photographs the same scene from several frames, so a random split would put
    frame 0001 in train and frame 0002 in test and report near-perfect transfer
    that is really memorisation. Grouping by scene is the same leakage control
    used everywhere else in this repository.
``leave-one-model-out``
    Train on every checkpoint but one, test on the held-out one. This asks
    whether failure is predictable *as a property of the image* rather than of a
    particular network's quirks. A policy fitted to Depth-Anything that also
    predicts DepthPro's mistakes has learned something about the scene; one that
    does not has learned something about Depth-Anything.

What this script must report, including when it is unflattering
---------------------------------------------------------------
The headline is not the learned AUROC. It is the learned AUROC **against the
best single feature**, computed on the identical folds. If the combination does
not beat ``mask_fraction`` alone, that is the result and it is printed as such;
a learned model that cannot beat one column does not deserve the complexity, and
finding that out ourselves is worth more than a reviewer finding it for us.

``mask_fraction`` is deliberately *included* as an input rather than excluded to
make the learned model look better. If the policy is mostly reading region size,
the permutation importances will say so.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# Features taken from the stereo protocol run, joined per image with the
# single-view protocol run for the same checkpoint.
STEREO_FEATURES = [
    "delta_de_photometric",   # separability, the quantity the methodology defines
    "residual_direct",        # belief: how well H_D explains the second view
    "residual_emissive",      # belief: how well H_E explains it
    "texture_x",              # how much structure there is to reveal a difference
    "mask_fraction",          # the trivial baseline, included on purpose
    "disparity_gap",          # geometric disagreement before photometric weighting
    "alignment_r2_outside",   # how well the model was calibrated outside the region
]
SINGLE_FEATURES = [
    "delta_de",               # the original planarity proxy
    "confidence",             # the model's own confidence
    "tta_disagreement",       # test-time-augmentation ensemble spread
]


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


def _latest(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(REPO / pattern)))
    return Path(hits[-1]) if hits else None


def load_checkpoint_rows(tag: str) -> list[dict]:
    """Join the stereo and single-view runs for one checkpoint, on image key."""
    stereo = _latest(f"experiments/external_identifiability_stereo_{tag}/run_*/predictions/per_image.csv")
    single = _latest(f"experiments/external_identifiability_{tag}/run_*/predictions/per_image.csv")
    if stereo is None:
        return []
    single_by_key: dict[str, dict] = {}
    if single is not None:
        single_by_key = {r["key"]: r for r in csv.DictReader(single.open())}

    rows: list[dict] = []
    for r in csv.DictReader(stereo.open()):
        if r.get("gt_reliable") != "True":
            continue
        rec = {"model": tag, "key": r["key"], "scene": r.get("scene", ""),
               "fooled": r["fooled"] == "True"}
        ok = True
        for f in STEREO_FEATURES:
            try:
                rec[f] = float(r[f])
            except (KeyError, ValueError):
                ok = False
                break
        if not ok:
            continue
        other = single_by_key.get(r["key"])
        for f in SINGLE_FEATURES:
            try:
                rec[f] = float(other[f]) if other else np.nan
            except (KeyError, ValueError, TypeError):
                rec[f] = np.nan
        rows.append(rec)
    return rows


def build_matrix(rows: list[dict], features: list[str]):
    X = np.array([[r[f] for f in features] for r in rows], dtype=float)
    y = np.array([r["fooled"] for r in rows], dtype=bool)
    groups = np.array([r["scene"] for r in rows])
    models = np.array([r["model"] for r in rows])
    # Median-impute per column; a missing feature must not decide the split.
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0
    return X, y, groups, models


def make_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "gbdt":
        # Shallow and heavily regularised: ~3.5k rows over ~83 scenes is not much
        # data, and an unconstrained booster will memorise scenes.
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["logistic", "gbdt"], default="logistic")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write metrics JSON here")
    args = p.parse_args(argv)

    from sklearn.model_selection import GroupKFold

    tags = ["dav2s", "dav2b", "dav2l", "transparent", "depthpro",
            "da3s", "da3b", "da3l", "moge2", "metric3ds", "metric3dl", "unidepth2"]
    rows: list[dict] = []
    present: list[str] = []
    for t in tags:
        got = load_checkpoint_rows(t)
        if got:
            rows.extend(got)
            present.append(t)
    if not rows:
        print("no data: run the stereo protocol first", file=sys.stderr)
        return 1

    features = STEREO_FEATURES + SINGLE_FEATURES
    X, y, groups, models = build_matrix(rows, features)
    print("=" * 78)
    print("R1  LEARNED ABSTENTION POLICY")
    print("=" * 78)
    print(f"  rows {len(rows)}   checkpoints {len(present)} ({', '.join(present)})")
    print(f"  scenes {len(set(groups))}   base failure rate {100 * y.mean():.1f}%")
    print(f"  features {len(features)}: {', '.join(features)}")

    # ---- single-feature baselines, on the same data -------------------------
    print("\n  single-feature baselines (AUROC, sign chosen to favour the feature):")
    singles: dict[str, float] = {}
    for j, f in enumerate(features):
        a = max(auroc(X[:, j], y), auroc(-X[:, j], y))
        singles[f] = a
        print(f"    {f:24s} {a:.3f}")
    best_feature = max(singles, key=singles.get)
    print(f"  --> best single feature: {best_feature} at {singles[best_feature]:.3f}")

    # ---- protocol 1: grouped cross-validation -------------------------------
    gkf = GroupKFold(n_splits=min(args.folds, len(set(groups))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        m = make_model(args.model)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    grouped_auroc = auroc(oof, y)

    # The baseline must be scored on the SAME folds to be comparable.
    j_best = features.index(best_feature)
    sign = 1.0 if auroc(X[:, j_best], y) >= auroc(-X[:, j_best], y) else -1.0
    grouped_baseline = auroc(sign * X[:, j_best], y)

    print(f"\n  [grouped {gkf.get_n_splits()}-fold, split by base scene]")
    print(f"    learned policy                 {grouped_auroc:.3f}")
    print(f"    best single feature            {grouped_baseline:.3f}  ({best_feature})")
    print(f"    difference                     {grouped_auroc - grouped_baseline:+.3f}")

    # ---- protocol 2: leave-one-model-out ------------------------------------
    print(f"\n  [leave-one-model-out: train on the other {len(present) - 1} checkpoints]")
    print(f"    {'held-out model':16s} {'learned':>8s} {'baseline':>9s} {'diff':>7s}  {'n':>5s}")
    lomo: dict[str, dict] = {}
    for t in present:
        te = models == t
        tr = ~te
        if te.sum() < 20 or len(set(y[tr])) < 2:
            continue
        m = make_model(args.model)
        m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[te])[:, 1]
        a_learn = auroc(pr, y[te])
        a_base = auroc(sign * X[te, j_best], y[te])
        lomo[t] = {"learned": a_learn, "baseline": a_base, "n": int(te.sum()),
                   "fooled_rate": float(y[te].mean())}
        print(f"    {t:16s} {a_learn:8.3f} {a_base:9.3f} {a_learn - a_base:+7.3f}  {int(te.sum()):5d}")
    if lomo:
        ml = float(np.mean([v["learned"] for v in lomo.values()]))
        mb = float(np.mean([v["baseline"] for v in lomo.values()]))
        wins = sum(1 for v in lomo.values() if v["learned"] > v["baseline"])
        print(f"    {'MEAN':16s} {ml:8.3f} {mb:9.3f} {ml - mb:+7.3f}")
        print(f"    learned beats the baseline on {wins}/{len(lomo)} held-out checkpoints")
    else:
        ml = mb = float("nan")
        wins = 0

    # ---- what is the policy actually using? ---------------------------------
    rng = np.random.default_rng(args.seed)
    m_full = make_model(args.model)
    m_full.fit(X, y)
    base_full = auroc(m_full.predict_proba(X)[:, 1], y)
    print("\n  permutation importance (AUROC drop when the column is shuffled):")
    importance: dict[str, float] = {}
    for j, f in enumerate(features):
        drops = []
        for _ in range(8):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base_full - auroc(m_full.predict_proba(Xp)[:, 1], y))
        importance[f] = float(np.mean(drops))
    for f, v in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"    {f:24s} {v:+.4f}")

    verdict = (
        "The learned policy beats the best single feature on the held-out-model protocol."
        if ml > mb else
        "The learned policy does NOT beat the best single feature out of sample. "
        "The combination is not earning its complexity, and this is reported as the result."
    )
    print(f"\n  VERDICT: {verdict}")

    metrics = {
        "experiment": "learned_abstention_policy",
        "model": args.model,
        "n_rows": len(rows),
        "checkpoints": present,
        "n_scenes": len(set(groups)),
        "base_failure_rate": float(y.mean()),
        "features": features,
        "single_feature_auroc": singles,
        "best_single_feature": best_feature,
        "grouped_cv": {"auroc": grouped_auroc, "baseline": grouped_baseline,
                       "folds": gkf.get_n_splits()},
        "leave_one_model_out": {"per_model": lomo, "mean_learned": ml,
                                "mean_baseline": mb, "wins": wins, "n": len(lomo)},
        "permutation_importance": importance,
        "verdict": verdict,
        "protocol_note": (
            "Folds are split by base scene: the benchmark shoots several frames of the "
            "same scene, so a random split would leak. mask_fraction is included as an "
            "input rather than excluded, so that if the policy is mostly reading region "
            "size the permutation importance shows it."
        ),
    }
    out = Path(args.out) if args.out else REPO / "results" / "learned_abstention" / f"metrics_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
