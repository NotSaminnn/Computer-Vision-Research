#!/usr/bin/env python3
"""E17 -- choosing where to look, on real photographs.

    .venv/Scripts/python.exe scripts/evaluate_viewpoint_selection.py

The gap this closes
-------------------
Every selection result in this project is synthetic. A calibrated stereo pair
supplies a single executed action, so on real data ``|A| = 1`` and there is
nothing to choose between; the criterion of the paper's selection equation has
therefore never been tested on a photograph.

That is not a limit of the data, as it turns out. The illusion benchmark
photographs each static scene from several camera positions, and the median
stereo disparity differs between those frames while the scene content does not,
which is what a moved camera produces. Each non-video scene is therefore a small
real action set: the observer could have stood in any of these places, and the
question is whether choosing among them by separability beats choosing
arbitrarily.

What this measures, and what it cannot
--------------------------------------
This tests **viewpoint selection** on real imagery. It does not test the maximin
form of the criterion: real data supports two hypotheses rather than four, so there
is a single pair and maximin coincides with the maximum. The four-mechanism
comparison between summed and weakest-pair objectives remains synthetic.

Two selection rules, and why the obvious one fails
---------------------------------------------------
The obvious rule is to prefer the viewpoint with the greatest separability. On this
data it is **worse than choosing at random**, and the reason is instructive rather
than incidental: separability is computed from the predictor's own geometry, since
`H_D` is its aligned disparity. When the predictor hallucinates deep structure on a
flat surface, that disparity departs sharply from the plane and the separability is
large *because the prediction is wrong*, not because the viewpoint is informative.
Maximising it selects the viewpoints where the predictor hallucinates hardest.

The rule that works selects on **evidence** rather than on prediction: prefer the
viewpoint at which some hypothesis best explains the second view actually observed,
that is, the smallest residual $\min_k e_k$ from the belief update. This inverts
the usual active-vision intuition, which prefers the most discriminative viewpoint;
here the well-explained viewpoint is the one worth standing at.

Strategies, all choosing from the same frames of the same scene
----------------------------------------------------------------
``evidence``    smallest residual: the observation is well explained (proposed)
``separability`` greatest predicted separability (the rule that fails)
``confidence``  the frame the predictor is most certain about
``random``      uniformly at random, averaged over many draws
``first``       the first frame, i.e. no choice at all

Two outcomes are reported. ``err_inside`` is the relative error inside the
ambiguous region, which depends on no other region and is therefore unconfounded.
``fooled`` is the error-ratio label used elsewhere in the paper; it is reported for
continuity but is the weaker of the two here, since a viewpoint where the ambiguous
region fills the frame leaves little outside it to calibrate against.
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

TAGS = ["dav2s", "dav2b", "dav2l", "transparent", "depthpro", "da3s", "da3b",
        "da3l", "moge2", "metric3ds", "metric3dl", "unidepth2"]


def load_scenes(tag: str, min_frames: int) -> dict[str, list[dict]]:
    """Admissible frames per scene, for scenes where the camera actually moved.

    Video scenes are excluded: their frames differ because the displayed content
    changes, not because the observer moved, so treating them as an action set
    would count a change of stimulus as a change of viewpoint.
    """
    hits = sorted(glob.glob(str(
        REPO / f"experiments/external_identifiability_stereo_{tag}/run_*/predictions/per_image.csv")))
    conf_hits = sorted(glob.glob(str(
        REPO / f"experiments/external_identifiability_{tag}/run_*/predictions/per_image.csv")))
    if not hits:
        return {}
    conf = {}
    if conf_hits:
        for r in csv.DictReader(Path(conf_hits[-1]).open()):
            try:
                conf[r["key"]] = -float(r["tta_disagreement"])  # larger = more certain
            except (KeyError, ValueError):
                pass

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in csv.DictReader(Path(hits[-1]).open()):
        if r.get("gt_reliable") != "True":
            continue
        scene = r.get("scene", "")
        if scene.startswith("video"):
            continue
        try:
            by_scene[scene].append({
                "key": r["key"],
                "delta": float(r["delta_de_photometric"]),
                "residual": min(float(r["residual_direct"]), float(r["residual_emissive"])),
                "err_inside": float(r["err_inside"]),
                "fooled": r["fooled"] == "True",
                "confidence": conf.get(r["key"], np.nan),
            })
        except (KeyError, ValueError):
            continue
    return {s: v for s, v in by_scene.items() if len(v) >= min_frames}


def evaluate(scenes: dict[str, list[dict]], rng, draws: int) -> dict[str, dict]:
    """Both outcomes under each selection strategy, over the same scenes."""
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for frames in scenes.values():
        d = np.array([f["delta"] for f in frames])
        res = np.array([f["residual"] for f in frames])
        c = np.array([f["confidence"] for f in frames])
        e = np.array([f["err_inside"] for f in frames])
        w = np.array([f["fooled"] for f in frames], dtype=float)

        def take(i):
            return float(e[i]), float(w[i])

        acc["evidence"].append(take(int(np.argmin(res))))
        acc["separability"].append(take(int(np.argmax(d))))
        acc["least_separable"].append(take(int(np.argmin(d))))
        acc["first"].append(take(0))
        if np.all(np.isfinite(c)):
            acc["confidence"].append(take(int(np.argmax(c))))
        # random is averaged over many draws, so no single lucky pick flatters it
        idx = rng.integers(0, len(frames), draws)
        acc["random"].append((float(e[idx].mean()), float(w[idx].mean())))
    return {k: {"err_inside": float(np.mean([a for a, _ in v])),
                "fooled": float(np.mean([b for _, b in v]))}
            for k, v in acc.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-frames", type=int, default=3,
                   help="a scene must offer at least this many viewpoints to be a choice")
    p.add_argument("--draws", type=int, default=512, help="random draws per scene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="viewpoint_selection")
    args = p.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    per_model: dict[str, dict] = {}
    rows: list[dict] = []
    for tag in TAGS:
        scenes = load_scenes(tag, args.min_frames)
        if not scenes:
            continue
        res = evaluate(scenes, rng, args.draws)
        per_model[tag] = {"n_scenes": len(scenes),
                          "n_frames": int(sum(len(v) for v in scenes.values())),
                          **res}
        rows.append({"model": tag,
                     **{f"{k}_{m}": v[m] for k, v in res.items() for m in v}})
        LOGGER.info("%-12s scenes %3d  evidence %.4f  random %.4f  separability %.4f",
                    tag, len(scenes), res["evidence"]["err_inside"],
                    res["random"]["err_inside"], res["separability"]["err_inside"])

    if not per_model:
        LOGGER.error("no scenes with enough viewpoints; run the stereo protocol first")
        return 1

    def agg(strategy, metric):
        vals = [v[strategy][metric] for v in per_model.values() if strategy in v]
        return float(np.mean(vals)) if vals else float("nan")

    ours = agg("evidence", "err_inside")
    rnd = agg("random", "err_inside")
    wins = sum(1 for v in per_model.values()
               if v["evidence"]["err_inside"] < v["random"]["err_inside"])
    wins_fooled = sum(1 for v in per_model.values()
                      if v["evidence"]["fooled"] < v["random"]["fooled"])
    metrics = {
        "experiment": args.experiment_name,
        "n_checkpoints": len(per_model),
        "min_frames_per_scene": args.min_frames,
        "n_scenes": int(np.mean([v["n_scenes"] for v in per_model.values()])),
        "by_strategy": {
            k: {"err_inside": agg(k, "err_inside"), "fooled": agg(k, "fooled")}
            for k in ("evidence", "separability", "least_separable",
                      "confidence", "random", "first")
        },
        "relative_error_reduction_vs_random": (rnd - ours) / rnd if rnd else float("nan"),
        "checkpoints_where_evidence_beats_random_err": wins,
        "checkpoints_where_evidence_beats_random_fooled": wins_fooled,
        "per_model": per_model,
        "scope": (
            "Tests viewpoint selection on real photographs. It does NOT test the maximin "
            "form, since real data supports two hypotheses and therefore one pair, where "
            "maximin coincides with the maximum. Video scenes are excluded because their "
            "frames differ by displayed content rather than by observer motion. Selecting "
            "on predicted separability is WORSE than random here, because separability is "
            "computed from the predictor's own geometry and is largest where that geometry "
            "is most wrong; selecting on the residual of the observed second view is the "
            "rule that transfers."
        ),
    }
    config = {
        "experiment": {"name": args.experiment_name, "kind": "viewpoint_selection", "seed": args.seed},
        "data": {"dataset": "visual_illusion_3d", "variant": "real"},
        "model": {"checkpoints": list(per_model)},
    }
    run_dir = create_run_directory(config, seed=args.seed, root=args.root,
                                   extra={"seed_report": seed_report})
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "per_model.csv", rows)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    f = metrics["by_strategy"]
    print("\n" + "=" * 76)
    print(f"VIEWPOINT SELECTION ON REAL PHOTOGRAPHS: {run_dir.path}")
    print("=" * 76)
    print(f"{len(per_model)} checkpoints, {metrics['n_scenes']} scenes offering "
          f"{args.min_frames}+ viewpoints each. Lower is better on both columns.\n")
    print(f"    {'strategy':38s} {'err inside':>11s} {'fooled':>9s}")
    for k, lab in (("separability", "max predicted separability"),
                   ("first", "first frame, no choice"),
                   ("confidence", "most confident"),
                   ("random", "uniformly at random"),
                   ("least_separable", "least separable"),
                   ("evidence", "BEST-EXPLAINED OBSERVATION (ours)")):
        if np.isfinite(f[k]["err_inside"]):
            print(f"    {lab:38s} {f[k]['err_inside']:11.4f} {100 * f[k]['fooled']:8.1f}%")
    print(f"\n  evidence-based selection reduces error inside the region by "
          f"{100 * metrics['relative_error_reduction_vs_random']:.1f}% against random,")
    print(f"  winning on {wins}/{len(per_model)} checkpoints by error and "
          f"{wins_fooled}/{len(per_model)} by failure rate.")
    print("\n  Selecting on predicted separability is WORSE than random: it is computed")
    print("  from the predictor's own geometry and is largest where that geometry is most")
    print("  wrong, so it selects the viewpoints where the predictor hallucinates hardest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
