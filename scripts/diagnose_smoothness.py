#!/usr/bin/env python3
"""Does prediction smoothness explain why identifiability degrades on larger models?

    python scripts/diagnose_smoothness.py

The claim under test
--------------------
The sweep found the identifiability AUROC falling monotonically across the
Depth-Anything-V2 family -- 0.632 (Small) -> 0.578 (Base) -> 0.448 (Large) --
while the fooled rate stayed flat at 65-67%. The explanation offered was a
hypothesis, not a measurement: ``delta_DE`` is a *planarity deviation* read off
the model's own prediction, so if a larger model simply predicts more smoothly
there is less deviation to measure and the signal loses its grip.

That is testable, and it could be wrong. If smoothness does NOT track the AUROC,
the degradation is not an artefact of this particular ``Delta`` and the problem
lies deeper -- in the formulation rather than the instantiation. Either answer is
worth having; only one of them is comfortable.

What is measured, per image, inside the illusion mask
-----------------------------------------------------
``grad_mag``
    Median magnitude of the spatial gradient of the prediction, after
    normalising each map to unit interquartile range so that models with
    different output scales are comparable at all. Without that normalisation
    this measures output units, not smoothness.
``roughness``
    RMS of the discrete Laplacian, same normalisation. Second-order, so it is
    insensitive to a smooth global ramp -- a model may be steep and still smooth.
``residual_rms``
    RMS residual from the least-squares plane. This is the quantity ``delta_DE``
    is built from, reported directly so the chain from smoothness to signal is
    visible rather than assumed.

Reports Spearman correlation against the per-model AUROC. With four checkpoints
this is directional evidence and is labelled as such: n=4 supports a statement
about consistency of sign, never a claim of significance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from intervene3d.data.external.loaders import get_reader
from intervene3d.models.foundation_encoders import CHECKPOINTS, MonocularDepthEncoder
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.io import dump_json, write_csv
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)

SWEEP = {
    "dav2s": "depth-anything/Depth-Anything-V2-Small-hf",
    "dav2b": "depth-anything/Depth-Anything-V2-Base-hf",
    "dav2l": "depth-anything/Depth-Anything-V2-Large-hf",
    "transparent": "depth-anything/prompt-depth-anything-vits-transparent-hf",
}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x), dtype=float)
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def scale_free(pred: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Normalise to unit IQR so smoothness is comparable across output scales.

    Every model here predicts up to an unknown scale, so an un-normalised
    gradient would rank models by their output units rather than by how smooth
    they are. IQR rather than std because the maps have heavy tails.
    """
    values = pred[sel]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    return pred / iqr if iqr > 1e-9 else pred


def measure(pred: np.ndarray, inside: np.ndarray) -> dict | None:
    if inside.sum() < 500:
        return None
    p = scale_free(pred, inside)
    gy, gx = np.gradient(p)
    grad = np.hypot(gx, gy)
    lap = np.roll(p, 1, 0) + np.roll(p, -1, 0) + np.roll(p, 1, 1) + np.roll(p, -1, 1) - 4 * p

    # Trim the image border out of the mask: gradients computed there wrap or
    # straddle the illusion boundary, which is a real discontinuity in the scene
    # rather than a property of the model's smoothness.
    core = inside.copy()
    core[0, :] = core[-1, :] = core[:, 0] = core[:, -1] = False
    if core.sum() < 200:
        core = inside

    ys, xs = np.nonzero(core)
    design = np.stack([xs.astype(float), ys.astype(float), np.ones(xs.size)], axis=1)
    coef, *_ = np.linalg.lstsq(design, p[core], rcond=None)
    resid = p[core] - design @ coef
    return {
        "grad_mag": float(np.median(grad[core])),
        "roughness": float(np.sqrt(np.mean(lap[core] ** 2))),
        "residual_rms": float(np.sqrt(np.mean(resid**2))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="visual_illusion_3d")
    parser.add_argument("--variant", default="real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--experiment-name", default="smoothness_diagnostic")
    args = parser.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)
    t0 = time.time()

    # The AUROC each checkpoint achieved, read from its sweep run rather than
    # retyped -- a transcribed number here would silently decide the conclusion.
    auroc: dict[str, float] = {}
    for tag in SWEEP:
        hits = sorted(Path(args.root).glob(f"external_identifiability_{tag}/run_*/metrics/metrics.json"))
        if not hits:
            LOGGER.error("no sweep run for %s -- run scripts/evaluate_identifiability.py first", tag)
            return 1
        auroc[tag] = json.loads(hits[-1].read_text())["auroc_predicting_failure"]["identifiability_ours"]

    reader = get_reader(args.dataset, variant=args.variant)
    if not reader.verify().get("verified"):
        LOGGER.error("dataset verification failed")
        return 1

    per_model: dict[str, dict] = {}
    rows: list[dict] = []
    for tag, ckpt in SWEEP.items():
        encoder = MonocularDepthEncoder(checkpoint=ckpt)
        values: list[dict] = []
        for sample in reader:
            if args.limit is not None and len(values) >= args.limit:
                break
            mask, disp = sample.extra.get("mask"), sample.disparity
            if mask is None or disp is None:
                continue
            inside = (np.asarray(mask) > 127) & np.isfinite(disp) & (disp > 0)
            measured = measure(encoder.predict_inverse_depth(sample.image), inside)
            if measured is None:
                continue
            values.append(measured)
            rows.append({"model": tag, "key": sample.key, **measured})
        if not values:
            LOGGER.error("no measurable images for %s", tag)
            return 1
        per_model[tag] = {
            "checkpoint": ckpt,
            "params_m": CHECKPOINTS.get(ckpt, {}).get("params_m"),
            "n_images": len(values),
            "identifiability_auroc": auroc[tag],
            **{
                k: float(np.median([v[k] for v in values]))
                for k in ("grad_mag", "roughness", "residual_rms")
            },
        }
        LOGGER.info(
            "%-12s grad %.4f  rough %.4f  resid %.4f  (auroc %.3f, %d images)",
            tag, per_model[tag]["grad_mag"], per_model[tag]["roughness"],
            per_model[tag]["residual_rms"], auroc[tag], len(values),
        )

    tags = list(per_model)
    a = np.array([per_model[t]["identifiability_auroc"] for t in tags])
    corr = {
        k: spearman(np.array([per_model[t][k] for t in tags]), a)
        for k in ("grad_mag", "roughness", "residual_rms")
    }
    fam = [t for t in ("dav2s", "dav2b", "dav2l") if t in per_model]
    fam_mono = {
        k: bool(np.all(np.diff([per_model[t][k] for t in fam]) < 0))
        for k in ("grad_mag", "roughness", "residual_rms")
    }

    metrics = {
        "experiment": args.experiment_name,
        "hypothesis": (
            "Identifiability degrades on larger models because they predict more smoothly, "
            "leaving less planarity deviation for delta_DE to measure."
        ),
        "prediction_if_true": (
            "grad_mag, roughness and residual_rms all DECREASE with model size within the "
            "DA-V2 family, and all correlate POSITIVELY with the identifiability AUROC."
        ),
        "n_checkpoints": len(tags),
        "per_model": per_model,
        "spearman_vs_auroc": corr,
        "decreasing_with_size_within_dav2_family": fam_mono,
        "caveat": (
            "n=4 checkpoints. These correlations describe the direction of a relationship "
            "and cannot establish significance. A consistent sign across all four is "
            "evidence; a magnitude is not."
        ),
    }
    config = {
        "experiment": {"name": args.experiment_name, "kind": "diagnostic", "seed": args.seed},
        "data": {"dataset": args.dataset, "variant": args.variant},
        "model": {"checkpoints": SWEEP},
    }
    run_dir = create_run_directory(config, seed=args.seed, root=args.root, extra={"seed_report": seed_report})
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "per_image.csv", rows)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    print("\n" + "=" * 78)
    print(f"SMOOTHNESS DIAGNOSTIC: {run_dir.path}")
    print("=" * 78)
    print(f"  {'model':13s} {'params':>8s} {'AUROC':>7s} {'grad':>9s} {'rough':>9s} {'resid':>9s}")
    for t in tags:
        m = per_model[t]
        print(f"  {t:13s} {m['params_m'] or 0:8.1f} {m['identifiability_auroc']:7.3f} "
              f"{m['grad_mag']:9.4f} {m['roughness']:9.4f} {m['residual_rms']:9.4f}")
    print("\n  Spearman against AUROC (n=4, directional only):")
    for k, v in corr.items():
        print(f"    {k:16s} {v:+.3f}")
    print("\n  strictly decreasing with size within the DA-V2 family:")
    for k, v in fam_mono.items():
        print(f"    {k:16s} {v}")
    print("\n  The hypothesis predicts smoothness rises with size (grad/rough/resid FALL)")
    print("  and that lower AUROC accompanies smoother predictions (positive Spearman).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
