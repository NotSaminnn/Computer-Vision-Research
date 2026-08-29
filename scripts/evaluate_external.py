#!/usr/bin/env python3
"""E10 -- external evaluation on independent, published data.

    python scripts/evaluate_external.py --limit 40
    python scripts/evaluate_external.py --checkpoint depth-anything/Depth-Anything-V2-Small-hf

The question
------------
3D Visual Illusion (NeurIPS 2025) photographs printed sheets and monitors inside
real rooms, and its stereo rig reports **contact** geometry: the depth of the
paper, not of the scene printed on it. Verified on this data -- inside the
illusion mask the measured disparity is nearly constant (a flat sheet) while the
surrounding room carries real structure.

So a monocular depth model faces a decision it is never asked to make explicitly:
report the surface it could touch, or the scene it appears to see. This script
measures which one it reports, and whether the mistake is predictable *before*
looking at the ground truth.

What is measured, per image
---------------------------
``scale_shift_error_inside`` / ``_outside``
    Relative disparity error after a scale-and-shift alignment fitted **only on
    the non-illusion region**. Fitting outside and testing inside is what makes
    this a test rather than a curve fit: the model is calibrated on the part of
    the scene it understands, then scored on the part that fools it.
``planarity_gt`` / ``planarity_pred``
    R^2 of a plane fitted to the disparity inside the mask. The ground truth is
    a flat sheet, so a model that reports the *depicted* geometry is markedly
    less planar than the truth. This is the fooled-ness signal, and it needs no
    metric scale.

Nothing here is trained, tuned, or fitted on this dataset. The depth model is a
published checkpoint used as released.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import _bootstrap  # noqa: F401
from intervene3d.data.external.loaders import get_reader
from intervene3d.models.foundation_encoders import MonocularDepthEncoder, align_scale_shift
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.io import dump_json, write_csv
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)


def plane_r2(values: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float:
    """R^2 of a least-squares plane through ``values`` -- 1.0 means perfectly flat."""
    if values.size < 16:
        return float("nan")
    A = np.stack([xs, ys, np.ones(values.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    resid = values - A @ coef
    var = float(np.var(values))
    return float(1.0 - np.var(resid) / var) if var > 0 else float("nan")


def evaluate_sample(sample, encoder) -> dict | None:
    """One image: calibrate the model outside the illusion, score it inside."""
    mask = sample.extra.get("mask")
    disp = sample.disparity
    if mask is None or disp is None:
        return None
    inside = np.asarray(mask) > 127
    finite = np.isfinite(disp) & (disp > 0)
    ins, out = inside & finite, (~inside) & finite
    if ins.sum() < 500 or out.sum() < 500:
        return None

    # The model predicts RELATIVE inverse depth, which is proportional to
    # disparity up to an unknown scale and shift -- exactly the two parameters
    # align_scale_shift fits.
    pred = encoder.predict_inverse_depth(sample.image)

    scale, shift, fit = align_scale_shift(pred[out], disp[out])
    aligned = scale * pred + shift

    def rel_err(sel):
        e = np.abs(aligned[sel] - disp[sel]) / np.maximum(disp[sel], 1e-6)
        return float(np.median(e))

    ys, xs = np.mgrid[0 : disp.shape[0], 0 : disp.shape[1]]
    scene = sample.extra.get("scene", "")
    return {
        "key": sample.key,
        "scene": scene,
        "category": ("video_monitor" if scene.startswith("video_monitor")
                     else scene.split("_")[0]),
        "mask_fraction": float(inside.mean()),
        "alignment_r2_outside": fit["r2"],
        "err_outside": rel_err(out),
        "err_inside": rel_err(ins),
        "planarity_gt": plane_r2(disp[ins], ys[ins], xs[ins]),
        "planarity_pred": plane_r2(aligned[ins], ys[ins], xs[ins]),
        "disp_std_gt_inside": float(disp[ins].std()),
        "disp_std_pred_inside": float(aligned[ins].std()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="visual_illusion_3d")
    p.add_argument("--variant", default="real")
    p.add_argument("--checkpoint", default=None, help="depth checkpoint (default: the Apache-2.0 one)")
    p.add_argument("--limit", type=int, default=None, help="evaluate at most this many images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="external_illusion_eval")
    args = p.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)

    reader = get_reader(args.dataset, variant=args.variant)
    verification, provenance = reader.verify(), reader.provenance()
    if not verification.get("verified"):
        LOGGER.error("dataset verification failed: %s", verification)
        return 1
    encoder = MonocularDepthEncoder(**({"checkpoint": args.checkpoint} if args.checkpoint else {}))
    LOGGER.info("source   : %s/%s @ %s", args.dataset, args.variant, provenance.get("revision"))

    rows: list[dict] = []
    t0 = time.time()
    for sample in reader:
        if args.limit is not None and len(rows) >= args.limit:
            break
        row = evaluate_sample(sample, encoder)
        if row is None:
            continue
        rows.append(row)
        if len(rows) % 20 == 0:
            LOGGER.info("  %d images in %.0f s", len(rows), time.time() - t0)
    if not rows:
        LOGGER.error("no evaluable images")
        return 1

    encoder_info = encoder.to_dict()
    config = {
        "experiment": {"name": args.experiment_name, "kind": "external_eval", "seed": args.seed},
        "data": {"dataset": args.dataset, "variant": args.variant, "provenance": provenance},
        "model": {k: encoder_info[k] for k in ("name", "checkpoint", "licence", "device")},
    }
    run_dir = create_run_directory(
        config, seed=args.seed, root=args.root,
        dataset_manifest={**provenance, "verification": verification, "n_images": len(rows)},
        extra={"seed_report": seed_report},
    )

    ins = np.array([r["err_inside"] for r in rows])
    out = np.array([r["err_outside"] for r in rows])
    pg = np.array([r["planarity_gt"] for r in rows])
    pp = np.array([r["planarity_pred"] for r in rows])
    fooled = ins > out

    by_cat: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in rows}):
        sel = [r for r in rows if r["category"] == cat]
        a = np.array([r["err_inside"] for r in sel])
        b = np.array([r["err_outside"] for r in sel])
        by_cat[cat] = {
            "n": len(sel),
            "err_inside_median": float(np.median(a)),
            "err_outside_median": float(np.median(b)),
            "ratio": float(np.median(a) / max(np.median(b), 1e-9)),
            "fooled_rate": float(np.mean(a > b)),
            "planarity_gt_median": float(np.nanmedian([r["planarity_gt"] for r in sel])),
            "planarity_pred_median": float(np.nanmedian([r["planarity_pred"] for r in sel])),
        }

    metrics = {
        "experiment": args.experiment_name,
        "n_images": len(rows),
        "encoder": encoder_info,
        "provenance": provenance,
        "overall": {
            "err_inside_median": float(np.median(ins)),
            "err_outside_median": float(np.median(out)),
            "error_ratio_inside_over_outside": float(np.median(ins) / max(np.median(out), 1e-9)),
            "fooled_rate": float(np.mean(fooled)),
            "planarity_gt_median": float(np.nanmedian(pg)),
            "planarity_pred_median": float(np.nanmedian(pp)),
        },
        "by_category": by_cat,
        "note": (
            "Scale and shift are fitted on the NON-illusion region only, then applied inside. "
            "The depth checkpoint is used as released: nothing is trained or tuned on this dataset."
        ),
    }
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "per_image.csv", rows)

    o = metrics["overall"]
    lines = [
        f"# {args.experiment_name}",
        "",
        f"Source : {args.dataset}/{args.variant} @ `{provenance.get('revision')}` ({provenance.get('licence')})",
        f"Model  : {encoder_info['checkpoint']} ({encoder_info['licence']}), used as released",
        f"Images : {len(rows)}",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| median relative error OUTSIDE the illusion | {o['err_outside_median']:.4f} |",
        f"| median relative error INSIDE the illusion | **{o['err_inside_median']:.4f}** |",
        f"| ratio inside/outside | **{o['error_ratio_inside_over_outside']:.2f}x** |",
        f"| images where inside error exceeds outside | {100 * o['fooled_rate']:.1f}% |",
        f"| planarity of the ground truth inside | {o['planarity_gt_median']:.4f} |",
        f"| planarity of the prediction inside | {o['planarity_pred_median']:.4f} |",
        "",
        "The ground truth inside the illusion is a flat sheet or screen. A model that",
        "reports the *depicted* scene rather than the surface it could touch is both",
        "less accurate and less planar there than the truth.",
        "",
        "| category | n | outside | inside | ratio | fooled |",
        "|---|---|---|---|---|---|",
    ]
    for cat, v in by_cat.items():
        lines.append(
            f"| {cat} | {v['n']} | {v['err_outside_median']:.4f} | {v['err_inside_median']:.4f} "
            f"| {v['ratio']:.2f}x | {100 * v['fooled_rate']:.0f}% |"
        )
    (run_dir.path / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    finalise_run(
        run_dir.path, status="success", metrics_file="metrics/metrics.json",
        summary=o, duration_seconds=round(time.time() - t0, 2), registry_root=args.root,
    )

    print("\n" + "=" * 72)
    print(f"EXTERNAL EVALUATION: {run_dir.path}")
    print("=" * 72)
    print(f"images        : {len(rows)}   model: {encoder_info['checkpoint']}")
    print(f"error outside : {o['err_outside_median']:.4f}")
    print(f"error inside  : {o['err_inside_median']:.4f}   ({o['error_ratio_inside_over_outside']:.2f}x worse)")
    print(f"fooled rate   : {100 * o['fooled_rate']:.1f}% of images")
    print(f"planarity     : GT {o['planarity_gt_median']:.3f} vs predicted {o['planarity_pred_median']:.3f}")
    for cat, v in by_cat.items():
        print(f"  {cat:18s} n={v['n']:3d}  {v['ratio']:5.2f}x  fooled {100 * v['fooled_rate']:3.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
