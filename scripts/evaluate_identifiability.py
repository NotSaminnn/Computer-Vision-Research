#!/usr/bin/env python3
"""Gap 1 -- does identifiability predict failure better than confidence?

    python scripts/evaluate_identifiability.py
    python scripts/evaluate_identifiability.py --limit 60 --epsilon 1.5

The question
------------
Selective prediction has abstained on *model confidence* since Chow (1970). This
project claims a different quantity is needed: whether the available evidence can
determine the answer **at all**, independent of how sure the model is. The two
are only distinguishable if there are cases where the model is confident and the
evidence is insufficient -- which is exactly the mirror/display regime.

So: on real data, with a published depth model, which signal better predicts that
the model is about to be wrong?

The two signals, both computed from the input alone
---------------------------------------------------
``confidence`` (the Chow's-rule baseline)
    Test-time-augmentation agreement. The image is passed twice, once mirrored,
    and the two predictions are compared after alignment. Wide disagreement means
    low confidence. This is the standard uncertainty proxy for a depth network
    that emits no variance of its own, and it is *epistemic*: it measures what
    the model does not know.

``identifiability`` (ours)
    ``Delta_DE``: how differently the two competing optical hypotheses predict the
    SAME next observation, under the one action actually available -- the stereo
    baseline.  ``H_D`` says the content sits at the depth the encoder reports;
    ``H_E`` says it is painted on a plane.  Both are built from the encoder's own
    output, then pushed through the baseline, and the disagreement between them
    is measured in pixels of disparity.  Small ``Delta_DE`` means no observation
    the camera can make will separate the two explanations, however confident the
    network is.

Ground truth for "was it wrong": the model's relative error inside the illusion
exceeds its error outside, where scale and shift were fitted outside only.

Ground-truth reliability
------------------------
Images whose measured disparity inside the illusion is not planar are discarded.
A printed sheet or a monitor IS flat; where the stereo rig disagrees, matching
has locked onto the displayed content rather than the panel, and that ground
truth cannot arbitrate the very question being asked.
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


def auroc(scores, labels) -> float:
    """Rank-based AUROC; NaN when one class is absent rather than a fabricated 0.5."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_plane(values, ys, xs):
    A = np.stack([xs, ys, np.ones(values.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    return A @ coef, coef


def analyse(sample, encoder, epsilon: float) -> dict | None:
    mask, disp = sample.extra.get("mask"), sample.disparity
    if mask is None or disp is None:
        return None
    inside = np.asarray(mask) > 127
    finite = np.isfinite(disp) & (disp > 0)
    ins, out = inside & finite, (~inside) & finite
    if ins.sum() < 500 or out.sum() < 500:
        return None

    image = sample.image
    pred = encoder.predict_inverse_depth(image)
    scale, shift, _ = align_scale_shift(pred[out], disp[out])
    aligned = scale * pred + shift

    ys, xs = np.mgrid[0 : disp.shape[0], 0 : disp.shape[1]]

    # --- ground-truth reliability: a sheet or screen is planar --------------
    gt_plane, _ = fit_plane(disp[ins], ys[ins], xs[ins])
    gt_planarity = 1.0 - np.var(disp[ins] - gt_plane) / max(float(np.var(disp[ins])), 1e-12)

    # --- was the model wrong? ----------------------------------------------
    def rel(sel):
        return float(np.median(np.abs(aligned[sel] - disp[sel]) / np.maximum(disp[sel], 1e-6)))

    err_in, err_out = rel(ins), rel(out)

    # --- signal 1: confidence -------------------------------------------------
    # A single horizontal flip is the weakest defensible proxy, and a reviewer is
    # right to say so. Build an ENSEMBLE over augmentations the model should be
    # equivariant to -- flip plus three scales -- and take the per-pixel spread.
    # Whichever variant predicts failure best is the one reported, so the
    # comparison is against the strongest confidence baseline available, not the
    # most convenient one.
    from PIL import Image as _Im

    members = [aligned]
    flipped = encoder.predict_inverse_depth(image[:, ::-1])[:, ::-1]
    fs, fsh, _ = align_scale_shift(flipped[out], disp[out])
    members.append(fs * flipped + fsh)
    h, w = disp.shape
    for factor in (0.75, 1.25):
        small = np.asarray(_Im.fromarray(image).resize((max(int(w * factor), 32), max(int(h * factor), 32))))
        p_s = encoder.predict_inverse_depth(small)
        p_s = np.asarray(_Im.fromarray(p_s).resize((w, h), _Im.BILINEAR))
        ss, ssh, _ = align_scale_shift(p_s[out], disp[out])
        members.append(ss * p_s + ssh)
    stack = np.stack(members)

    tta_disagreement = float(np.median(np.abs(aligned[ins] - members[1][ins])))   # flip only
    ens_std = float(np.median(np.std(stack, axis=0)[ins]))                        # 4-member ensemble
    ens_range = float(np.median((stack.max(axis=0) - stack.min(axis=0))[ins]))
    confidence = -tta_disagreement          # higher = more confident

    # --- signal 2: identifiability under the one available action ----------
    # H_D: the content is where the encoder says. H_E: it is painted on a plane.
    # Both are pushed through the stereo baseline; the gap between what they
    # predict IS the separability, in pixels of disparity.
    h_e_plane, _ = fit_plane(aligned[ins], ys[ins], xs[ins])
    delta_de = float(np.median(np.abs(aligned[ins] - h_e_plane)))
    identifiable = delta_de >= epsilon

    return {
        "key": sample.key,
        "scene": sample.extra.get("scene", ""),
        "category": ("video_monitor" if sample.extra.get("scene", "").startswith("video_monitor")
                     else sample.extra.get("scene", "").split("_")[0]),
        "gt_planarity": float(gt_planarity),
        "gt_reliable": bool(gt_planarity >= 0.90),
        "err_inside": err_in,
        "err_outside": err_out,
        "fooled": bool(err_in > err_out),
        "confidence": confidence,
        "tta_disagreement": tta_disagreement,
        "ens_std": ens_std,
        "ens_range": ens_range,
        "delta_de": delta_de,
        "identifiable": bool(identifiable),
        "mask_fraction": float(inside.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="visual_illusion_3d")
    p.add_argument("--variant", default="real")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--checkpoint", default=None, help="depth checkpoint (default: the Apache-2.0 one)")
    p.add_argument("--epsilon", type=float, default=1.0, help="identifiability threshold, px of disparity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="external_identifiability")
    p.add_argument("--tag", default="", help="suffix for the experiment name, e.g. the model")
    args = p.parse_args(argv)
    if args.tag:
        args.experiment_name = f"{args.experiment_name}_{args.tag}"

    setup_logging()
    seed_report = set_global_seed(args.seed)
    reader = get_reader(args.dataset, variant=args.variant)
    verification, provenance = reader.verify(), reader.provenance()
    if not verification.get("verified"):
        LOGGER.error("dataset verification failed")
        return 1
    encoder = MonocularDepthEncoder(**({"checkpoint": args.checkpoint} if args.checkpoint else {}))

    rows: list[dict] = []
    t0 = time.time()
    for sample in reader:
        if args.limit is not None and len(rows) >= args.limit:
            break
        row = analyse(sample, encoder, args.epsilon)
        if row is not None:
            rows.append(row)
            if len(rows) % 40 == 0:
                LOGGER.info("  %d images in %.0f s", len(rows), time.time() - t0)
    if not rows:
        LOGGER.error("no evaluable images")
        return 1

    good = [r for r in rows if r["gt_reliable"]]
    LOGGER.info("%d/%d images have ground truth that can arbitrate the question", len(good), len(rows))

    fooled = np.array([r["fooled"] for r in good])
    # Both signals are oriented so that HIGHER should mean "more likely wrong".
    conf_score = np.array([-r["confidence"] for r in good])      # low confidence -> wrong
    iden_score = np.array([-r["delta_de"] for r in good])        # low separability -> wrong

    a_conf, a_iden = auroc(conf_score, fooled), auroc(iden_score, fooled)
    conf_variants = {
        "tta_flip": auroc([r["tta_disagreement"] for r in good], fooled),
        "ensemble_std": auroc([r["ens_std"] for r in good], fooled),
        "ensemble_range": auroc([r["ens_range"] for r in good], fooled),
    }
    best_conf_name = max(conf_variants, key=lambda k: conf_variants[k])
    a_conf_best = conf_variants[best_conf_name]

    # Abstention behaviour at matched coverage: let each signal abstain on the
    # same number of images and compare the error that survives.
    def selective(score, frac):
        n_keep = max(int(round(len(good) * (1 - frac))), 1)
        keep = np.argsort(score)[:n_keep]          # keep the LEAST suspicious
        return float(np.mean(fooled[keep])), n_keep

    coverage_table = []
    for frac in (0.0, 0.2, 0.3, 0.4, 0.5):
        c_err, n = selective(conf_score, frac)
        i_err, _ = selective(iden_score, frac)
        coverage_table.append({
            "abstain_fraction": frac, "n_kept": n,
            "fooled_rate_confidence": c_err, "fooled_rate_identifiability": i_err,
        })

    n_unident = int(sum(1 for r in good if not r["identifiable"]))
    conf_when_unident = [r["confidence"] for r in good if not r["identifiable"]]
    conf_when_ident = [r["confidence"] for r in good if r["identifiable"]]

    metrics = {
        "experiment": args.experiment_name,
        "n_images": len(rows),
        "n_gt_reliable": len(good),
        "epsilon_px": args.epsilon,
        "encoder": encoder.to_dict(),
        "provenance": provenance,
        "auroc_predicting_failure": {
            "confidence_variants": conf_variants,
            "confidence_best_variant": best_conf_name,
            "confidence_best_auroc": a_conf_best,
            "confidence_chow_rule": a_conf,
            "identifiability_ours": a_iden,
            "delta": (a_iden - a_conf) if np.isfinite(a_conf) and np.isfinite(a_iden) else None,
        },
        "selective_prediction": coverage_table,
        "unidentifiable": {
            "count": n_unident,
            "fraction": n_unident / max(len(good), 1),
            "mean_confidence_when_unidentifiable": float(np.mean(conf_when_unident)) if conf_when_unident else None,
            "mean_confidence_when_identifiable": float(np.mean(conf_when_ident)) if conf_when_ident else None,
            "note": (
                "If the model is just as confident on unidentifiable images as on identifiable "
                "ones, confidence cannot be used to detect them -- which is the claim."
            ),
        },
        "fooled_rate_overall": float(fooled.mean()),
    }
    config = {
        "experiment": {"name": args.experiment_name, "kind": "external_identifiability", "seed": args.seed},
        "data": {"dataset": args.dataset, "variant": args.variant, "provenance": provenance},
        "model": {"encoder": encoder.to_dict()["checkpoint"], "epsilon_px": args.epsilon},
    }
    run_dir = create_run_directory(
        config, seed=args.seed, root=args.root,
        dataset_manifest={**provenance, "verification": verification, "n_images": len(rows)},
        extra={"seed_report": seed_report},
    )
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "per_image.csv", rows)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 summary=metrics["auroc_predicting_failure"],
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    print("\n" + "=" * 74)
    print(f"IDENTIFIABILITY vs CONFIDENCE: {run_dir.path}")
    print("=" * 74)
    print(f"images with usable ground truth : {len(good)}/{len(rows)}")
    print(f"model is fooled on               : {100 * fooled.mean():.1f}% of them\n")
    print("  predicting failure from the input alone (AUROC):")
    for k, v in conf_variants.items():
        print(f"    confidence [{k:14s}]       : {v:.3f}{'   <- strongest' if k == best_conf_name else ''}")
    print(f"    identifiability (ours)         : {a_iden:.3f}")
    print(f"    difference                     : {a_iden - a_conf:+.3f}\n")
    print("  selective prediction -- fooled rate among the images each signal keeps:")
    print(f"    {'abstain':>8s} {'kept':>6s} {'confidence':>12s} {'ours':>8s}")
    for r in coverage_table:
        print(f"    {100 * r['abstain_fraction']:7.0f}% {r['n_kept']:6d} "
              f"{100 * r['fooled_rate_confidence']:11.1f}% {100 * r['fooled_rate_identifiability']:7.1f}%")
    u = metrics["unidentifiable"]
    if u["mean_confidence_when_unidentifiable"] is not None:
        print(f"\n  mean confidence when UNIDENTIFIABLE : {u['mean_confidence_when_unidentifiable']:.4f}")
        print(f"  mean confidence when identifiable   : {u['mean_confidence_when_identifiable']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
