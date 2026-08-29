#!/usr/bin/env python3
"""E16 -- identifiability measured through the action, not guessed from one image.

    python scripts/evaluate_identifiability_stereo.py
    python scripts/evaluate_identifiability_stereo.py --checkpoint apple/DepthPro-hf --tag depthpro

What was wrong with the previous measurement
--------------------------------------------
``scripts/evaluate_identifiability.py`` computes

    delta_de = median |aligned_prediction - fitted_plane|

which is a *planarity deviation read off a single image*. The methodology says
something quite different: separability is the distance between two hypotheses'
predicted **consequences of an action**,

    Delta_ij(a) = D( p(O' | H_i, a), p(O' | H_j, a) )

Nothing in the old script ever applies an action or predicts an observation. It
is a proxy that happened to correlate with the answer on posters and screens, and
two results exposed it: on real glass the direction of the error reverses
(``evaluate_layered.py``), and the deviation it measures grows on larger models
while its predictive power falls (``diagnose_smoothness.py``).

The action was available all along
----------------------------------
Every sample in this dataset carries a second photograph -- ``extra["right"]`` --
the other camera of a calibrated rectified stereo pair. That is a known, executed,
lateral translation: precisely the intervention the method is built around, with
``|A| = 1``. It was never opened.

So this script implements the stated method:

1. **H_D (direct)** asserts the surface lies where the model says. Its predicted
   disparity is the model's own, scale-and-shift aligned outside the illusion.
2. **H_E (emissive)** asserts the same appearance comes from a flat panel. Its
   predicted disparity is the least-squares plane through **the model's own
   predicted disparity** over the region -- never through the ground truth.
   Both hypotheses are generated from ``F_t = E(I_0)`` alone, so Delta stays
   computable at test time from the image.
3. Warp the right image into the left frame under each hypothesis. The two warps
   differ *only* in the disparity used, so their difference is attributable to
   the mechanism and to nothing else:

       Delta_DE = median | R_D - R_E |   over the illusion region

4. Compare each warp against the actual left image to see which mechanism
   explains the observation -- the belief update, on real photographs.

Why this is the right quantity, and why it is not the obvious one
-----------------------------------------------------------------
Delta is now in **intensity units**, and that is the point. A disparity
disagreement between two hypotheses is only detectable where the image has
structure to reveal it: on a blank patch of screen, the two hypotheses predict
different geometry and *identical pixels*, so the scene is genuinely
unidentifiable no matter how large the baseline. The old disparity-space proxy
scored such a region as highly identifiable, which is exactly backwards.

Comparing the two warps to each other, rather than either to the true right
image, is deliberate: both are warps of the same photograph, so illumination
differences, sensor noise and rectification error appear in both and cancel.

The sign convention (right-frame sample at ``x - d``) was verified empirically
against this data before this script was written, not assumed.
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


def warp_left(right_gray: np.ndarray, disparity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample the right image into the left frame: ``R(x, y) = I1(x - d, y)``.

    Bilinear along x only -- the pair is rectified, so there is no vertical
    component and introducing one would invent structure. Returns the warped
    image and a validity mask; out-of-frame samples are invalid rather than
    clamped, because clamping fabricates edge pixels that then look like
    hypothesis disagreement.
    """
    H, W = disparity.shape
    xs = np.arange(W, dtype=np.float64)[None, :]
    src = xs - disparity
    valid = np.isfinite(src) & (src >= 0.0) & (src <= W - 1.0)
    safe = np.where(valid, src, 0.0)
    x0 = np.floor(safe).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    fx = safe - x0
    rows = np.arange(H)[:, None]
    warped = right_gray[rows, x0] * (1.0 - fx) + right_gray[rows, x1] * fx
    return warped, valid


def plane_fit(values: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Least-squares plane over the selected pixels, evaluated everywhere."""
    ys, xs = np.nonzero(sel)
    design = np.stack([xs.astype(float), ys.astype(float), np.ones(xs.size)], axis=1)
    coef, *_ = np.linalg.lstsq(design, values[sel], rcond=None)
    H, W = values.shape
    yy, xx = np.mgrid[0:H, 0:W]
    return coef[0] * xx + coef[1] * yy + coef[2]


def evaluate_sample(sample, encoder, epsilon: float) -> dict | None:
    mask, disp, right = sample.extra.get("mask"), sample.disparity, sample.extra.get("right")
    if mask is None or disp is None or right is None:
        return None
    inside = np.asarray(mask) > 127
    finite = np.isfinite(disp) & (disp > 0)
    ins, out = inside & finite, (~inside) & finite
    if ins.sum() < 500 or out.sum() < 500:
        return None

    left_gray = sample.image.astype(np.float64).mean(axis=-1)
    right_gray = np.asarray(right).astype(np.float64)
    if right_gray.ndim == 3:
        right_gray = right_gray.mean(axis=-1)

    # H_D: the surface is where the model says. Calibrate OUTSIDE, apply inside,
    # so the alignment cannot absorb the effect being measured.
    pred = encoder.predict_inverse_depth(sample.image)
    scale, shift, fit = align_scale_shift(pred[out], disp[out])
    d_direct = scale * pred + shift
    # H_E: the same appearance, produced by a flat panel.
    #
    # BOTH hypotheses must be generated from the model's own estimate F_t = E(I_0).
    # Fitting this plane to the ground-truth disparity instead -- as an earlier
    # version of this script did -- is wrong twice over. It puts ground truth
    # inside a quantity that must be computable at test time from the image
    # alone, so it is not a usable signal; and because `gt_reliable` selects
    # regions whose truth is already near-planar, the fitted plane collapses onto
    # the ground truth and Delta silently becomes |warp-by-model - warp-by-truth|,
    # a model *error* measure rather than a separability between two mechanisms.
    # The hypothesis is "this content lies on a plane", and the plane it refers to
    # is the one the model's own geometry implies.
    d_emissive = plane_fit(d_direct, ins)

    R_d, ok_d = warp_left(right_gray, d_direct)
    R_e, ok_e = warp_left(right_gray, d_emissive)
    both = ins & ok_d & ok_e
    if both.sum() < 500:
        return None

    # Separability: the two hypotheses' predicted observations, differing only in
    # the disparity used. Intensity units.
    delta_de = float(np.median(np.abs(R_d[both] - R_e[both])))

    # Belief update: which mechanism actually explains the observed left image.
    resid_d = float(np.median(np.abs(R_d[both] - left_gray[both])))
    resid_e = float(np.median(np.abs(R_e[both] - left_gray[both])))

    # How much structure is there to reveal a disagreement at all? Reported so a
    # low Delta can be attributed to a textureless region rather than to agreement.
    gy, gx = np.gradient(left_gray)
    texture = float(np.median(np.abs(gx)[both]))
    disp_gap = float(np.median(np.abs(d_direct[both] - d_emissive[both])))

    mask_fraction = float(inside.mean())

    def rel_err(sel):
        e = np.abs(d_direct[sel] - disp[sel]) / np.maximum(disp[sel], 1e-6)
        return float(np.median(e))

    gt_plane = plane_fit(disp, ins)
    gt_planarity = 1.0 - np.var(disp[ins] - gt_plane[ins]) / max(float(np.var(disp[ins])), 1e-12)
    err_in, err_out = rel_err(ins), rel_err(out)
    return {
        "key": sample.key,
        "scene": sample.extra.get("scene", ""),
        "category": ("video" if sample.extra.get("scene", "").startswith("video")
                     else sample.extra.get("scene", "").split("_")[0]),
        "gt_planarity": float(gt_planarity),
        "gt_reliable": bool(gt_planarity >= 0.90),
        "err_inside": err_in,
        "err_outside": err_out,
        "fooled": bool(err_in > err_out),
        "alignment_r2_outside": fit["r2"],
        "delta_de_photometric": delta_de,
        "identifiable": bool(delta_de >= epsilon),
        "residual_direct": resid_d,
        "residual_emissive": resid_e,
        "belief_favours_emissive": bool(resid_e < resid_d),
        "texture_x": texture,
        "mask_fraction": mask_fraction,
        "disparity_gap": disp_gap,
        "n_pixels": int(both.sum()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="visual_illusion_3d")
    p.add_argument("--variant", default="real")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epsilon", type=float, default=2.0,
                   help="identifiability threshold in INTENSITY units (0-255), not pixels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="external_identifiability_stereo")
    args = p.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)
    t0 = time.time()

    reader = get_reader(args.dataset, variant=args.variant)
    if not reader.verify().get("verified"):
        LOGGER.error("dataset verification failed")
        return 1
    encoder = MonocularDepthEncoder(**({"checkpoint": args.checkpoint} if args.checkpoint else {}))

    rows: list[dict] = []
    for sample in reader:
        if args.limit is not None and len(rows) >= args.limit:
            break
        row = evaluate_sample(sample, encoder, args.epsilon)
        if row is not None:
            rows.append(row)
        if len(rows) % 40 == 0 and rows:
            LOGGER.info("  %d images in %.0f s", len(rows), time.time() - t0)
    if not rows:
        LOGGER.error("no evaluable images")
        return 1

    good = [r for r in rows if r["gt_reliable"]]
    if not good:
        LOGGER.error("no images with arbitrating ground truth")
        return 1
    wrong = np.array([r["fooled"] for r in good])
    col = lambda k: np.array([r[k] for r in good], float)
    # Low separability -> likely wrong, so the score is negated to make "larger
    # means more likely wrong" for AUROC, matching the old script's convention.
    iden = np.array([-r["delta_de_photometric"] for r in good])

    name = args.experiment_name + (f"_{args.tag}" if args.tag else "")
    encoder_info = encoder.to_dict()
    metrics = {
        "experiment": name,
        "n_images": len(rows),
        "n_gt_reliable": len(good),
        "fooled_rate_overall": float(wrong.mean()),
        "epsilon_intensity_units": args.epsilon,
        "encoder": encoder_info,
        "auroc_predicting_failure": {
            "identifiability_stereo": auroc(iden, wrong),
            # TRIVIAL BASELINES, reported whether or not they flatter the method.
            #
            # `mask_fraction` -- how much of the frame the ambiguous region covers
            # -- needs no model, no action and no theory, and on this data it is
            # the strongest single predictor of failure by a wide margin. Any
            # reviewer can compute it in five minutes, so omitting it would not
            # protect the claim, only delay the objection. A separability measure
            # that cannot beat "the illusion is big" has not earned its complexity.
            "baseline_mask_fraction": auroc(col("mask_fraction"), wrong),
            # Delta is correlated with local texture, so texture alone is reported
            # to show how much of Delta's performance is just image structure.
            "baseline_texture": auroc(col("texture_x"), wrong),
            # The disparity-space disagreement, before the photometric weighting.
            "baseline_disparity_gap": auroc(col("disparity_gap"), wrong),
        },
        "baseline_note": (
            "AUROCs are reported with the sign that makes each score's LARGER value "
            "mean 'more likely wrong'. A value below 0.5 therefore means the signal "
            "is anti-correlated with failure and would be better used inverted; that "
            "is stated rather than hidden by silently flipping the sign."
        ),
        "separability": {
            "delta_median": float(np.median([r["delta_de_photometric"] for r in good])),
            "disparity_gap_median": float(np.median([r["disparity_gap"] for r in good])),
            "texture_median": float(np.median([r["texture_x"] for r in good])),
            "unidentifiable_rate": float(np.mean([not r["identifiable"] for r in good])),
        },
        "belief_update": {
            "favours_emissive_rate": float(np.mean([r["belief_favours_emissive"] for r in good])),
            "residual_direct_median": float(np.median([r["residual_direct"] for r in good])),
            "residual_emissive_median": float(np.median([r["residual_emissive"] for r in good])),
        },
        "method": (
            "Delta is the median absolute difference between the right image warped into "
            "the left frame under H_D and under H_E, in intensity units. Both are warps of "
            "the same photograph, so illumination and sensor differences cancel and the "
            "residual is attributable to the mechanism alone."
        ),
    }
    config = {
        "experiment": {"name": name, "kind": "external_identifiability_stereo", "seed": args.seed},
        "data": {"dataset": args.dataset, "variant": args.variant, "n_images": len(rows)},
        "model": {k: encoder_info[k] for k in ("name", "checkpoint", "licence", "family")},
    }
    run_dir = create_run_directory(config, seed=args.seed, root=args.root, extra={"seed_report": seed_report})
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "per_image.csv", rows)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    sep, bel = metrics["separability"], metrics["belief_update"]
    print("\n" + "=" * 76)
    print(f"STEREO IDENTIFIABILITY: {run_dir.path}")
    print("=" * 76)
    print(f"model  : {encoder_info['checkpoint']} ({encoder_info['family']})")
    print(f"images : {len(rows)}   with arbitrating ground truth: {len(good)}")
    print(f"fooled : {100 * wrong.mean():.1f}%\n")
    print(f"  Delta (intensity units)          median : {sep['delta_median']:.3f}")
    print(f"  disparity gap between hypotheses median : {sep['disparity_gap_median']:.2f} px")
    print(f"  horizontal texture               median : {sep['texture_median']:.3f}")
    print(f"  unidentifiable at eps={args.epsilon:<5.2f}         : {100 * sep['unidentifiable_rate']:.1f}%")
    print(f"\n  AUROC predicting failure (stereo Delta) : "
          f"{metrics['auroc_predicting_failure']['identifiability_stereo']:.3f}")
    print(f"\n  belief favours H_E (the flat panel)     : {100 * bel['favours_emissive_rate']:.1f}% of images")
    print(f"    residual under H_D : {bel['residual_direct_median']:.3f}")
    print(f"    residual under H_E : {bel['residual_emissive_median']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
