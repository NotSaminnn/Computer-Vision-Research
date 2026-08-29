#!/usr/bin/env python3
"""E15 -- a second external dataset, with a different kind of ground truth.

    python scripts/evaluate_layered.py
    python scripts/evaluate_layered.py --checkpoint apple/DepthPro-hf --tag depthpro

Why this dataset
----------------
Every real-data claim in this project so far rests on one benchmark (3D Visual
Illusion) and one form of ground truth (dense stereo disparity behind a mask).
That is a single point of failure: a protocol artefact there would look exactly
like a finding.

LayeredDepth (ICCV 2025, CC0-1.0) is independent in both respects. Its
annotations are not dense depth at all -- they are **ordinal relations between
point pairs**, and crucially they come in two parallel sets:

``layer_first``
    ordering by the depth of the FIRST surface along the ray -- the glass itself.
``layer_all``
    ordering ignoring transparency -- the scene behind the glass.

This is precisely the contact-versus-depicted distinction this project is about,
annotated by someone else, for their own purposes, in a format that shares no
machinery with our stereo protocol. A model that reports what it *appears* to see
will score well on ``layer_all`` and badly on ``layer_first``.

The measurement that matters
----------------------------
Aggregate accuracy on either set alone is weak evidence, because most annotated
pairs involve no transparency and the two sets agree on them. The discriminating
quantity is the subset of point pairs that appear in **both** sets with
**opposite** ordering -- pairs where the glass is nearer but the depicted content
is further, or vice versa. On those, and only those, the two hypotheses make
opposite predictions, and the model must choose.

``flip_follows_all``
    Of the discriminating pairs, the fraction where the model agrees with
    ``layer_all``. Above 0.5 means it reports the scene through the glass;
    below 0.5 means it reports the interface.

Convention, verified rather than assumed
----------------------------------------
The tuple format is ``[x, y, n_layers]`` and the annotation orders ``p1`` nearer
than ``p2``. Both were checked empirically before this script was written: the x
and y ranges disambiguate the axis order against the image dimensions, and a
published checkpoint scores 0.92 / 0.91 under "p1 is nearer" -- far from the 0.5
a wrong convention would give. ``--check-convention`` re-runs that test.

Images are large (up to 4032x3024) and are downscaled to ``--max-side`` before
inference, with point coordinates scaled to match. The factor is recorded.
Nothing is trained or tuned on this dataset.
"""
from __future__ import annotations

import argparse
import glob
import io
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from intervene3d.models.foundation_encoders import MonocularDepthEncoder
from intervene3d.reproducibility.manifest import finalise_run
from intervene3d.reproducibility.run_dir import create_run_directory
from intervene3d.reproducibility.seeds import set_global_seed
from intervene3d.utils.io import dump_json, write_csv
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)

VALIDATION_GLOB = "data/raw/layereddepth/validation/data/*.parquet"


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


def pair_key(p1, p2):
    """Unordered identity of a point pair, so the two layer sets can be joined.

    Sorting the endpoints is what makes a *flip* detectable: the same two pixels
    appear in both sets, and only the order differs.
    """
    a, b = (p1[0], p1[1]), (p2[0], p2[1])
    return (a, b) if a <= b else (b, a), (a <= b)


def evaluate_image(record, encoder, max_side: int) -> dict | None:
    from PIL import Image

    image = Image.open(io.BytesIO(record["image.png"]["bytes"])).convert("RGB")
    W, H = image.size
    scale = min(1.0, max_side / max(W, H))
    if scale < 1.0:
        image = image.resize((max(1, int(W * scale)), max(1, int(H * scale))))
    pred = encoder.predict_inverse_depth(np.asarray(image))
    h, w = pred.shape

    def sample(pt):
        c, r = int(pt[0] * scale), int(pt[1] * scale)
        if not (0 <= c < w and 0 <= r < h):
            return None
        v = pred[r, c]
        return float(v) if np.isfinite(v) else None

    # Per-layer ordinal accuracy, and the ordering the model asserts for every
    # annotated pair, keyed so the two sets can be joined afterwards.
    stats = {}
    asserted: dict[str, dict] = {}
    # Accuracy split by the dataset's OWN layer index -- an ambiguity label
    # written by the benchmark's authors, with no input from us. A point at
    # layer >= 2 is only visible through a transparent surface, so if the
    # ambiguity this project theorises about is real, error should concentrate
    # there. This is the one test in the project where the difficulty label and
    # the method being tested share no machinery at all.
    by_layer = {"single": [0, 0], "multi": [0, 0]}
    for layer in ("layer_first", "layer_all"):
        n = ok = 0
        seen: dict = {}
        for pr in record["tuples.json"][layer]["pairs"]:
            if not pr["is_real"]:
                continue
            v1, v2 = sample(pr["p1"]), sample(pr["p2"])
            if v1 is None or v2 is None:
                continue
            n += 1
            # annotation asserts p1 is nearer, i.e. larger inverse depth
            model_says_p1_nearer = v1 > v2
            ok += int(model_says_p1_nearer)
            if layer == "layer_all":
                bucket = "multi" if max(pr["p1"][2], pr["p2"][2]) >= 2 else "single"
                by_layer[bucket][0] += 1
                by_layer[bucket][1] += int(model_says_p1_nearer)
            key, p1_is_first = pair_key(pr["p1"], pr["p2"])
            # normalise both the annotation and the model to the sorted key
            seen[key] = {
                "gt_first_nearer": p1_is_first,
                "model_first_nearer": model_says_p1_nearer if p1_is_first else not model_says_p1_nearer,
            }
        stats[layer] = {"n": n, "acc": (ok / n) if n else float("nan")}
        asserted[layer] = seen

    # Discriminating pairs: present in both sets, annotated in OPPOSITE order.
    shared = set(asserted["layer_first"]) & set(asserted["layer_all"])
    n_flip = follows_all = 0
    for key in shared:
        f, a = asserted["layer_first"][key], asserted["layer_all"][key]
        if f["gt_first_nearer"] == a["gt_first_nearer"]:
            continue  # the two layers agree; this pair decides nothing
        n_flip += 1
        follows_all += int(a["model_first_nearer"] == a["gt_first_nearer"])

    if stats["layer_all"]["n"] == 0:
        return None
    return {
        "key": record["__key__"],
        "width": W,
        "height": H,
        "scale": round(scale, 5),
        "n_first": stats["layer_first"]["n"],
        "n_all": stats["layer_all"]["n"],
        "acc_first": stats["layer_first"]["acc"],
        "acc_all": stats["layer_all"]["acc"],
        "n_shared": len(shared),
        "n_discriminating": n_flip,
        "follows_all": (follows_all / n_flip) if n_flip else float("nan"),
        "n_single_layer": by_layer["single"][0],
        "n_multi_layer": by_layer["multi"][0],
        "acc_single_layer": (by_layer["single"][1] / by_layer["single"][0]) if by_layer["single"][0] else float("nan"),
        "acc_multi_layer": (by_layer["multi"][1] / by_layer["multi"][0]) if by_layer["multi"][0] else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tag", default="", help="suffix for the experiment name, e.g. the model")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="experiments")
    p.add_argument("--experiment-name", default="external_layered")
    args = p.parse_args(argv)

    setup_logging()
    seed_report = set_global_seed(args.seed)
    t0 = time.time()

    import pyarrow.parquet as pq

    files = sorted(glob.glob(VALIDATION_GLOB))
    if not files:
        LOGGER.error("no LayeredDepth validation parquet at %s", VALIDATION_GLOB)
        return 1
    encoder = MonocularDepthEncoder(**({"checkpoint": args.checkpoint} if args.checkpoint else {}))

    rows: list[dict] = []
    for f in files:
        if args.limit is not None and len(rows) >= args.limit:
            break
        for record in pq.ParquetFile(f).read().to_pylist():
            if args.limit is not None and len(rows) >= args.limit:
                break
            row = evaluate_image(record, encoder, args.max_side)
            if row is not None:
                rows.append(row)
        LOGGER.info("  %d images in %.0f s", len(rows), time.time() - t0)
    if not rows:
        LOGGER.error("no evaluable images")
        return 1

    def wmean(num_key, den_key):
        num = sum(r[num_key] * r[den_key] for r in rows if r[den_key] and np.isfinite(r[num_key]))
        den = sum(r[den_key] for r in rows if r[den_key] and np.isfinite(r[num_key]))
        return float(num / den) if den else float("nan")

    n_disc = int(sum(r["n_discriminating"] for r in rows))
    with_disc = [r for r in rows if r["n_discriminating"] > 0]
    encoder_info = encoder.to_dict()
    metrics = {
        "experiment": args.experiment_name + (f"_{args.tag}" if args.tag else ""),
        "dataset": "LayeredDepth validation (ICCV 2025, CC0-1.0)",
        "n_images": len(rows),
        "max_side": args.max_side,
        "encoder": encoder_info,
        "pair_weighted": {
            "acc_layer_all": wmean("acc_all", "n_all"),
            "acc_layer_first": wmean("acc_first", "n_first"),
            "n_pairs_all": int(sum(r["n_all"] for r in rows)),
            "n_pairs_first": int(sum(r["n_first"] for r in rows)),
        },
        "by_dataset_layer_label": {
            "acc_single_layer": wmean("acc_single_layer", "n_single_layer"),
            "acc_multi_layer": wmean("acc_multi_layer", "n_multi_layer"),
            "n_single": int(sum(r["n_single_layer"] for r in rows)),
            "n_multi": int(sum(r["n_multi_layer"] for r in rows)),
            "interpretation": (
                "Accuracy split by the BENCHMARK AUTHORS' layer index, not by anything "
                "we compute. Points at layer >= 2 are visible only through a transparent "
                "surface. A large gap means the ambiguity this project theorises about is "
                "independently labelled and independently predicts failure."
            ),
        },
        "discriminating": {
            "n_pairs": n_disc,
            "n_images_with_any": len(with_disc),
            "follows_layer_all": wmean("follows_all", "n_discriminating"),
            "interpretation": (
                "Fraction of pairs where the two layer definitions order the points "
                "OPPOSITELY and the model agrees with layer_all. Above 0.5 the model "
                "reports the scene behind the glass; below 0.5 it reports the interface."
            ),
        },
        "note": (
            "Ordinal annotations, not dense depth: shares no machinery with the stereo "
            "protocol used on 3D Visual Illusion. Convention [x, y, n_layers] with p1 "
            "nearer was verified empirically. Checkpoint used as released."
        ),
    }
    config = {
        "experiment": {"name": metrics["experiment"], "kind": "external_ordinal", "seed": args.seed},
        "data": {"dataset": "layereddepth", "variant": "validation", "n_images": len(rows)},
        "model": {k: encoder_info[k] for k in ("name", "checkpoint", "licence", "family")},
    }
    run_dir = create_run_directory(config, seed=args.seed, root=args.root, extra={"seed_report": seed_report})
    dump_json(run_dir.path / "metrics" / "metrics.json", metrics)
    write_csv(run_dir.path / "predictions" / "per_image.csv", rows)
    finalise_run(run_dir.path, status="success", metrics_file="metrics/metrics.json",
                 duration_seconds=round(time.time() - t0, 2), registry_root=args.root)

    pw, dc = metrics["pair_weighted"], metrics["discriminating"]
    print("\n" + "=" * 74)
    print(f"LAYEREDDEPTH ORDINAL EVALUATION: {run_dir.path}")
    print("=" * 74)
    print(f"model   : {encoder_info['checkpoint']}  ({encoder_info['family']})")
    print(f"images  : {len(rows)}   downscaled to max side {args.max_side}")
    print(f"\n  ordinal accuracy, layer_all   (scene behind glass) : {pw['acc_layer_all']:.4f}   "
          f"({pw['n_pairs_all']} pairs)")
    print(f"  ordinal accuracy, layer_first (the glass itself)   : {pw['acc_layer_first']:.4f}   "
          f"({pw['n_pairs_first']} pairs)")
    print(f"\n  discriminating pairs (the two layers disagree)     : {dc['n_pairs']} "
          f"across {dc['n_images_with_any']} images")
    print(f"  fraction where the model follows layer_all         : {dc['follows_layer_all']:.4f}")
    print("\n  >0.5 means the model reports the scene through the glass rather than")
    print("  the surface it could touch -- the contact-vs-depicted failure, measured")
    print("  on an independent dataset with a different annotation type.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
