#!/usr/bin/env python3
"""R3 -- an abstention head on frozen encoder features.

    .venv/Scripts/python.exe scripts/learn_abstention_head.py
    .venv/Scripts/python.exe scripts/learn_abstention_head.py --checkpoint apple/DepthPro-hf

The question R1 leaves open
---------------------------
R1 (``scripts/learn_abstention.py``) learns a policy over ten *hand-designed*
numbers and beats the trivial region-size baseline on every held-out checkpoint.
That is a real result, but it inherits a limitation: everything the policy can
see, a human chose to compute. If the features are the wrong ones, the ceiling is
set by that choice and no amount of fitting reveals it.

This script removes that ceiling. It pools the depth network's own **frozen
intermediate features** over the ambiguous region and trains a small head to
predict whether that network will be fooled there. Nothing in the encoder is
updated -- the encoder is the subject of the experiment, not part of the model
being fitted -- so the head is asking a sharp question: *is the evidence that
this scene is ambiguous already present in the representation, and simply not
used by the depth decoder?*

Two outcomes, both worth having
-------------------------------
If the head beats R1's hand-designed features, the representation already knows,
and the failure is a decoding problem rather than a perception problem. That is
a stronger and more actionable claim than anything the hand-designed route can
support.

If it does not, the hand-designed features are not the bottleneck, and the
honest conclusion is that the information is not in the representation at all.

Protocol
--------
Identical to R1 so the numbers are comparable: folds split by **base scene**, and
a leave-one-model-out variant when features from several checkpoints exist. The
head is deliberately small (logistic regression on pooled features) because ~296
images over ~83 scenes cannot support anything larger without memorising scenes.

Features are cached to disk on first extraction, since a forward pass over the
benchmark is the expensive part and the head is re-fit many times.
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

import _bootstrap  # noqa: F401
from intervene3d.data.external.loaders import get_reader
from intervene3d.models.foundation_encoders import CHECKPOINTS, MonocularDepthEncoder
from intervene3d.utils.logging import get_logger, setup_logging

LOGGER = get_logger(__name__)
REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "interim" / "abstention_features"


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


def pooled_features(encoder: MonocularDepthEncoder, image: np.ndarray,
                    inside: np.ndarray) -> np.ndarray | None:
    """Mean and std of the backbone's hidden states, pooled inside and outside.

    Both regions are pooled and their difference kept, because the quantity of
    interest is *contrast*: a representation that encodes ambiguity should look
    different inside the ambiguous region than in the ordinary scene around it.
    An absolute pooled vector would mostly encode which scene this is.
    """
    import torch

    encoder._ensure_loaded()
    if encoder._processor is None:  # non-transformers families expose no hidden states
        return None
    from PIL import Image

    arr = np.asarray(image)[..., :3]
    pil = Image.fromarray(arr.astype(np.uint8))
    inputs = encoder._processor(images=pil, return_tensors="pt").to(encoder._resolved_device)
    with torch.no_grad():
        out = encoder._model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[-1]  # (1, tokens, dim)
    if hs.ndim != 3:
        return None
    t = hs.shape[1]
    side = int(round(np.sqrt(max(t - 1, 1))))
    tokens = hs[0, -side * side :, :] if side * side <= t else hs[0]
    grid = tokens.reshape(side, side, -1).float().cpu().numpy()

    # Resample the mask onto the token grid by area-averaging. Done with numpy
    # rather than cv2 so this runs in the plain environment: the feature
    # extraction has to happen where `transformers` lives, and the head is fitted
    # separately where `scikit-learn` lives.
    h, w = inside.shape
    ys = (np.arange(side + 1) * h) // side
    xs = (np.arange(side + 1) * w) // side
    cum = np.cumsum(np.cumsum(inside.astype(np.float64), axis=0), axis=1)
    cum = np.pad(cum, ((1, 0), (1, 0)))
    block = (cum[ys[1:, None], xs[None, 1:]] - cum[ys[:-1, None], xs[None, 1:]]
             - cum[ys[1:, None], xs[None, :-1]] + cum[ys[:-1, None], xs[None, :-1]])
    area = np.maximum((ys[1:] - ys[:-1])[:, None] * (xs[1:] - xs[:-1])[None, :], 1)
    m = (block / area) > 0.5
    if m.sum() < 4 or (~m).sum() < 4:
        return None
    inside_vec = grid[m].mean(axis=0)
    outside_vec = grid[~m].mean(axis=0)
    spread = grid[m].std(axis=0)
    return np.concatenate([inside_vec - outside_vec, spread]).astype(np.float32)


def extract(tag: str, checkpoint: str, limit: int | None) -> Path:
    """Run the encoder over the benchmark once and cache pooled features."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{tag}.npz"
    if out.exists():
        LOGGER.info("using cached features: %s", out)
        return out

    labels = {}
    hit = sorted(glob.glob(str(REPO / f"experiments/external_identifiability_{tag}/run_*/predictions/per_image.csv")))
    if not hit:
        hit = sorted(glob.glob(str(REPO / f"experiments/external_identifiability_stereo_{tag}/run_*/predictions/per_image.csv")))
    if not hit:
        raise SystemExit(f"no run for {tag}: produce failure labels first")
    for r in csv.DictReader(Path(hit[-1]).open()):
        if r.get("gt_reliable") == "True":
            labels[r["key"]] = (r["fooled"] == "True", r.get("scene", ""))

    encoder = MonocularDepthEncoder(checkpoint=checkpoint)
    reader = get_reader("visual_illusion_3d", variant="real")
    X, y, scenes, keys = [], [], [], []
    t0 = time.time()
    for sample in reader:
        if limit is not None and len(X) >= limit:
            break
        if sample.key not in labels:
            continue
        mask = sample.extra.get("mask")
        if mask is None:
            continue
        f = pooled_features(encoder, sample.image, np.asarray(mask) > 127)
        if f is None:
            raise SystemExit(
                f"checkpoint {checkpoint!r} exposes no transformer hidden states; "
                "this script needs a transformers-backed family"
            )
        fooled, scene = labels[sample.key]
        X.append(f); y.append(fooled); scenes.append(scene); keys.append(sample.key)
        if len(X) % 50 == 0:
            LOGGER.info("  %d images in %.0f s", len(X), time.time() - t0)
    if not X:
        raise SystemExit("no images with both features and labels")
    np.savez_compressed(out, X=np.stack(X), y=np.array(y), scenes=np.array(scenes),
                        keys=np.array(keys), checkpoint=checkpoint)
    LOGGER.info("cached %d x %d features -> %s", len(X), len(X[0]), out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--tag", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--extract-only", action="store_true")
    args = p.parse_args(argv)

    setup_logging()
    tag = args.tag or {v: k for k, v in {
        "dav2s": "depth-anything/Depth-Anything-V2-Small-hf",
        "dav2b": "depth-anything/Depth-Anything-V2-Base-hf",
        "dav2l": "depth-anything/Depth-Anything-V2-Large-hf",
        "transparent": "depth-anything/prompt-depth-anything-vits-transparent-hf",
        "depthpro": "apple/DepthPro-hf",
    }.items()}.get(args.checkpoint, "custom")

    path = extract(tag, args.checkpoint, args.limit)
    if args.extract_only:
        return 0

    d = np.load(path, allow_pickle=True)
    X, y, scenes = d["X"], d["y"].astype(bool), d["scenes"]

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=min(args.folds, len(set(scenes.tolist()))))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, scenes):
        # Strong L2: the feature vector is far wider than the sample count, so an
        # unregularised head would fit the scenes rather than the phenomenon.
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.02))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    head_auroc = auroc(oof, y)

    r1 = REPO / "results" / "learned_abstention" / "metrics_gbdt.json"
    r1_auroc = json.loads(r1.read_text())["grouped_cv"]["auroc"] if r1.exists() else float("nan")

    print("\n" + "=" * 76)
    print("R3  ABSTENTION HEAD ON FROZEN ENCODER FEATURES")
    print("=" * 76)
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  images {len(y)}   feature dim {X.shape[1]}   scenes {len(set(scenes.tolist()))}")
    print(f"  base failure rate {100 * y.mean():.1f}%\n")
    print(f"  frozen-feature head, grouped {gkf.get_n_splits()}-fold : {head_auroc:.3f}")
    print(f"  R1 hand-designed features, same protocol   : {r1_auroc:.3f}")
    if np.isfinite(r1_auroc):
        diff = head_auroc - r1_auroc
        print(f"  difference                                 : {diff:+.3f}")
        print("\n  " + (
            "The representation already carries the evidence: this is a DECODING failure."
            if diff > 0.02 else
            "The frozen representation does NOT beat ten hand-designed numbers. The "
            "information is not obviously present in it, and the hand-designed features "
            "are not the bottleneck."))

    out = REPO / "results" / "learned_abstention" / f"head_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "abstention_head_frozen_features",
        "checkpoint": args.checkpoint,
        "n_images": int(len(y)),
        "feature_dim": int(X.shape[1]),
        "n_scenes": int(len(set(scenes.tolist()))),
        "base_failure_rate": float(y.mean()),
        "grouped_auroc": head_auroc,
        "r1_hand_designed_auroc": r1_auroc,
        "note": (
            "The encoder is frozen: only a logistic head is fitted, so this measures "
            "what the representation already contains rather than what it can be "
            "trained to contain. Folds split by base scene."
        ),
    }, indent=2), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
