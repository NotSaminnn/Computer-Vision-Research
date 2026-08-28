"""Generate, render and write the synthetic Intervene3D benchmark.

Layout on disk::

    <root>/<name>/
      manifest.json          dataset manifest: config, hashes, counts, action set, checksums
      scenes.jsonl           one JSON record per scene variant (all metadata)
      splits.json            base-scene -> split assignment and the leakage policy
      scenes/<scene_id>.npz  arrays: reference feature, per-action observations, oracle tensors
      previews/*.png         optional reference-view renders (first few scenes only)

The dataset is fully self-contained: an experiment reads observations from the
npz files and never re-simulates, so a run cannot silently depend on a changed
simulator.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.config.loader import config_hash, full_config_hash
from intervene3d.data.splits import build_splits
from intervene3d.data.synthetic.ground_truth import compute_ground_truth
from intervene3d.data.synthetic.optical_variants import (
    build_hypothesis_set,
    reference_observation,
    simulate,
)
from intervene3d.data.synthetic.scene_generator import BaseScene, generate_base_scene
from intervene3d.data.synthetic.trajectory_generator import (
    action_space_from_config,
    mcrb_baseline_sweep,
)
from intervene3d.data.types import CHANNEL_CONTENT, CHANNEL_FRAME, CHANNEL_MARKER
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
from intervene3d.models.transition import AnalyticalTransitionModel
from intervene3d.utils.io import dump_json, dump_jsonl
from intervene3d.utils.logging import get_logger

LOGGER = get_logger(__name__)
DATASET_FORMAT_VERSION = "1.0"


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ rendering
def _draw_line(image: np.ndarray, p0: np.ndarray, p1: np.ndarray, color: np.ndarray) -> None:
    h, w = image.shape[:2]
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        return
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    n = min(n, 4 * (h + w))
    ts = np.linspace(0.0, 1.0, max(n, 2))
    xs = np.rint(p0[0] + ts * (p1[0] - p0[0])).astype(int)
    ys = np.rint(p0[1] + ts * (p1[1] - p0[1])).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    image[ys[ok], xs[ok]] = color


def render_observation(observation, colors: np.ndarray, render_cfg: dict[str, Any]) -> np.ndarray:
    """A small painter's-algorithm splat render of one observation.

    Deliberately simple: enough to illustrate the matched appearance in figures
    and to carry a photometric channel, not a physically based renderer.  The
    optics that matter are all in the landmark geometry.
    """
    out_w = int(render_cfg["width"])
    out_h = int(render_cfg["height"])
    radius = float(render_cfg.get("point_radius_px", 2.2))
    f = observation.feature
    intr = f.camera.intrinsics
    sx = out_w / float(intr.width)
    sy = out_h / float(intr.height)

    yy = np.linspace(0.0, 1.0, out_h)[:, None]
    image = np.zeros((out_h, out_w, 3), dtype=np.float64) + (0.10 + 0.08 * yy)[..., None]

    # interface frame
    fr = np.nonzero(f.channel == CHANNEL_FRAME)[0]
    pts = np.stack([f.uv[fr, 0] * sx, f.uv[fr, 1] * sy], axis=1)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        if f.visible[fr[a]] or f.visible[fr[b]]:
            _draw_line(image, pts[a], pts[b], np.array([0.55, 0.55, 0.60]))

    # content, far to near
    ci = np.nonzero(f.channel == CHANNEL_CONTENT)[0]
    vis = ci[f.visible[ci]]
    order = vis[np.argsort(-np.nan_to_num(f.depth[vis], nan=-1.0))]
    gy, gx = np.mgrid[0:out_h, 0:out_w]
    for idx in order:
        u, v = f.uv[idx, 0] * sx, f.uv[idx, 1] * sy
        d2 = (gx - u) ** 2 + (gy - v) ** 2
        w = np.exp(-d2 / (2.0 * radius**2))
        c = colors[idx % colors.shape[0]] if colors.size else np.array([0.8, 0.8, 0.8])
        image = image * (1.0 - w[..., None]) + w[..., None] * c[None, None, :]

    # virtual observer markers, in a distinct colour
    mi = np.nonzero((f.channel == CHANNEL_MARKER) & f.visible)[0]
    for idx in mi:
        u, v = f.uv[idx, 0] * sx, f.uv[idx, 1] * sy
        d2 = (gx - u) ** 2 + (gy - v) ** 2
        w = np.exp(-d2 / (2.0 * (radius * 1.4) ** 2))
        image = image * (1.0 - w[..., None]) + w[..., None] * np.array([1.0, 0.25, 0.15])[None, None, :]

    return np.clip(image, 0.0, 1.0)


# ------------------------------------------------------------------ generation
def generate_dataset(config: dict[str, Any], *, output_root: Path | str | None = None, log_every: int = 5) -> dict[str, Any]:
    """Generate the full dataset described by ``config`` and write it to disk."""
    started = time.time()
    ds_cfg = config["dataset"]
    name = str(ds_cfg["name"])
    seed = int(ds_cfg["seed"])
    n_base = int(ds_cfg["num_base_scenes"])

    root = Path(output_root or config["output"]["root"]) / name
    scenes_dir = root / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = root / "previews"

    weights = DistanceWeights.from_dict(config["identifiability"]["distance"])
    epsilon = float(config["identifiability"]["epsilon_px"])
    estimator = GeometrySeparabilityEstimator(AnalyticalTransitionModel(), weights)
    ident = EpsilonIdentifiabilityEstimator(epsilon=epsilon)

    actions = action_space_from_config(config["action_space"])
    baselines, lateral_space = mcrb_baseline_sweep(config["mcrb"])
    render_cfg = config["render"]
    preview_limit = int(render_cfg.get("preview_limit", 6))

    records: list[dict[str, Any]] = []
    base_ids: list[str] = []

    LOGGER.info(
        "generating %d base scenes x %d mechanisms = %d variants (|A|=%d, epsilon=%.3g px)",
        n_base, len(config["mechanisms"]), n_base * len(config["mechanisms"]), len(actions), epsilon,
    )

    for b in range(n_base):
        rng = np.random.default_rng([seed, b])
        base: BaseScene = generate_base_scene(rng, config, b)
        base_ids.append(base.base_scene_id)
        hypotheses = build_hypothesis_set(base.content.interface, config, np.random.default_rng([seed, b, 7]))

        for h_idx, hypothesis in enumerate(hypotheses):
            if hypothesis.mechanism.value not in config["mechanisms"]:
                continue
            scene_id = f"{base.base_scene_id}_{hypothesis.mechanism.value}"
            reference = reference_observation(base.content, hypothesis)
            gt = compute_ground_truth(
                reference, hypotheses, h_idx, actions, estimator, ident,
                markers_cam=base.content.observer_markers_cam,
                baselines=baselines, lateral_space=lateral_space,
            )

            obs_uv, obs_depth, obs_vis, obs_contact = [], [], [], []
            for action in actions:
                obs = simulate(base.content, hypothesis, action, reference_feature=reference.feature)
                obs_uv.append(obs.feature.uv)
                obs_depth.append(obs.feature.depth)
                obs_vis.append(obs.feature.visible)
                obs_contact.append(obs.contact_depth)

            npz_path = scenes_dir / f"{scene_id}.npz"
            np.savez_compressed(
                npz_path,
                content_points=base.content.points,
                content_colors=base.content.colors,
                markers_cam=base.content.observer_markers_cam,
                ref_uv=reference.feature.uv,
                ref_depth=reference.feature.depth,
                ref_visible=reference.feature.visible,
                ref_channel=reference.feature.channel,
                ref_contact_depth=reference.contact_depth,
                obs_uv=np.stack(obs_uv),
                obs_depth=np.stack(obs_depth),
                obs_visible=np.stack(obs_vis),
                obs_contact_depth=np.stack(obs_contact),
                separability=gt.separability_by_action,
                identifiability=gt.identifiability_matrix,
                oracle_utility=gt.oracle_action_utility,
                lateral_separability=gt.lateral_separability,
                lateral_residual=gt.lateral_residual,
                baselines=gt.baselines,
                T_wc_ref=reference.feature.camera.T_wc,
            )

            if render_cfg.get("enabled", True) and len(records) < preview_limit:
                previews_dir.mkdir(parents=True, exist_ok=True)
                _save_preview(previews_dir / f"{scene_id}_ref.png", render_observation(reference, base.content.colors, render_cfg))

            record: dict[str, Any] = {
                "scene_id": scene_id,
                "base_scene_id": base.base_scene_id,
                "hypothesis": hypothesis.mechanism.value,
                "hypothesis_name": hypothesis.name,
                "hypothesis_index": h_idx,
                "hypothesis_set": [h.name for h in hypotheses],
                "hypothesis_mechanisms": [h.mechanism.value for h in hypotheses],
                "hypothesis_params": hypothesis.to_dict()["params"],
                # Full serialisation so the competing family is reconstructed
                # exactly -- interface pose, slab parameters, display mode and all.
                "hypothesis_set_full": [h.to_dict() for h in hypotheses],
                "interface": base.content.interface.to_dict(),
                "camera": reference.feature.camera.to_dict(),
                "n_content_landmarks": base.content.n_content,
                "n_markers": base.content.n_markers,
                "n_landmarks": base.content.n_landmarks,
                "action_set_size": len(actions),
                "arrays": f"scenes/{scene_id}.npz",
                **base.to_dict(),
                **gt.to_metadata(hypotheses),
            }
            records.append(record)

        if log_every and (b + 1) % log_every == 0:
            LOGGER.info("  generated %d/%d base scenes", b + 1, n_base)

    splits = build_splits(base_ids, config["splits"], salt=name)
    for record in records:
        record["split"] = splits["assignment"][record["base_scene_id"]]

    dump_jsonl(root / "scenes.jsonl", records)
    dump_json(root / "splits.json", splits)

    manifest = _build_manifest(config, root, records, actions, splits, epsilon, weights, started)
    dump_json(root / "manifest.json", manifest)

    LOGGER.info(
        "wrote %d scene variants to %s (%.1f%% resolvable) in %.1f s",
        len(records), root, 100.0 * manifest["statistics"]["resolvable_fraction"], time.time() - started,
    )
    return manifest


def _save_preview(path: Path, image: np.ndarray) -> None:
    """Write a PNG without adding a Pillow dependency (matplotlib is already required)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.image as mpimg

    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, np.clip(image, 0.0, 1.0))


def _build_manifest(
    config, root: Path, records, actions, splits, epsilon: float, weights: DistanceWeights, started: float
) -> dict[str, Any]:
    resolvable = np.array([bool(r["resolvable"]) for r in records])
    mcrb = np.array([r["mcrb"] if r["mcrb"] is not None else np.nan for r in records], dtype=np.float64)
    by_mech: dict[str, Any] = {}
    for mech in sorted({r["hypothesis"] for r in records}):
        m = np.array([r["hypothesis"] == mech for r in records])
        by_mech[mech] = {
            "count": int(np.count_nonzero(m)),
            "resolvable_fraction": float(np.mean(resolvable[m])) if np.any(m) else float("nan"),
        }

    checksums = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            checksums[str(path.relative_to(root))] = sha256_file(path)

    return {
        "format_version": DATASET_FORMAT_VERSION,
        "name": config["dataset"]["name"],
        "dataset_version": config["dataset"].get("version", "0.1.0"),
        "seed": int(config["dataset"]["seed"]),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_seconds": round(time.time() - started, 3),
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "config_hash": config_hash(config),
        "config_hash_full": full_config_hash(config),
        "action_space": actions.to_dict(),
        "identifiability": {"epsilon_px": epsilon, "distance": weights.to_dict()},
        "splits": {k: v for k, v in splits.items() if k != "assignment"},
        "statistics": {
            "n_scene_variants": len(records),
            "n_base_scenes": len({r["base_scene_id"] for r in records}),
            "resolvable_count": int(np.count_nonzero(resolvable)),
            "non_resolvable_count": int(np.count_nonzero(~resolvable)),
            "resolvable_fraction": float(np.mean(resolvable)) if len(records) else float("nan"),
            "mcrb_defined_fraction": float(np.mean(np.isfinite(mcrb))) if len(records) else float("nan"),
            "mcrb_median": float(np.nanmedian(mcrb)) if np.any(np.isfinite(mcrb)) else float("nan"),
            "by_mechanism": by_mech,
        },
        "files": checksums,
    }
