"""Configuration validation.

Validation is intentionally explicit rather than schema-library-driven: the
error messages name the offending key and say what was expected, which is what
matters when a run fails three hours in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from intervene3d.config.loader import ConfigError, deep_merge

VALID_MECHANISMS = ("direct", "reflection", "transmission", "emissive", "mixed")
VALID_ENCODERS = ("ground_truth", "mock", "moge", "vggt_like")
VALID_TRANSITIONS = ("analytical", "hybrid", "learned_only", "no_hypothesis_conditioning")
VALID_SELECTORS = ("max_separability", "entropy_nbv", "random", "max_baseline", "fixed", "null")

SYNTHETIC_DEFAULTS: dict[str, Any] = {
    "dataset": {"name": "intervene3d_synth", "version": "0.1.0", "seed": 12345, "num_base_scenes": 8},
    "mechanisms": ["direct", "emissive", "reflection"],
    "camera": {"fx": 320.0, "fy": 320.0, "cx": 160.0, "cy": 120.0, "width": 320, "height": 240},
    "scene": {
        "interface_distance": [1.2, 2.5],
        # Aperture half-width as a fraction of the image half-width projected onto
        # the interface plane.  1.0 means the interface exactly fills the frame;
        # values above 1.0 give apertures wider than the field of view.
        "aperture_half_width_frac": [0.45, 1.15],
        "aperture_aspect": [0.60, 0.95],
        "content_fill": 0.85,
        "num_content_landmarks": 40,
        # Offset of the nearest content behind the interface, and the depth
        # *spread* of the content.  The spread is the difficulty knob: MCRB scales
        # as 1 / |1/Z1 - 1/Z2|, so a shallow scene needs a large baseline and a
        # sufficiently shallow one is unresolvable within the action bounds.
        "content_depth_near": [0.02, 0.80],
        "content_depth_spread": [0.02, 2.50],
        "interface_tilt_deg": 8.0,
        "camera_jitter_translation": 0.30,
        "camera_jitter_rotation_deg": 6.0,
    },
    # Structure rigidly attached to the observer.  Its virtual image in a mirror
    # moves when the observer moves; nothing else in a static scene does.  The
    # whole rig is offset as a unit so that some scenes place every marker's
    # virtual image outside the aperture -- those are the mirror cases no allowed
    # action can resolve.
    "observer_markers": {
        "count": 3,
        # Tuned so the virtual image of the observer rig lands near the aperture
        # boundary: some scenes need a large baseline to bring it into view, some
        # cannot be resolved by any allowed action.  Wider ranges push every
        # mirror permanently out of frame and inflate the non-identifiable
        # fraction for an uninteresting reason.
        "rig_offset_lateral": [-1.8, 1.8],
        "rig_offset_vertical": [-0.8, 0.8],
        "jitter": 0.12,
        "forward_offset": [-0.10, 0.02],
    },
    # A view-tracked (perspective-correct) display re-renders for the current
    # observer pose, so inside its aperture it is geometrically IDENTICAL to a
    # direct scene: no baseline resolves it.  Including such cases is deliberate
    # -- a benchmark in which every ambiguity resolves would undermine the
    # research question.
    "display": {"mode": "static", "view_tracked_probability": 0.25},
    "glass": {"thickness": [0.004, 0.030], "refractive_index": [1.45, 1.60]},
    "action_space": {
        "translation_steps": [0.05, 0.10, 0.20, 0.30],
        "rotation_steps_deg": [5.0, 10.0],
        "enabled_kinds": ["translate_x", "translate_y", "translate_z", "yaw", "pitch"],
        "max_translation": 0.35,
        "max_rotation_deg": 15.0,
        "include_null": True,
        "symmetric": True,
    },
    "identifiability": {
        "epsilon_px": 1.0,
        "distance": {
            "lambda_motion": 1.0,
            "lambda_occlusion": 4.0,
            "lambda_geometry": 0.0,
            "lambda_feature": 0.0,
        },
    },
    "mcrb": {"num_baseline_samples": 41, "max_baseline": 0.35},
    "render": {"enabled": True, "width": 160, "height": 120, "point_radius_px": 2.2},
    "splits": {"train": 0.5, "val": 0.2, "test": 0.3, "policy": "base_scene"},
    "output": {"root": "data/processed"},
}

EXPERIMENT_DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "unnamed_experiment", "seed": 42, "root": "experiments"},
    "data": {"dataset_dir": None, "synthetic_config": None, "split": "test"},
    "model": {
        "geometry_encoder": {"name": "ground_truth"},
        "transition": {"name": "analytical"},
        "belief": {"beta": 1.0, "floor": 1.0e-6},
        "identifiability": {"epsilon_px": 1.0},
        "abstention": {"enabled": True, "tau": 0.8},
        "distance": {
            "lambda_motion": 1.0,
            "lambda_occlusion": 4.0,
            "lambda_geometry": 0.0,
            "lambda_feature": 0.0,
        },
    },
    "methods": [],
    "action_noise": {"enabled": False, "translation_std": 0.0, "rotation_std_deg": 0.0},
    "metrics": [
        "causal_explanation_accuracy",
        "contact_depth_error",
        "identifiability_auroc",
        "mcrb_error",
        "fpcr",
        "intervention_regret",
    ],
    "visualization": {"ieee": True, "formats": ["pdf", "png"], "dpi": 400, "column": "single"},
}


def _require(cfg: Mapping[str, Any], key: str, kind: type | tuple[type, ...], path: str = "") -> Any:
    if key not in cfg:
        raise ConfigError(f"missing required configuration key {path + key!r}")
    value = cfg[key]
    if not isinstance(value, kind):
        raise ConfigError(
            f"configuration key {path + key!r} must be {kind}, got {type(value).__name__}"
        )
    return value


def _check_range(value: Any, path: str) -> None:
    if not (isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2):
        raise ConfigError(f"{path} must be a two-element [low, high] range, got {value!r}")
    lo, hi = float(value[0]), float(value[1])
    if hi < lo:
        raise ConfigError(f"{path} has high < low: {value!r}")


def validate_synthetic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults and validate a synthetic-generator configuration."""
    cfg = deep_merge(SYNTHETIC_DEFAULTS, dict(config))

    ds = _require(cfg, "dataset", Mapping)
    if int(ds.get("num_base_scenes", 0)) < 1:
        raise ConfigError("dataset.num_base_scenes must be >= 1")

    mechanisms = _require(cfg, "mechanisms", (list, tuple))
    if len(mechanisms) < 2:
        raise ConfigError("mechanisms must list at least two competing mechanisms")
    for m in mechanisms:
        if m not in VALID_MECHANISMS:
            raise ConfigError(f"unknown mechanism {m!r}; valid: {VALID_MECHANISMS}")
    if len(set(mechanisms)) != len(mechanisms):
        raise ConfigError(f"mechanisms must be unique, got {mechanisms}")

    cam = _require(cfg, "camera", Mapping)
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        if key not in cam:
            raise ConfigError(f"camera.{key} is required")
        if float(cam[key]) <= 0:
            raise ConfigError(f"camera.{key} must be positive")

    scene = _require(cfg, "scene", Mapping)
    for key in (
        "interface_distance",
        "aperture_half_width_frac",
        "aperture_aspect",
        "content_depth_near",
        "content_depth_spread",
    ):
        _check_range(scene[key], f"scene.{key}")
    if not 0.0 < float(scene["content_fill"]) <= 1.0:
        raise ConfigError("scene.content_fill must lie in (0, 1]")
    if int(scene["num_content_landmarks"]) < 4:
        raise ConfigError("scene.num_content_landmarks must be >= 4")

    if "emissive" in mechanisms:
        mode = cfg["display"]["mode"]
        if mode not in ("static", "view_tracked"):
            raise ConfigError(f"display.mode must be 'static' or 'view_tracked', got {mode!r}")
        prob = float(cfg["display"].get("view_tracked_probability", 0.0))
        if not 0.0 <= prob <= 1.0:
            raise ConfigError("display.view_tracked_probability must lie in [0, 1]")

    ident = _require(cfg, "identifiability", Mapping)
    if float(ident["epsilon_px"]) <= 0:
        raise ConfigError("identifiability.epsilon_px must be positive")
    if float(ident["distance"].get("lambda_feature", 0.0)) != 0.0:
        raise ConfigError("identifiability.distance.lambda_feature must be 0 (D_feature is NOT IMPLEMENTED)")

    splits = _require(cfg, "splits", Mapping)
    total = sum(float(splits[k]) for k in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"splits.train+val+test must sum to 1.0, got {total}")
    if splits.get("policy", "base_scene") != "base_scene":
        raise ConfigError(
            "splits.policy must be 'base_scene': matched causal variants of one base scene "
            "must never be separated across splits (see docs/DATASET_MATRIX.md, leakage policy)"
        )
    return cfg


def validate_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults and validate an experiment configuration."""
    cfg = deep_merge(EXPERIMENT_DEFAULTS, dict(config))

    exp = _require(cfg, "experiment", Mapping)
    name = str(exp.get("name", ""))
    if not name or name == "unnamed_experiment":
        raise ConfigError("experiment.name must be set to a descriptive, filesystem-safe name")
    if not all(ch.isalnum() or ch in "_-" for ch in name):
        raise ConfigError(f"experiment.name {name!r} must contain only alphanumerics, '_' and '-'")

    data = _require(cfg, "data", Mapping)
    if not data.get("dataset_dir") and not data.get("synthetic_config"):
        raise ConfigError("data.dataset_dir or data.synthetic_config must be provided")
    if data.get("split") not in ("train", "val", "test", "all"):
        raise ConfigError("data.split must be one of train/val/test/all")

    model = _require(cfg, "model", Mapping)
    enc = model["geometry_encoder"]["name"]
    if enc not in VALID_ENCODERS:
        raise ConfigError(f"unknown geometry_encoder {enc!r}; valid: {VALID_ENCODERS}")
    tr = model["transition"]["name"]
    if tr not in VALID_TRANSITIONS:
        raise ConfigError(f"unknown transition {tr!r}; valid: {VALID_TRANSITIONS}")
    if float(model["distance"].get("lambda_feature", 0.0)) != 0.0:
        raise ConfigError("model.distance.lambda_feature must be 0 (D_feature is NOT IMPLEMENTED)")
    if not 0.0 < float(model["abstention"]["tau"]) <= 1.0:
        raise ConfigError("model.abstention.tau must lie in (0, 1]")

    methods = cfg.get("methods") or []
    if not methods:
        raise ConfigError("experiment must declare at least one entry under 'methods'")
    seen = set()
    for i, m in enumerate(methods):
        if not isinstance(m, Mapping):
            raise ConfigError(f"methods[{i}] must be a mapping")
        if "name" not in m:
            raise ConfigError(f"methods[{i}] is missing 'name'")
        if m["name"] in seen:
            raise ConfigError(f"duplicate method name {m['name']!r}")
        seen.add(m["name"])
        sel = m.get("selector", "max_separability")
        if sel not in VALID_SELECTORS:
            raise ConfigError(f"methods[{i}].selector {sel!r} invalid; valid: {VALID_SELECTORS}")
        if sel == "fixed" and not m.get("fixed_action"):
            raise ConfigError(f"methods[{i}] uses selector 'fixed' but has no 'fixed_action'")
        tname = m.get("transition", tr)
        if tname not in VALID_TRANSITIONS:
            raise ConfigError(f"methods[{i}].transition {tname!r} invalid; valid: {VALID_TRANSITIONS}")
    return cfg
