"""Reading a generated Intervene3D synthetic dataset."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.data.types import GeometryFeature, Observation
from intervene3d.geometry.camera import Camera
from intervene3d.geometry.planes import Aperture
from intervene3d.hypotheses.base import Hypothesis, HypothesisSet
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.utils.io import load_json, load_jsonl


@dataclass
class Scene:
    """One scene variant: metadata plus lazily loaded arrays."""

    record: dict[str, Any]
    root: Path

    @property
    def scene_id(self) -> str:
        return str(self.record["scene_id"])

    @property
    def base_scene_id(self) -> str:
        return str(self.record["base_scene_id"])

    @property
    def mechanism(self) -> str:
        return str(self.record["hypothesis"])

    @property
    def split(self) -> str:
        return str(self.record.get("split", "all"))

    @property
    def resolvable(self) -> bool:
        return bool(self.record["resolvable"])

    @property
    def true_index(self) -> int:
        return int(self.record["hypothesis_index"])

    @property
    def camera(self) -> Camera:
        return Camera.from_dict(self.record["camera"])

    @property
    def interface(self) -> Aperture:
        return Aperture.from_dict(self.record["interface"])

    def arrays(self) -> dict[str, np.ndarray]:
        return _load_npz(str(self.root / self.record["arrays"]))

    @property
    def markers_cam(self) -> np.ndarray:
        return self.arrays()["markers_cam"]

    @property
    def content_colors(self) -> np.ndarray:
        return self.arrays()["content_colors"]

    def reference_observation(self) -> Observation:
        a = self.arrays()
        feature = GeometryFeature(a["ref_uv"], a["ref_depth"], a["ref_visible"], a["ref_channel"], self.camera)
        return Observation(feature, a["ref_contact_depth"], self.mechanism)

    def observation_for_action(self, action_index: int) -> Observation:
        """The pre-simulated observation obtained by executing action ``action_index``."""
        a = self.arrays()
        cam_ref = self.camera
        feature = GeometryFeature(
            a["obs_uv"][action_index],
            a["obs_depth"][action_index],
            a["obs_visible"][action_index],
            a["ref_channel"],
            cam_ref,  # placeholder; replaced by the caller who knows the action
        )
        return Observation(feature, a["obs_contact_depth"][action_index], self.mechanism)

    def observation_with_camera(self, action_index: int, camera: Camera) -> Observation:
        a = self.arrays()
        feature = GeometryFeature(
            a["obs_uv"][action_index], a["obs_depth"][action_index], a["obs_visible"][action_index],
            a["ref_channel"], camera,
        )
        return Observation(feature, a["obs_contact_depth"][action_index], self.mechanism)

    def hypothesis_set(self) -> HypothesisSet:
        """Reconstruct the competing hypothesis family exactly as it was generated."""
        full = self.record.get("hypothesis_set_full")
        if not full:
            raise KeyError(
                f"scene {self.scene_id} predates hypothesis_set_full; regenerate the dataset with "
                "python scripts/generate_synthetic_data.py"
            )
        return HypothesisSet(tuple(Hypothesis.from_dict(h) for h in full))

    @property
    def display_mode(self) -> str | None:
        for h in self.record.get("hypothesis_set_full", []):
            if h["mechanism"] == "emissive":
                return h["params"].get("display_mode")
        return None

    def oracle_utility(self) -> np.ndarray:
        return self.arrays()["oracle_utility"]

    def oracle_separability(self) -> np.ndarray:
        return self.arrays()["separability"]


@lru_cache(maxsize=256)
def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


class SyntheticDataset:
    """A generated Intervene3D benchmark on disk."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no dataset manifest at {manifest_path}. Generate one with:\n"
                f"  python scripts/generate_synthetic_data.py --config configs/synthetic/smoke.yaml"
            )
        self.manifest = load_json(manifest_path)
        self.records = load_jsonl(self.root / "scenes.jsonl")
        self.splits = load_json(self.root / "splits.json")

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Scene]:
        return (Scene(r, self.root) for r in self.records)

    @property
    def name(self) -> str:
        return str(self.manifest["name"])

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace.from_dict(self.manifest["action_space"])

    @property
    def epsilon(self) -> float:
        return float(self.manifest["identifiability"]["epsilon_px"])

    @property
    def distance_weights(self) -> dict[str, float]:
        return dict(self.manifest["identifiability"]["distance"])

    def scenes(self, split: str = "all") -> list[Scene]:
        if split == "all":
            return [Scene(r, self.root) for r in self.records]
        return [Scene(r, self.root) for r in self.records if r.get("split") == split]

    def manifest_summary(self) -> dict[str, Any]:
        """The subset of the manifest archived inside a run directory."""
        return {
            "name": self.name,
            "dataset_version": self.manifest.get("dataset_version"),
            "format_version": self.manifest.get("format_version"),
            "seed": self.manifest.get("seed"),
            "config_hash": self.manifest.get("config_hash"),
            "generated_utc": self.manifest.get("generated_utc"),
            "statistics": self.manifest.get("statistics"),
            "identifiability": self.manifest.get("identifiability"),
            "splits": self.manifest.get("splits"),
            "root": str(self.root),
            "n_files": len(self.manifest.get("files", {})),
        }
