"""Datasets: shared types, the synthetic benchmark, splits and external adapters."""

from intervene3d.data.dataset import Scene, SyntheticDataset
from intervene3d.data.splits import assign_split, build_splits, check_no_leakage
from intervene3d.data.types import (
    CHANNEL_CONTENT,
    CHANNEL_FRAME,
    CHANNEL_MARKER,
    CHANNEL_NAMES,
    GeometryFeature,
    Observation,
    SceneContent,
)

__all__ = [
    "Scene",
    "SyntheticDataset",
    "assign_split",
    "build_splits",
    "check_no_leakage",
    "CHANNEL_CONTENT",
    "CHANNEL_FRAME",
    "CHANNEL_MARKER",
    "CHANNEL_NAMES",
    "GeometryFeature",
    "Observation",
    "SceneContent",
]
