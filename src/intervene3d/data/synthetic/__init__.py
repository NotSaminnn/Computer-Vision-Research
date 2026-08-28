"""Synthetic Intervene3D benchmark generation."""

from intervene3d.data.synthetic.camera_generator import intrinsics_from_config, reference_camera
from intervene3d.data.synthetic.dataset_writer import (
    generate_dataset,
    render_observation,
    sha256_file,
)
from intervene3d.data.synthetic.ground_truth import SceneGroundTruth, compute_ground_truth
from intervene3d.data.synthetic.optical_variants import (
    build_hypothesis,
    build_hypothesis_set,
    content_depth_range,
    reference_observation,
    simulate,
)
from intervene3d.data.synthetic.scene_generator import BaseScene, generate_base_scene
from intervene3d.data.synthetic.trajectory_generator import (
    action_space_from_config,
    mcrb_baseline_sweep,
)
from intervene3d.data.synthetic.validator import ValidationReport, validate_dataset

__all__ = [
    "intrinsics_from_config",
    "reference_camera",
    "generate_dataset",
    "render_observation",
    "sha256_file",
    "SceneGroundTruth",
    "compute_ground_truth",
    "build_hypothesis",
    "build_hypothesis_set",
    "content_depth_range",
    "reference_observation",
    "simulate",
    "BaseScene",
    "generate_base_scene",
    "action_space_from_config",
    "mcrb_baseline_sweep",
    "ValidationReport",
    "validate_dataset",
]
