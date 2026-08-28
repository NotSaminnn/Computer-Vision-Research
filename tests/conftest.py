"""Shared fixtures.  Keeps every test tiny and CPU-only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intervene3d.config import validate_synthetic_config  # noqa: E402
from intervene3d.data.synthetic import (  # noqa: E402
    action_space_from_config,
    build_hypothesis_set,
    generate_base_scene,
    reference_observation,
)
from intervene3d.models.separability import (  # noqa: E402
    DistanceWeights,
    GeometrySeparabilityEstimator,
)
from intervene3d.models.transition import AnalyticalTransitionModel  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def synth_config() -> dict:
    return validate_synthetic_config(
        {
            "dataset": {"name": "pytest_synth", "seed": 4242, "num_base_scenes": 6},
            "scene": {"num_content_landmarks": 16},
            "camera": {"fx": 200.0, "fy": 200.0, "cx": 100.0, "cy": 75.0, "width": 200, "height": 150},
            "action_space": {
                "translation_steps": [0.05, 0.15, 0.30],
                "rotation_steps_deg": [8.0],
                "enabled_kinds": ["translate_x", "translate_y", "yaw"],
            },
            "mcrb": {"num_baseline_samples": 15, "max_baseline": 0.35},
            "render": {"enabled": False, "width": 80, "height": 60, "preview_limit": 0},
        }
    )


@pytest.fixture
def base_scene(synth_config):
    return generate_base_scene(np.random.default_rng([1, 0]), synth_config, 0)


@pytest.fixture
def hypotheses(base_scene, synth_config):
    return build_hypothesis_set(base_scene.content.interface, synth_config, np.random.default_rng([1, 0, 7]))


@pytest.fixture
def actions(synth_config):
    return action_space_from_config(synth_config["action_space"])


@pytest.fixture
def reference_feature(base_scene, hypotheses):
    return reference_observation(base_scene.content, hypotheses[0]).feature


@pytest.fixture
def estimator(synth_config):
    return GeometrySeparabilityEstimator(
        AnalyticalTransitionModel(), DistanceWeights.from_dict(synth_config["identifiability"]["distance"])
    )
