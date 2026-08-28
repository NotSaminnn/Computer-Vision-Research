"""Geometry encoders -- the ``F_t = E(I_t)`` adapter layer.

The repository is *not* hard-coded to any external foundation model.  Encoders
are built through :func:`build_geometry_encoder`, and the preliminary pipeline
runs end to end on the ``ground_truth`` / ``mock`` encoders alone.

Available names
---------------
``ground_truth``
    Returns the simulator's landmark feature unchanged.  This models a *perfect*
    geometry foundation model that is nevertheless fooled by the optical
    illusion in exactly the way the NeurIPS 2025 "3D Visual Illusion Depth
    Estimation" results document: it reports the *apparent* content depth, which
    is identical across every matched causal variant at the reference view.
``mock``
    ``ground_truth`` plus configurable pixel / depth noise and random landmark
    dropout, for robustness studies.
``moge`` / ``vggt_like``
    NOT IMPLEMENTED.  These raise with installation instructions rather than
    silently degrading, so a run can never quietly claim to have used a
    foundation encoder it did not use.  See ``docs/DEVELOPMENT_ROADMAP.md``
    (Gate 6) for the integration checklist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.data.types import GeometryFeature, Observation


@dataclass
class GroundTruthEncoder:
    """Noiseless oracle encoder."""

    name: str = "ground_truth"

    def encode(self, observation: Observation) -> GeometryFeature:
        return observation.feature

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass
class MockEncoder:
    """Oracle encoder plus controlled measurement noise.

    ``pixel_noise_std`` is in pixels, ``depth_noise_rel`` is a relative depth
    error, and ``dropout`` is the probability that a visible landmark is missed.
    All noise is drawn from an explicitly seeded generator so encoding is
    reproducible.
    """

    pixel_noise_std: float = 0.0
    depth_noise_rel: float = 0.0
    dropout: float = 0.0
    seed: int = 0
    name: str = "mock"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def encode(self, observation: Observation) -> GeometryFeature:
        f = observation.feature
        uv = f.uv.copy()
        depth = f.depth.copy()
        visible = f.visible.copy()

        if self.pixel_noise_std > 0.0:
            uv = uv + self._rng.normal(0.0, self.pixel_noise_std, size=uv.shape)
        if self.depth_noise_rel > 0.0:
            depth = depth * (1.0 + self._rng.normal(0.0, self.depth_noise_rel, size=depth.shape))
        if self.dropout > 0.0:
            drop = self._rng.random(visible.shape) < self.dropout
            visible = visible & ~drop
        return GeometryFeature(uv, depth, visible, f.channel, f.camera)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pixel_noise_std": self.pixel_noise_std,
            "depth_noise_rel": self.depth_noise_rel,
            "dropout": self.dropout,
            "seed": self.seed,
        }


class _UnavailableEncoder:
    """Registered but deliberately non-functional foundation-model adapters."""

    def __init__(self, name: str, instructions: str) -> None:
        self.name = name
        self._instructions = instructions

    def encode(self, observation: Observation) -> GeometryFeature:  # pragma: no cover
        raise NotImplementedError(
            f"geometry encoder {self.name!r} is NOT IMPLEMENTED in the preliminary "
            f"codebase.\n{self._instructions}"
        )


_MOGE_INSTRUCTIONS = """\
To integrate MoGe-2 (Wang et al., arXiv:2507.02546):
  1. verify the current official repository and its licence;
  2. install its dependencies in the optional GPU environment
     (pip install -r requirements-gpu.txt, then the model's own requirements);
  3. download the published checkpoint and record its identifier in the run config;
  4. implement encode() so that it returns a GeometryFeature with the SAME
     landmark layout as the synthetic simulator (see intervene3d.data.types).
Until all four steps are done the preliminary pipeline must be run with
encoder=ground_truth or encoder=mock."""

_VGGT_INSTRUCTIONS = """\
To integrate a VGGT-style geometry representation (cf. VGGT-World,
arXiv:2603.12655, which forecasts frozen geometry-foundation-model features):
  1. verify the official repository, licence and checkpoint availability;
  2. confirm Python / PyTorch compatibility and GPU memory requirements;
  3. implement encode() returning the shared GeometryFeature layout.
Until then use encoder=ground_truth or encoder=mock."""


def build_geometry_encoder(config: dict[str, Any] | None):
    """Instantiate an encoder from configuration.

    ``config`` example::

        {"name": "mock", "pixel_noise_std": 0.25, "seed": 7}
    """
    config = dict(config or {})
    name = str(config.pop("name", "ground_truth"))
    if name == "ground_truth":
        return GroundTruthEncoder()
    if name == "mock":
        return MockEncoder(
            pixel_noise_std=float(config.get("pixel_noise_std", 0.0)),
            depth_noise_rel=float(config.get("depth_noise_rel", 0.0)),
            dropout=float(config.get("dropout", 0.0)),
            seed=int(config.get("seed", 0)),
        )
    if name == "moge":
        return _UnavailableEncoder("moge", _MOGE_INSTRUCTIONS)
    if name == "vggt_like":
        return _UnavailableEncoder("vggt_like", _VGGT_INSTRUCTIONS)
    raise ValueError(
        f"unknown geometry encoder {name!r}; available: ground_truth, mock, moge (NOT IMPLEMENTED), "
        "vggt_like (NOT IMPLEMENTED)"
    )


AVAILABLE_ENCODERS = ("ground_truth", "mock", "moge", "vggt_like")
IMPLEMENTED_ENCODERS = ("ground_truth", "mock")
