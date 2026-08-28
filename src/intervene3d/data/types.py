"""Shared data types for observations, geometry features and scenes.

Landmark layout
---------------
Every observation and every predicted transition uses **one** landmark array
with a fixed, hypothesis-independent layout so that index-wise comparison
between a prediction and an observation is always valid:

===================  ==========  ==========================================
slice                channel     meaning
===================  ==========  ==========================================
``[0 : N]``          ``CONTENT``  the apparent scene content
``[N : N+4]``        ``FRAME``    the four corners of the optical interface
``[N+4 : N+4+K]``    ``MARKER``   virtual images of observer-attached markers
===================  ==========  ==========================================

The ``MARKER`` channel is what makes a planar mirror geometrically identifiable
at all.  A static planar mirror reflecting a static scene produces a *static
virtual scene*, which is indistinguishable from a real scene seen through an
opening of the same shape.  What is **not** static is the virtual image of
anything rigidly attached to the observer: it moves whenever the observer moves,
at twice the rate along the mirror normal.  Whether any allowed action brings
that virtual marker inside the interface aperture *and* inside the image is
precisely an action-set-dependent identifiability question -- which is the point
of this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.geometry.camera import Camera
from intervene3d.geometry.planes import Aperture

CHANNEL_CONTENT = 0
CHANNEL_FRAME = 1
CHANNEL_MARKER = 2
CHANNEL_NAMES = {CHANNEL_CONTENT: "content", CHANNEL_FRAME: "frame", CHANNEL_MARKER: "marker"}


@dataclass(frozen=True)
class GeometryFeature:
    """``F_t`` -- the geometry-space representation the model reasons about.

    This is deliberately *not* raw RGB: the research specification argues the
    world model should forecast geometry-foundation-model features rather than
    photometric detail.  In the preliminary codebase the "features" are posed
    landmark projections plus their perceived depth, which keeps every quantity
    interpretable and every transition analytically checkable.

    Attributes
    ----------
    uv:
        ``(M, 2)`` pixel coordinates; NaN where the landmark is not visible.
    depth:
        ``(M,)`` *perceived* (apparent) depth in metres; NaN where not visible.
        This is the depth a geometry encoder reports -- for a display it is the
        hallucinated content depth, not the screen depth.
    visible:
        ``(M,)`` boolean visibility, combining frustum bounds with occlusion by
        the optical interface aperture.
    channel:
        ``(M,)`` one of ``CHANNEL_CONTENT`` / ``CHANNEL_FRAME`` / ``CHANNEL_MARKER``.
    camera:
        The camera this feature was produced from.
    """

    uv: np.ndarray
    depth: np.ndarray
    visible: np.ndarray
    channel: np.ndarray
    camera: Camera

    def __post_init__(self) -> None:
        uv = np.asarray(self.uv, dtype=np.float64).reshape(-1, 2)
        m = uv.shape[0]
        depth = np.asarray(self.depth, dtype=np.float64).reshape(m)
        visible = np.asarray(self.visible, dtype=bool).reshape(m)
        channel = np.asarray(self.channel, dtype=np.int8).reshape(m)
        # Invisible landmarks must not leak coordinates into any distance.
        uv = uv.copy()
        depth = depth.copy()
        uv[~visible] = np.nan
        depth[~visible] = np.nan
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "channel", channel)

    def __len__(self) -> int:
        return int(self.uv.shape[0])

    def mask(self, channel: int) -> np.ndarray:
        return self.channel == channel

    def select(self, channel: int) -> GeometryFeature:
        m = self.mask(channel)
        return GeometryFeature(self.uv[m], self.depth[m], self.visible[m], self.channel[m], self.camera)

    @property
    def n_visible(self) -> int:
        return int(np.count_nonzero(self.visible))

    def points_camera(self) -> np.ndarray:
        """Back-project visible landmarks into camera-frame 3D points (NaN elsewhere)."""
        from intervene3d.geometry.camera import backproject

        pts = np.full((len(self), 3), np.nan)
        m = self.visible & np.isfinite(self.depth)
        if np.any(m):
            pts[m] = backproject(self.camera.intrinsics, self.uv[m], self.depth[m])
        return pts

    def to_dict(self) -> dict[str, Any]:
        return {
            "uv": self.uv.tolist(),
            "depth": self.depth.tolist(),
            "visible": self.visible.tolist(),
            "channel": self.channel.tolist(),
            "camera": self.camera.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GeometryFeature:
        return cls(
            np.asarray(payload["uv"], dtype=np.float64),
            np.asarray(payload["depth"], dtype=np.float64),
            np.asarray(payload["visible"], dtype=bool),
            np.asarray(payload["channel"], dtype=np.int8),
            Camera.from_dict(payload["camera"]),
        )


@dataclass(frozen=True)
class Observation:
    """A rendered observation plus the ground truth an oracle has access to.

    ``feature`` is the only part an inference method may read.  ``contact_depth``
    and ``mechanism`` are supervision / evaluation signals.
    """

    feature: GeometryFeature
    contact_depth: np.ndarray  # (M,) depth of the first physical surface along each landmark ray
    mechanism: str
    image: np.ndarray | None = None  # (H, W, 3) float in [0, 1], optional

    def __post_init__(self) -> None:
        cd = np.asarray(self.contact_depth, dtype=np.float64).reshape(len(self.feature))
        object.__setattr__(self, "contact_depth", cd)

    @property
    def camera(self) -> Camera:
        return self.feature.camera


@dataclass(frozen=True)
class SceneContent:
    """The *apparent* scene content, anchored at the reference view.

    ``points`` are world positions of the apparent geometry as a direct
    interpretation would place them.  Every causal variant of a base scene shares
    the same ``SceneContent``; the variants differ only in which optical
    mechanism produces that appearance.  This is what makes the counterfactuals
    matched by construction rather than by optimisation.
    """

    points: np.ndarray  # (N, 3) world
    colors: np.ndarray  # (N, 3) in [0, 1]
    reference_camera: Camera
    interface: Aperture
    observer_markers_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        cols = np.asarray(self.colors, dtype=np.float64).reshape(-1, 3)
        markers = np.asarray(self.observer_markers_cam, dtype=np.float64).reshape(-1, 3)
        if cols.shape[0] != pts.shape[0]:
            raise ValueError("colors and points must have the same length")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "colors", cols)
        object.__setattr__(self, "observer_markers_cam", markers)

    @property
    def n_content(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_markers(self) -> int:
        return int(self.observer_markers_cam.shape[0])

    @property
    def n_landmarks(self) -> int:
        return self.n_content + 4 + self.n_markers

    def channel_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.full(self.n_content, CHANNEL_CONTENT, dtype=np.int8),
                np.full(4, CHANNEL_FRAME, dtype=np.int8),
                np.full(self.n_markers, CHANNEL_MARKER, dtype=np.int8),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points.tolist(),
            "colors": self.colors.tolist(),
            "reference_camera": self.reference_camera.to_dict(),
            "interface": self.interface.to_dict(),
            "observer_markers_cam": self.observer_markers_cam.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SceneContent:
        return cls(
            np.asarray(payload["points"], dtype=np.float64),
            np.asarray(payload["colors"], dtype=np.float64),
            Camera.from_dict(payload["reference_camera"]),
            Aperture.from_dict(payload["interface"]),
            np.asarray(payload.get("observer_markers_cam", np.zeros((0, 3))), dtype=np.float64),
        )
