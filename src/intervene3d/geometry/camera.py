"""Pinhole camera model.

Conventions are documented in ``intervene3d.geometry.__init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.geometry.se3 import invert, se3_from_Rt, transform_points


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, plus the image size that bounds visibility."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )


@dataclass(frozen=True)
class Camera:
    """A posed pinhole camera.

    ``T_wc`` is the camera-to-world transform; the camera centre is ``T_wc[:3, 3]``.
    """

    intrinsics: CameraIntrinsics
    T_wc: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        T = np.asarray(self.T_wc, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T_wc must be 4x4, got {T.shape}")
        object.__setattr__(self, "T_wc", T)

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return self.T_wc[:3, 3].copy()

    @property
    def R_wc(self) -> np.ndarray:
        return self.T_wc[:3, :3].copy()

    @property
    def T_cw(self) -> np.ndarray:
        """World-to-camera transform."""
        return invert(self.T_wc)

    @property
    def forward(self) -> np.ndarray:
        """Unit optical axis (camera ``+z``) expressed in world coordinates."""
        return self.T_wc[:3, 2].copy()

    def moved(self, delta_T: np.ndarray) -> Camera:
        """Apply an action expressed in this camera's own frame.

        ``T_wc_new = T_wc @ delta_T``.  This is the single place the intervention
        convention is realised; everything else calls through here.
        """
        return Camera(self.intrinsics, self.T_wc @ np.asarray(delta_T, dtype=np.float64))

    def with_pose(self, T_wc: np.ndarray) -> Camera:
        return Camera(self.intrinsics, np.asarray(T_wc, dtype=np.float64))

    def world_to_camera(self, points_w: np.ndarray) -> np.ndarray:
        return transform_points(self.T_cw, points_w)

    def camera_to_world(self, points_c: np.ndarray) -> np.ndarray:
        return transform_points(self.T_wc, points_c)

    def project(self, points_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points.

        Returns ``(uv, depth, in_view)`` where ``uv`` is ``(N, 2)`` pixel
        coordinates (NaN behind the camera), ``depth`` is the camera-frame ``z``
        (NaN behind the camera) and ``in_view`` is the boolean visibility mask
        combining ``z > 0`` with the image bounds.
        """
        return project_points(self.intrinsics, self.world_to_camera(points_w))

    def ray_directions(self, points_w: np.ndarray) -> np.ndarray:
        """Unit world-frame directions from the camera centre towards each point."""
        points_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
        d = points_w - self.center[None, :]
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return d / norms

    def to_dict(self) -> dict[str, Any]:
        return {"intrinsics": self.intrinsics.to_dict(), "T_wc": self.T_wc.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Camera:
        return cls(
            CameraIntrinsics.from_dict(payload["intrinsics"]),
            np.asarray(payload["T_wc"], dtype=np.float64),
        )


def project_points(
    intr: CameraIntrinsics, points_c: np.ndarray, *, min_depth: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points onto the image plane.

    Points at or behind the pinhole (``z <= min_depth``) yield NaN coordinates
    and ``in_view = False`` rather than raising, because hypothesis-conditioned
    prediction routinely evaluates counterfactual geometry that falls behind the
    camera.
    """
    points_c = np.asarray(points_c, dtype=np.float64).reshape(-1, 3)
    z = points_c[:, 2]
    valid = z > min_depth
    safe_z = np.where(valid, z, 1.0)
    u = intr.fx * points_c[:, 0] / safe_z + intr.cx
    v = intr.fy * points_c[:, 1] / safe_z + intr.cy
    uv = np.stack([u, v], axis=1)
    uv[~valid] = np.nan
    depth = np.where(valid, z, np.nan)
    in_view = (
        valid
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= intr.width - 1.0)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= intr.height - 1.0)
    )
    return uv, depth, in_view


def backproject(intr: CameraIntrinsics, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Inverse of :func:`project_points` -- lift pixels to camera-frame points."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    x = (uv[:, 0] - intr.cx) * depth / intr.fx
    y = (uv[:, 1] - intr.cy) * depth / intr.fy
    return np.stack([x, y, depth], axis=1)


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Build ``T_wc`` for a camera at ``eye`` looking at ``target``.

    ``up`` is the world direction that should appear "up" in the image; the
    default ``(0, 0, 1)`` matches a z-up world with an OpenCV y-down camera.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        raise ValueError("look_at: eye and target coincide")
    forward = forward / norm

    if abs(float(np.dot(forward, up))) > 1.0 - 1e-8:
        up = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # camera +y points "down" in the image
    R_wc = np.stack([right, down, forward], axis=1)
    return se3_from_Rt(R_wc, eye)
