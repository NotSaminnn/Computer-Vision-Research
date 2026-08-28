"""Planar primitives: infinite planes, finite rectangular apertures, reflection
and the plane-induced homography.

The finite *aperture* is what makes the preliminary benchmark scientifically
non-trivial.  A planar mirror showing a static scene produces a **static virtual
scene**, so the reflected content alone is geometrically indistinguishable from a
direct scene under observer motion.  What breaks the tie is the mirror's finite
frame: the virtual content is only visible through the aperture, and the
aperture is at a different depth than the content, so observer motion produces
aperture-relative parallax and progressive clipping of the content.  The same is
true for a screen bezel and a glass pane's edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Plane:
    """Infinite plane ``{X : n . X + d = 0}`` with a unit normal.

    ``point`` is any point on the plane; ``normal`` is stored normalised.
    """

    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.point, dtype=np.float64).reshape(3)
        n = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError("Plane normal must be non-zero")
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "normal", n / norm)

    @property
    def d(self) -> float:
        """Offset such that ``n . X + d = 0`` on the plane."""
        return float(-np.dot(self.normal, self.point))

    def to_dict(self) -> dict[str, Any]:
        return {"point": self.point.tolist(), "normal": self.normal.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Plane:
        return cls(np.asarray(payload["point"]), np.asarray(payload["normal"]))


def signed_distance(plane: Plane, points: np.ndarray) -> np.ndarray:
    """Signed distance of each point from ``plane`` along its normal."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ plane.normal + plane.d


def ray_plane_intersection(
    origins: np.ndarray, directions: np.ndarray, plane: Plane, *, min_t: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays with a plane.

    Returns ``(points, hit)``.  ``hit`` is False for rays parallel to the plane
    or whose intersection lies behind the origin (``t <= min_t``); the
    corresponding rows of ``points`` are NaN.
    """
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if origins.shape[0] == 1 and directions.shape[0] > 1:
        origins = np.repeat(origins, directions.shape[0], axis=0)

    denom = directions @ plane.normal
    numer = -(origins @ plane.normal + plane.d)
    parallel = np.abs(denom) < 1e-12
    safe_denom = np.where(parallel, 1.0, denom)
    t = numer / safe_denom
    hit = (~parallel) & (t > min_t)
    points = origins + t[:, None] * directions
    points[~hit] = np.nan
    return points, hit


def reflect_plane_matrix(plane: Plane) -> np.ndarray:
    """4x4 householder reflection about ``plane`` (an improper isometry).

    Note this matrix has ``det = -1``; it is *not* in SE(3).  Use
    :func:`reflect_pose` when a valid camera pose is required.
    """
    n = plane.normal
    d = plane.d
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - 2.0 * np.outer(n, n)
    M[:3, 3] = -2.0 * d * n
    return M


def reflect_points(plane: Plane, points: np.ndarray) -> np.ndarray:
    """Mirror points about ``plane``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dist = signed_distance(plane, points)
    return points - 2.0 * dist[:, None] * plane.normal[None, :]


def reflect_pose(plane: Plane, T_wc: np.ndarray) -> np.ndarray:
    """Virtual (reflected) camera pose.

    A raw Householder reflection of the pose would flip handedness, giving a
    determinant of ``-1``.  Physically, viewing a scene in a planar mirror is
    equivalent to viewing it from the mirrored camera *with a mirrored image*.
    We keep the pose a proper rotation by negating the ``x`` axis, which
    corresponds exactly to the left-right image flip a mirror produces.

    ``tests/unit/test_optics.py`` asserts that projecting the *physical* points
    with this virtual camera reproduces (up to that horizontal flip) projecting
    the *virtual* points with the real camera -- two independent derivations of
    the same optics.
    """
    T_wc = np.asarray(T_wc, dtype=np.float64)
    M = reflect_plane_matrix(plane)
    T = M @ T_wc
    R = T[:3, :3].copy()
    R[:, 0] = -R[:, 0]  # restore det(+1); equivalent to the mirror image flip
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T[:3, 3]
    return out


@dataclass(frozen=True)
class Aperture:
    """A finite axis-aligned rectangle lying in ``plane``.

    ``u_axis``/``v_axis`` are orthonormal in-plane directions and
    ``half_extent`` gives the half-width along each.  ``contains`` tests
    membership of 3D points that are assumed to already lie on the plane.
    """

    plane: Plane
    u_axis: np.ndarray
    v_axis: np.ndarray
    half_extent: np.ndarray  # (2,) metres

    def __post_init__(self) -> None:
        u = np.asarray(self.u_axis, dtype=np.float64).reshape(3)
        v = np.asarray(self.v_axis, dtype=np.float64).reshape(3)
        u = u / np.linalg.norm(u)
        v = v - np.dot(v, u) * u
        v = v / np.linalg.norm(v)
        he = np.asarray(self.half_extent, dtype=np.float64).reshape(2)
        if np.any(he <= 0):
            raise ValueError("Aperture half_extent must be positive")
        object.__setattr__(self, "u_axis", u)
        object.__setattr__(self, "v_axis", v)
        object.__setattr__(self, "half_extent", he)

    @classmethod
    def from_plane(cls, plane: Plane, half_width: float, half_height: float) -> Aperture:
        """Build an aperture with in-plane axes derived deterministically from
        the plane normal (so the same plane always yields the same axes)."""
        n = plane.normal
        seed = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(seed, n)
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)
        return cls(plane, u, v, np.array([half_width, half_height]))

    @property
    def area(self) -> float:
        return float(4.0 * self.half_extent[0] * self.half_extent[1])

    def local_coords(self, points: np.ndarray) -> np.ndarray:
        """Project points onto the in-plane ``(u, v)`` coordinate system."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rel = points - self.plane.point[None, :]
        return np.stack([rel @ self.u_axis, rel @ self.v_axis], axis=1)

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: is each (on-plane) point inside the rectangle?"""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        finite = np.all(np.isfinite(points), axis=1)
        local = self.local_coords(np.where(finite[:, None], points, 0.0))
        inside = (np.abs(local[:, 0]) <= self.half_extent[0]) & (
            np.abs(local[:, 1]) <= self.half_extent[1]
        )
        return inside & finite

    def corners(self) -> np.ndarray:
        """The four rectangle corners in world coordinates, counter-clockwise."""
        hu, hv = self.half_extent
        signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        return (
            self.plane.point[None, :]
            + signs[:, 0:1] * hu * self.u_axis[None, :]
            + signs[:, 1:2] * hv * self.v_axis[None, :]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane.to_dict(),
            "u_axis": self.u_axis.tolist(),
            "v_axis": self.v_axis.tolist(),
            "half_extent": self.half_extent.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Aperture:
        return cls(
            Plane.from_dict(payload["plane"]),
            np.asarray(payload["u_axis"]),
            np.asarray(payload["v_axis"]),
            np.asarray(payload["half_extent"]),
        )


def plane_induced_homography(
    K_ref: np.ndarray,
    T_wc_ref: np.ndarray,
    K_new: np.ndarray,
    T_wc_new: np.ndarray,
    plane: Plane,
) -> np.ndarray:
    """Homography mapping reference pixels to new-view pixels for points on ``plane``.

    Derivation: with ``T_cw_new @ T_wc_ref = [R | t]`` the relative pose from the
    reference camera to the new camera, and the plane expressed in the
    *reference camera frame* as ``n_ref . X + d_ref = 0``, the classical result is

        H = K_new (R - t n_ref^T / d_ref) K_ref^{-1}.

    This is an *independent* derivation of the static-display transition, used in
    ``tests/unit/test_optics.py`` to cross-check the screen-point construction.
    """
    K_ref = np.asarray(K_ref, dtype=np.float64).reshape(3, 3)
    K_new = np.asarray(K_new, dtype=np.float64).reshape(3, 3)
    T_wc_ref = np.asarray(T_wc_ref, dtype=np.float64)
    T_wc_new = np.asarray(T_wc_new, dtype=np.float64)

    R_ref = T_wc_ref[:3, :3]
    C_ref = T_wc_ref[:3, 3]
    # Plane in the reference camera frame.
    n_ref = R_ref.T @ plane.normal
    d_ref = float(plane.d + np.dot(plane.normal, C_ref))
    if abs(d_ref) < 1e-12:
        raise ValueError("plane_induced_homography: reference camera lies on the plane")

    T_rel = np.linalg.inv(T_wc_new) @ T_wc_ref  # new_from_ref
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    H = K_new @ (R - np.outer(t, n_ref) / d_ref) @ np.linalg.inv(K_ref)
    return H


def apply_homography(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to ``(N, 2)`` pixel coordinates."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1)
    out = hom @ np.asarray(H, dtype=np.float64).T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return out[:, :2] / w


def fit_homography_dlt(src_uv: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    """Least-squares homography from ``src_uv`` to ``dst_uv`` (normalised DLT).

    Needs at least four finite correspondences.  Hartley normalisation is applied
    to both point sets for conditioning.  Raises ``ValueError`` if the system is
    underdetermined.

    Used to *compensate* an unknown planar (screen) transformation, so that what
    remains is exactly the non-planar differential parallax the Minimum Causal
    Resolving Baseline theory is about.
    """
    src = np.asarray(src_uv, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_uv, dtype=np.float64).reshape(-1, 2)
    ok = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
    src, dst = src[ok], dst[ok]
    if src.shape[0] < 4:
        raise ValueError(f"homography fit needs >= 4 finite correspondences, got {src.shape[0]}")

    T_s, src_n = _hartley_normalise(src)
    T_d, dst_n = _hartley_normalise(dst)

    n = src_n.shape[0]
    A = np.zeros((2 * n, 9))
    x, y = src_n[:, 0], src_n[:, 1]
    u, v = dst_n[:, 0], dst_n[:, 1]
    A[0::2, 0] = -x
    A[0::2, 1] = -y
    A[0::2, 2] = -1.0
    A[0::2, 6] = u * x
    A[0::2, 7] = u * y
    A[0::2, 8] = u
    A[1::2, 3] = -x
    A[1::2, 4] = -y
    A[1::2, 5] = -1.0
    A[1::2, 6] = v * x
    A[1::2, 7] = v * y
    A[1::2, 8] = v
    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_d) @ H_n @ T_s
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _hartley_normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_dist = float(np.mean(np.linalg.norm(centred, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array([[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]])
    return T, centred * scale


def homography_residual(src_uv: np.ndarray, dst_uv: np.ndarray) -> float:
    """RMS reprojection residual (pixels) of the best-fit homography.

    Zero for content confined to a plane; grows with the scene's *differential*
    parallax, which is precisely the quantity a static planar display cannot
    reproduce.
    """
    src = np.asarray(src_uv, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_uv, dtype=np.float64).reshape(-1, 2)
    ok = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
    if int(np.count_nonzero(ok)) < 4:
        return float("nan")
    H = fit_homography_dlt(src[ok], dst[ok])
    warped = apply_homography(H, src[ok])
    return float(np.sqrt(np.mean(np.sum((warped - dst[ok]) ** 2, axis=1))))
