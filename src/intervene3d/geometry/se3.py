"""Minimal SE(3) utilities (NumPy only).

Only what the preliminary pipeline actually needs.  Every function is covered by
``tests/unit/test_geometry.py``.
"""

from __future__ import annotations

import numpy as np

SE3_IDENTITY = np.eye(4, dtype=np.float64)


def _as_matrix(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"expected a 4x4 SE(3) matrix, got shape {T.shape}")
    return T


def is_rotation(R: np.ndarray, tol: float = 1e-8) -> bool:
    """True if ``R`` is a proper rotation matrix (orthonormal, det == +1)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    return bool(abs(np.linalg.det(R) - 1.0) < 1e-6)


def rotation_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' formula.  ``axis`` need not be normalised."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_rad) < 1e-15:
        return np.eye(3)
    k = axis / norm
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def euler_to_rotation(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Rotation from camera-frame Euler angles, in radians.

    In the OpenCV camera frame used here:
    ``yaw`` rotates about ``+y`` (down)  -> pan left/right,
    ``pitch`` rotates about ``+x`` (right) -> tilt up/down,
    ``roll`` rotates about ``+z`` (forward) -> optical-axis roll.

    Composition order is ``R = R_yaw @ R_pitch @ R_roll`` (applied right to left).
    """
    R_yaw = rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), yaw)
    R_pitch = rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), pitch)
    R_roll = rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), roll)
    return R_yaw @ R_pitch @ R_roll


def se3_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from a rotation and a translation."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert(T: np.ndarray) -> np.ndarray:
    """Exact SE(3) inverse (transpose of R, not a general matrix inverse)."""
    T = _as_matrix(T)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Left-to-right composition: ``compose(A, B) == A @ B``."""
    out = np.eye(4)
    for T in transforms:
        out = out @ _as_matrix(T)
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply ``T`` to an ``(N, 3)`` array of points."""
    T = _as_matrix(T)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points of shape (N, 3), got {points.shape}")
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_angle(R: np.ndarray) -> float:
    """Geodesic rotation magnitude in radians, numerically clamped."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def translation_norm(T: np.ndarray) -> float:
    """Euclidean magnitude of the translation part, in world units (metres)."""
    return float(np.linalg.norm(_as_matrix(T)[:3, 3]))


def log_se3(T: np.ndarray) -> tuple[float, float]:
    """Return ``(translation_norm, rotation_angle)`` -- the two motion budgets
    the action space is bounded by."""
    T = _as_matrix(T)
    return translation_norm(T), rotation_angle(T[:3, :3])
