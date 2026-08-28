"""Rigid-body geometry, pinhole cameras and planar primitives.

Coordinate conventions used **everywhere** in this repository (see
``docs/SOFTWARE_ARCHITECTURE.md`` section "Conventions"):

* World frame: right-handed, metres.
* Camera frame: OpenCV convention -- ``+x`` right, ``+y`` down, ``+z`` forward.
* A pose is stored as ``T_wc`` (4x4), the *camera-to-world* transform.
  The camera centre in world coordinates is ``T_wc[:3, 3]``.
* ``X_cam = T_cw @ X_world`` with ``T_cw = inv(T_wc)``.
* Projection: ``u = fx * x/z + cx``, ``v = fy * y/z + cy``, valid only for ``z > 0``.
* An action is a delta expressed **in the reference camera frame**:
  ``T_wc_new = T_wc_ref @ delta``.  Lateral translation is therefore motion
  along the reference camera's own x-axis.
"""

from intervene3d.geometry.camera import (
    Camera,
    CameraIntrinsics,
    backproject,
    look_at,
    project_points,
)
from intervene3d.geometry.planes import (
    Aperture,
    Plane,
    plane_induced_homography,
    ray_plane_intersection,
    reflect_plane_matrix,
    reflect_points,
    reflect_pose,
    signed_distance,
)
from intervene3d.geometry.se3 import (
    SE3_IDENTITY,
    compose,
    euler_to_rotation,
    invert,
    is_rotation,
    log_se3,
    rotation_angle,
    rotation_from_axis_angle,
    transform_points,
    translation_norm,
)

__all__ = [
    "SE3_IDENTITY",
    "compose",
    "euler_to_rotation",
    "invert",
    "is_rotation",
    "log_se3",
    "rotation_angle",
    "rotation_from_axis_angle",
    "transform_points",
    "translation_norm",
    "Camera",
    "CameraIntrinsics",
    "look_at",
    "project_points",
    "backproject",
    "Aperture",
    "Plane",
    "plane_induced_homography",
    "ray_plane_intersection",
    "reflect_plane_matrix",
    "reflect_points",
    "reflect_pose",
    "signed_distance",
]
