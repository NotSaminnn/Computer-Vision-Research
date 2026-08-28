"""Reference-camera generation.

Scenes are built in a canonical *rig* frame (camera at the origin looking down
``+z``) and then mapped into the world by a random rigid transform.  That random
transform is scientifically irrelevant -- everything the benchmark measures is
relative -- but it deliberately exercises the coordinate conventions, so a
mistake in a pose composition cannot hide behind an identity reference pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from intervene3d.geometry.camera import Camera, CameraIntrinsics
from intervene3d.geometry.se3 import euler_to_rotation, rotation_from_axis_angle, se3_from_Rt


def intrinsics_from_config(config: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(config["fx"]),
        fy=float(config["fy"]),
        cx=float(config["cx"]),
        cy=float(config["cy"]),
        width=int(config["width"]),
        height=int(config["height"]),
    )


def random_rig_to_world(rng: np.random.Generator, *, translation_scale: float = 2.0) -> np.ndarray:
    """A random rigid transform mapping the rig frame into the world."""
    axis = rng.normal(size=3)
    angle = rng.uniform(-np.pi, np.pi)
    R = rotation_from_axis_angle(axis, angle)
    t = rng.uniform(-translation_scale, translation_scale, size=3)
    return se3_from_Rt(R, t)


def reference_camera(
    rng: np.random.Generator, camera_config: dict[str, Any], scene_config: dict[str, Any]
) -> tuple[Camera, np.ndarray]:
    """Build the reference camera ``C_0`` and the rig-to-world transform.

    The camera additionally receives a small *pointing* jitter within the rig
    (``scene.camera_jitter_rotation_deg``) so the optical interface is not always
    perfectly centred in the image.
    """
    intr = intrinsics_from_config(camera_config)
    T_world_rig = random_rig_to_world(rng)

    jitter_deg = float(scene_config.get("camera_jitter_rotation_deg", 0.0))
    if jitter_deg > 0.0:
        yaw, pitch = np.radians(rng.uniform(-jitter_deg, jitter_deg, size=2))
        R_jitter = euler_to_rotation(yaw=yaw, pitch=pitch)
    else:
        R_jitter = np.eye(3)
    T_rig_cam = se3_from_Rt(R_jitter, np.zeros(3))

    T_wc = T_world_rig @ T_rig_cam
    return Camera(intr, T_wc), T_world_rig
