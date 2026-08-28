"""SE(3), camera conventions and projection."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.geometry.camera import (
    Camera,
    CameraIntrinsics,
    backproject,
    look_at,
    project_points,
)
from intervene3d.geometry.se3 import (
    compose,
    euler_to_rotation,
    invert,
    is_rotation,
    rotation_angle,
    rotation_from_axis_angle,
    se3_from_Rt,
    transform_points,
    translation_norm,
)

INTR = CameraIntrinsics(fx=250.0, fy=250.0, cx=160.0, cy=120.0, width=320, height=240)


def random_se3(rng: np.random.Generator) -> np.ndarray:
    return se3_from_Rt(rotation_from_axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi)),
                       rng.uniform(-2, 2, size=3))


def test_rotation_is_proper():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = rotation_from_axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        assert is_rotation(R)


def test_euler_composition_order():
    R = euler_to_rotation(yaw=0.3, pitch=-0.1, roll=0.05)
    expected = (
        rotation_from_axis_angle([0, 1, 0], 0.3)
        @ rotation_from_axis_angle([1, 0, 0], -0.1)
        @ rotation_from_axis_angle([0, 0, 1], 0.05)
    )
    assert np.allclose(R, expected)


def test_inverse_is_exact_and_involutive():
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = random_se3(rng)
        assert np.allclose(compose(T, invert(T)), np.eye(4), atol=1e-12)
        assert np.allclose(invert(invert(T)), T, atol=1e-12)


def test_transform_points_round_trip():
    rng = np.random.default_rng(2)
    T = random_se3(rng)
    pts = rng.normal(size=(30, 3))
    assert np.allclose(transform_points(invert(T), transform_points(T, pts)), pts, atol=1e-12)


def test_composition_is_associative():
    rng = np.random.default_rng(3)
    A, B, C = random_se3(rng), random_se3(rng), random_se3(rng)
    assert np.allclose(compose(compose(A, B), C), compose(A, compose(B, C)), atol=1e-12)


def test_projection_backprojection_round_trip():
    rng = np.random.default_rng(4)
    pts_cam = np.stack(
        [rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50), rng.uniform(1.0, 6.0, 50)], axis=1
    )
    uv, depth, _ = project_points(INTR, pts_cam)
    assert np.allclose(backproject(INTR, uv, depth), pts_cam, atol=1e-10)


def test_projection_matches_the_documented_convention():
    """u = fx x/z + cx, v = fy y/z + cy, with +z forward."""
    p = np.array([[0.4, -0.2, 2.0]])
    uv, depth, in_view = project_points(INTR, p)
    assert np.isclose(uv[0, 0], INTR.fx * 0.4 / 2.0 + INTR.cx)
    assert np.isclose(uv[0, 1], INTR.fy * -0.2 / 2.0 + INTR.cy)
    assert np.isclose(depth[0], 2.0)
    assert bool(in_view[0])


def test_points_behind_the_camera_are_not_visible():
    uv, depth, in_view = project_points(INTR, np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    assert not in_view.any()
    assert np.isnan(uv).all() and np.isnan(depth).all()


def test_points_outside_the_frame_are_not_in_view():
    _, _, in_view = project_points(INTR, np.array([[10.0, 0.0, 1.0]]))
    assert not in_view[0]


def test_camera_center_and_pose_convention():
    rng = np.random.default_rng(5)
    T = random_se3(rng)
    cam = Camera(INTR, T)
    assert np.allclose(cam.center, T[:3, 3])
    # The camera centre maps to the origin of the camera frame.
    assert np.allclose(cam.world_to_camera(cam.center[None, :])[0], np.zeros(3), atol=1e-12)
    # +z of the camera frame is the world-frame optical axis.
    assert np.allclose(cam.forward, T[:3, 2])


def test_moved_applies_the_delta_in_the_camera_frame():
    cam = Camera(INTR, look_at([0, -3, 0], [0, 0, 0]))
    delta = se3_from_Rt(np.eye(3), np.array([0.25, 0.0, 0.0]))
    moved = cam.moved(delta)
    # A pure +x delta moves the centre along the camera's own x axis.
    assert np.allclose(moved.center - cam.center, 0.25 * cam.T_wc[:3, 0], atol=1e-12)
    assert np.allclose(moved.R_wc, cam.R_wc, atol=1e-12)


def test_look_at_produces_a_valid_pose_facing_the_target():
    T = look_at([1.0, -2.0, 0.5], [0.0, 0.0, 0.0])
    assert is_rotation(T[:3, :3])
    cam = Camera(INTR, T)
    target_cam = cam.world_to_camera(np.zeros((1, 3)))[0]
    assert target_cam[2] > 0  # the target is in front
    assert abs(target_cam[0]) < 1e-9 and abs(target_cam[1]) < 1e-9  # and on the optical axis


def test_look_at_rejects_degenerate_input():
    with pytest.raises(ValueError):
        look_at([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


def test_translation_and_rotation_magnitudes():
    T = se3_from_Rt(rotation_from_axis_angle([0, 0, 1], 0.4), np.array([0.3, 0.4, 0.0]))
    assert np.isclose(translation_norm(T), 0.5)
    assert np.isclose(rotation_angle(T[:3, :3]), 0.4)
