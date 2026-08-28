"""Base-scene generation: the optical interface, the content and the observer markers.

One *base scene* provides the apparent geometry that every causal variant must
reproduce.  Because all variants are built from the same reference rays, the
matched-counterfactual property holds **by construction** rather than by
appearance optimisation:

    at the reference view every variant projects to identical pixels.

The interface is a finite rectangle -- an opening, a mirror, a screen or a pane
of glass depending on the hypothesis -- and content is inset from its border by
``scene.content_fill`` so that aperture clipping requires a genuine baseline
rather than appearing at the first millimetre of motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from intervene3d.data.synthetic.camera_generator import reference_camera
from intervene3d.data.types import SceneContent
from intervene3d.geometry.camera import Camera
from intervene3d.geometry.planes import Aperture, Plane
from intervene3d.geometry.se3 import rotation_from_axis_angle, transform_points


def _uniform(rng: np.random.Generator, span: Any) -> float:
    lo, hi = float(span[0]), float(span[1])
    return float(rng.uniform(lo, hi))


@dataclass(frozen=True)
class BaseScene:
    """Everything shared by the causal variants of one underlying scene."""

    base_scene_id: str
    content: SceneContent
    interface_distance: float
    content_depth_near: float
    content_depth_far: float
    aperture_half_width: float
    aperture_half_height: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_scene_id": self.base_scene_id,
            "interface_distance": self.interface_distance,
            "content_depth_near": self.content_depth_near,
            "content_depth_far": self.content_depth_far,
            "aperture_half_width": self.aperture_half_width,
            "aperture_half_height": self.aperture_half_height,
        }


def _build_interface(
    rng: np.random.Generator,
    scene_cfg: dict[str, Any],
    T_world_rig: np.ndarray,
    camera: Camera,
) -> tuple[Aperture, float, float, float]:
    """Place the finite optical interface in front of the reference camera.

    The aperture half-width is expressed as a fraction of the *image* half-width
    projected onto the interface plane, so the difficulty knob is resolution- and
    focal-length-independent: 1.0 exactly fills the frame, larger values give an
    interface wider than the field of view.
    """
    distance = _uniform(rng, scene_cfg["interface_distance"])
    intr = camera.intrinsics
    frame_half_width = distance * (intr.width / 2.0) / intr.fx
    half_w = _uniform(rng, scene_cfg["aperture_half_width_frac"]) * frame_half_width
    half_h = half_w * _uniform(rng, scene_cfg["aperture_aspect"])

    offset = float(scene_cfg.get("camera_jitter_translation", 0.0))
    centre_rig = np.array([rng.uniform(-offset, offset), rng.uniform(-offset, offset), distance])

    tilt = np.radians(float(scene_cfg.get("interface_tilt_deg", 0.0)))
    normal_rig = np.array([0.0, 0.0, -1.0])
    if tilt > 0.0:
        axis = rng.normal(size=3)
        axis[2] = 0.0  # tilt about an in-image axis so the plane stays roughly facing us
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        normal_rig = rotation_from_axis_angle(axis, rng.uniform(-tilt, tilt)) @ normal_rig

    centre_w = transform_points(T_world_rig, centre_rig[None, :])[0]
    normal_w = T_world_rig[:3, :3] @ normal_rig
    plane = Plane(centre_w, normal_w)
    return Aperture.from_plane(plane, half_w, half_h), distance, half_w, half_h


def _stratified_unit_square(rng: np.random.Generator, n: int) -> np.ndarray:
    """Jittered stratified samples in ``[-1, 1]^2`` -- better coverage than i.i.d."""
    side = int(np.ceil(np.sqrt(n)))
    gi, gj = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    cells = np.stack([gi.ravel(), gj.ravel()], axis=1)[:n].astype(np.float64)
    jitter = rng.uniform(0.0, 1.0, size=cells.shape)
    return 2.0 * ((cells + jitter) / side) - 1.0


def _build_content(
    rng: np.random.Generator,
    scene_cfg: dict[str, Any],
    camera: Camera,
    interface: Aperture,
    interface_distance: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sample apparent content behind the interface, visible through the aperture.

    Points are seeded on the interface plane inside a shrunk rectangle and then
    pushed back along their viewing rays to a depth drawn from a "corridor"
    profile: the aperture centre recedes furthest, the border stays near.  This
    guarantees (a) reference visibility and (b) a genuine depth spread, which is
    what gives the scene a finite Minimum Causal Resolving Baseline.
    """
    n = int(scene_cfg["num_content_landmarks"])
    fill = float(scene_cfg["content_fill"])
    near = _uniform(rng, scene_cfg["content_depth_near"])
    spread = _uniform(rng, scene_cfg["content_depth_spread"])
    far = near + spread

    st = _stratified_unit_square(rng, n)
    hu, hv = interface.half_extent * fill
    seeds = (
        interface.plane.point[None, :]
        + st[:, 0:1] * hu * interface.u_axis[None, :]
        + st[:, 1:2] * hv * interface.v_axis[None, :]
    )

    centre = camera.center
    directions = seeds - centre[None, :]
    ranges = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / ranges

    radial = np.clip(np.linalg.norm(st, axis=1) / np.sqrt(2.0), 0.0, 1.0)
    profile = 1.0 - radial  # 1 at the centre (far), 0 at the border (near)
    profile = np.clip(profile + rng.normal(0.0, 0.08, size=n), 0.0, 1.0)
    target_depth = interface_distance + near + (far - near) * profile

    axis = camera.forward
    cos_theta = np.clip(directions @ axis, 1e-3, None)
    point_ranges = target_depth / cos_theta
    points = centre[None, :] + point_ranges[:, None] * directions

    # Colour by depth profile so figures read consistently across variants.
    t = (target_depth - target_depth.min()) / max(float(np.ptp(target_depth)), 1e-9)
    colors = np.stack([0.20 + 0.65 * t, 0.35 + 0.30 * (1.0 - t), 0.85 - 0.55 * t], axis=1)

    return points, np.clip(colors, 0.0, 1.0), float(target_depth.min()), float(target_depth.max())


def _build_observer_markers(rng: np.random.Generator, marker_cfg: dict[str, Any]) -> np.ndarray:
    """Positions, in the camera frame, of structure rigidly attached to the observer.

    Their *virtual images* in a mirror move whenever the observer moves; nothing
    else in a static scene does.  The whole observer rig is displaced as a unit by
    a wide random offset, so in some scenes every marker's virtual image lies
    outside the aperture or outside the frame and **no allowed action brings it
    in**.  Those are the mirror cases the benchmark labels non-identifiable, and
    they arise from geometry rather than from being declared.
    """
    k = int(marker_cfg.get("count", 0))
    if k <= 0:
        return np.zeros((0, 3))
    rig_lat = _uniform(rng, marker_cfg["rig_offset_lateral"])
    rig_ver = _uniform(rng, marker_cfg["rig_offset_vertical"])
    jitter = float(marker_cfg.get("jitter", 0.0))
    lateral = rig_lat + rng.uniform(-jitter, jitter, size=k)
    vertical = rig_ver + rng.uniform(-jitter, jitter, size=k)
    forward = np.array([_uniform(rng, marker_cfg["forward_offset"]) for _ in range(k)])
    return np.stack([lateral, vertical, forward], axis=1)


#: How many marker draws to try before pushing the observer rig further off-axis.
_MARKER_REJECTION_ATTEMPTS = 24


def generate_base_scene(rng: np.random.Generator, config: dict[str, Any], index: int) -> BaseScene:
    """Generate one base scene, deterministically given ``rng``.

    Observer markers are drawn by **rejection sampling**: a draw whose virtual
    image would already be visible at the reference view is rejected, because it
    would make a mirror recognisable without any intervention and so break the
    matched-counterfactual property the whole benchmark rests on.  This models a
    careful experimenter choosing a reference viewpoint from which their own
    reflection is out of frame.  If no draw succeeds, the rig is pushed
    progressively further off-axis until it does.
    """
    from intervene3d.data.synthetic.optical_variants import observer_reflection_visible_at_reference

    scene_cfg = config["scene"]
    camera, T_world_rig = reference_camera(rng, config["camera"], scene_cfg)
    interface, distance, half_w, half_h = _build_interface(rng, scene_cfg, T_world_rig, camera)
    points, colors, z_near, z_far = _build_content(rng, scene_cfg, camera, interface, distance)

    def _content_with(markers: np.ndarray) -> SceneContent:
        return SceneContent(
            points=points, colors=colors, reference_camera=camera,
            interface=interface, observer_markers_cam=markers,
        )

    markers = _build_observer_markers(rng, config["observer_markers"])
    content = _content_with(markers)
    attempts = 0
    while observer_reflection_visible_at_reference(content) and attempts < _MARKER_REJECTION_ATTEMPTS:
        markers = _build_observer_markers(rng, config["observer_markers"])
        content = _content_with(markers)
        attempts += 1
    # If resampling did not succeed, displace the whole rig laterally in small
    # ADDITIVE steps.  Scaling multiplicatively cannot move a rig that happens to
    # sit near the optical axis, which silently emitted scenes violating the
    # matched property.  Small steps keep the virtual image just outside the
    # aperture -- the regime where a modest action can bring it back in, which is
    # the scientifically interesting case.
    if observer_reflection_visible_at_reference(content):
        direction = np.sign(markers[:, 0].mean()) or 1.0
        step = 0.05
        offset = 0.0
        while observer_reflection_visible_at_reference(content) and offset < 12.0:
            offset += step
            shifted = markers.copy()
            shifted[:, 0] += direction * offset
            content = _content_with(shifted)
        if observer_reflection_visible_at_reference(content):
            raise RuntimeError(
                f"scene {index}: could not place the observer rig so that its virtual image is "
                "hidden at the reference view. The matched-counterfactual property would be "
                "violated, so no scene is emitted rather than a silently broken one."
            )
    markers = content.observer_markers_cam
    return BaseScene(
        base_scene_id=f"base_{index:05d}",
        content=content,
        interface_distance=distance,
        content_depth_near=z_near,
        content_depth_far=z_far,
        aperture_half_width=half_w,
        aperture_half_height=half_h,
    )
