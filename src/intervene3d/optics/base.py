"""Analytical optical transition layer -- shared machinery.

Design rule from the research plan (section 9): *do not hide all physics inside
an opaque neural network*.  Each mechanism therefore reduces to an explicit,
inspectable world construction plus one shared projection routine.

The interface every transition exposes is::

    predicted_feature = transition.predict(state=F_t, hypothesis=H_k, action=a)

Two-step semantics
------------------
1. ``build_world(F_t, H_k)`` -- turn the reference-view feature into the world
   structure that ``H_k`` *asserts* is responsible for it.  This is where the
   mechanisms genuinely differ: each assigns a **different depth** to the same
   reference pixels.  A direct scene keeps the perceived depth; a static display
   places the content on the screen plane; a glass pane pushes it back by the
   slab's axial shift; a mirror keeps the virtual structure but adds the moving
   virtual image of observer-attached markers.
2. ``project_world(world, action)`` -- render that world from the intervened
   camera.  Identical for every mechanism, so any difference between predictions
   is attributable to step 1 alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from intervene3d.data.types import (
    CHANNEL_CONTENT,
    CHANNEL_FRAME,
    CHANNEL_MARKER,
    GeometryFeature,
)
from intervene3d.geometry.camera import Camera, backproject
from intervene3d.geometry.planes import Aperture, ray_plane_intersection, reflect_points
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.interventions.actions import Action

#: Landmark rays whose intersection with the interface is this much closer than
#: the landmark itself still count as "on the interface" (screen content).
_APERTURE_EPS = 1e-6


@dataclass(frozen=True)
class HypothesisWorld:
    """The world structure a hypothesis asserts, anchored at a reference camera."""

    content_points: np.ndarray  # (N, 3) world positions that determine projected pixels
    reference_camera: Camera
    interface: Aperture | None
    markers_cam: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    reflects_observer: bool = False
    axial_shift: float = 0.0  # metres; transmission only
    contact_on_interface: bool = False
    mechanism: OpticalMechanism = OpticalMechanism.DIRECT

    @property
    def n_content(self) -> int:
        return int(self.content_points.shape[0])

    @property
    def n_markers(self) -> int:
        return int(self.markers_cam.shape[0])


class OpticalTransition(ABC):
    """Base class for every analytical hypothesis-conditioned transition."""

    mechanism: OpticalMechanism

    # ------------------------------------------------------------------ public
    def predict(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> GeometryFeature:
        """Predict ``F_{t+1}`` given ``F_t``, a hypothesis and an intervention."""
        world = self.build_world(state, hypothesis, markers_cam=markers_cam)
        return self.project_world(world, action)

    def predict_contact_depth(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> np.ndarray:
        """Depth of the first physical surface along each landmark ray."""
        world = self.build_world(state, hypothesis, markers_cam=markers_cam)
        return contact_depth_for_world(world, action)

    # -------------------------------------------------------------- extension
    @abstractmethod
    def build_world(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> HypothesisWorld:
        """Turn the reference feature into the world this mechanism asserts."""

    # --------------------------------------------------------------- rendering
    @staticmethod
    def project_world(world: HypothesisWorld, action: Action) -> GeometryFeature:
        return project_world(world, action)


# --------------------------------------------------------------------- helpers
def reference_rays(state: GeometryFeature) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose the reference feature's content channel into ``(X_direct, r, u_hat)``.

    ``X_direct`` is the world position a *direct* reading assigns to each content
    landmark, ``r`` its Euclidean range from the reference camera centre, and
    ``u_hat`` the unit viewing direction.  All mechanisms are then expressed as
    displacements along these same reference rays, which is exactly why every
    variant is pixel-identical at the reference view.
    """
    cam = state.camera
    m = state.channel == CHANNEL_CONTENT
    uv = state.uv[m]
    depth = state.depth[m]
    pts_cam = np.full((uv.shape[0], 3), np.nan)
    ok = np.isfinite(depth) & np.all(np.isfinite(uv), axis=1)
    if np.any(ok):
        pts_cam[ok] = backproject(cam.intrinsics, uv[ok], depth[ok])
    X_direct = np.full_like(pts_cam, np.nan)
    if np.any(ok):
        X_direct[ok] = cam.camera_to_world(pts_cam[ok])
    delta = X_direct - cam.center[None, :]
    r = np.linalg.norm(delta, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u_hat = delta / np.where(r[:, None] > 1e-12, r[:, None], np.nan)
    return X_direct, r, u_hat


def _aperture_visibility(
    camera: Camera, points: np.ndarray, interface: Aperture, *, allow_on_plane: bool
) -> np.ndarray:
    """Does the ray from ``camera`` to each point pass through the aperture?

    ``allow_on_plane`` accepts landmarks that lie *on* the interface (screen
    content), which would otherwise fail the "interface is nearer than the
    landmark" test by a floating-point hair.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = points.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    center = camera.center
    delta = points - center[None, :]
    t_point = np.linalg.norm(delta, axis=1)
    finite = np.isfinite(t_point) & (t_point > 1e-12)
    dirs = np.zeros_like(delta)
    dirs[finite] = delta[finite] / t_point[finite, None]
    hit_points, hit = ray_plane_intersection(
        np.repeat(center[None, :], n, axis=0), dirs, interface.plane
    )
    inside = interface.contains(hit_points)
    t_plane = np.linalg.norm(hit_points - center[None, :], axis=1)
    if allow_on_plane:
        nearer = t_plane <= t_point + 1e-6
    else:
        nearer = t_plane < t_point - _APERTURE_EPS
    return finite & hit & inside & np.nan_to_num(nearer, nan=False)


def project_world(world: HypothesisWorld, action: Action) -> GeometryFeature:
    """Render a :class:`HypothesisWorld` from the intervened camera.

    Identical for every mechanism -- all mechanism-specific physics lives in
    :meth:`OpticalTransition.build_world`.
    """
    cam = world.reference_camera.moved(action.delta_T())
    interface = world.interface

    # ---- content channel -------------------------------------------------
    uv_c, z_c, in_view_c = cam.project(world.content_points)
    vis_c = in_view_c.copy()
    if interface is not None:
        on_plane = world.mechanism is OpticalMechanism.EMISSIVE
        vis_c &= _aperture_visibility(cam, world.content_points, interface, allow_on_plane=on_plane)

    depth_c = z_c.copy()
    if world.axial_shift != 0.0:
        # Perceived (apparent) depth: a plane-parallel slab makes content appear
        # closer by `axial_shift` along the current viewing ray.
        rng = np.linalg.norm(world.content_points - cam.center[None, :], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(rng > 1e-9, (rng - world.axial_shift) / rng, np.nan)
        depth_c = depth_c * scale

    # ---- interface frame channel ----------------------------------------
    if interface is not None:
        corners = interface.corners()
        uv_f, z_f, vis_f = cam.project(corners)
    else:  # pragma: no cover - every hypothesis in the benchmark carries an interface
        uv_f = np.zeros((4, 2))
        z_f = np.full(4, np.nan)
        vis_f = np.zeros(4, dtype=bool)

    # ---- observer-marker channel ----------------------------------------
    k = world.n_markers
    if k and world.reflects_observer and interface is not None:
        markers_world = cam.camera_to_world(world.markers_cam)
        virtual = reflect_points(interface.plane, markers_world)
        uv_m, z_m, in_view_m = cam.project(virtual)
        vis_m = in_view_m & _aperture_visibility(cam, virtual, interface, allow_on_plane=False)
    else:
        uv_m = np.full((k, 2), np.nan)
        z_m = np.full(k, np.nan)
        vis_m = np.zeros(k, dtype=bool)

    uv = np.concatenate([uv_c, uv_f, uv_m], axis=0)
    depth = np.concatenate([depth_c, z_f, z_m], axis=0)
    visible = np.concatenate([vis_c, vis_f, vis_m], axis=0)
    channel = np.concatenate(
        [
            np.full(world.n_content, CHANNEL_CONTENT, dtype=np.int8),
            np.full(4, CHANNEL_FRAME, dtype=np.int8),
            np.full(k, CHANNEL_MARKER, dtype=np.int8),
        ]
    )
    return GeometryFeature(uv, depth, visible, channel, cam)


def contact_depth_for_world(world: HypothesisWorld, action: Action) -> np.ndarray:
    """Physical *contact* depth per landmark (the first surface an agent would touch).

    For a direct scene the contact surface is the content itself; for a mirror,
    display or glass pane it is the interface.  This is the quantity a robot
    needs and the quantity monocular depth models get wrong on non-Lambertian
    surfaces.
    """
    cam = world.reference_camera.moved(action.delta_T())
    interface = world.interface
    n = world.n_content

    if world.contact_on_interface and interface is not None:
        dirs = cam.ray_directions(world.content_points)
        hits, ok = ray_plane_intersection(
            np.repeat(cam.center[None, :], n, axis=0), dirs, interface.plane
        )
        contact_c = np.full(n, np.nan)
        if np.any(ok):
            contact_c[ok] = cam.world_to_camera(hits[ok])[:, 2]
    else:
        contact_c = cam.world_to_camera(world.content_points)[:, 2]

    if interface is not None:
        contact_f = cam.world_to_camera(interface.corners())[:, 2]
    else:  # pragma: no cover
        contact_f = np.full(4, np.nan)

    contact_m = np.full(world.n_markers, np.nan)
    if world.n_markers and interface is not None:
        contact_m = np.full(world.n_markers, np.nan)
        dirs = np.repeat(cam.forward[None, :], world.n_markers, axis=0)
        hits, ok = ray_plane_intersection(
            np.repeat(cam.center[None, :], world.n_markers, axis=0), dirs, interface.plane
        )
        if np.any(ok):
            contact_m[ok] = cam.world_to_camera(hits[ok])[:, 2]

    return np.concatenate([contact_c, contact_f, contact_m])
