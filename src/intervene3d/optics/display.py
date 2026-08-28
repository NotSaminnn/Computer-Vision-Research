"""H_E -- emissive / display-induced geometry.

Two display modes, and the difference between them is the whole point:

``static``
    A poster, print or ordinary monitor.  The content is *painted on the screen
    plane*, so every content landmark is placed at the intersection of its
    reference viewing ray with the screen.  Under lateral motion it exhibits the
    parallax of the **screen plane**, not of the depth it appears to have.  This
    is the case the Minimum Causal Resolving Baseline theory covers.

``view_tracked``
    A perspective-correct / head-tracked display that re-renders for the current
    observer pose.  Within the screen aperture its predicted geometry is
    *identical* to ``H_D``.  It is included deliberately: a benchmark in which
    every ambiguity resolves would undermine the scientific question.

An independent derivation of the static case is available through
:func:`static_display_homography`: for content confined to a plane, the mapping
from reference pixels to new-view pixels is the plane-induced homography.  The
unit tests check the two agree to sub-micron pixel accuracy.
"""

from __future__ import annotations

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.geometry.planes import Aperture, plane_induced_homography, ray_plane_intersection
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.interventions.actions import Action
from intervene3d.optics.base import HypothesisWorld, OpticalTransition, reference_rays


class DisplayTransition(OpticalTransition):
    """The apparent content is an image displayed on a planar surface."""

    mechanism = OpticalMechanism.EMISSIVE

    def build_world(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> HypothesisWorld:
        interface = hypothesis.interface
        assert interface is not None  # guaranteed by Hypothesis.validate
        mode = hypothesis.params["display_mode"]
        X_direct, _, u_hat = reference_rays(state)
        markers = np.zeros((0, 3)) if markers_cam is None else np.asarray(markers_cam, dtype=np.float64)

        if mode == "static":
            n = X_direct.shape[0]
            origins = np.repeat(state.camera.center[None, :], n, axis=0)
            screen_points, hit = ray_plane_intersection(origins, u_hat, interface.plane)
            screen_points = np.where(hit[:, None], screen_points, np.nan)
            content = screen_points
        elif mode == "view_tracked":
            # The display renders the correct parallax, so its predicted content
            # geometry coincides exactly with the direct hypothesis.
            content = X_direct
        else:  # pragma: no cover - Hypothesis.validate rejects other modes
            raise ValueError(f"unknown display_mode {mode!r}")

        return HypothesisWorld(
            content_points=content,
            reference_camera=state.camera,
            interface=interface,
            markers_cam=markers,
            reflects_observer=False,
            axial_shift=0.0,
            contact_on_interface=True,  # you touch the screen, not the displayed room
            mechanism=OpticalMechanism.EMISSIVE,
        )


def screen_points(
    state: GeometryFeature, interface: Aperture
) -> tuple[np.ndarray, np.ndarray]:
    """Where the reference content rays meet the screen plane.

    Returns ``(points_world, hit)``.
    """
    _, _, u_hat = reference_rays(state)
    origins = np.repeat(state.camera.center[None, :], u_hat.shape[0], axis=0)
    return ray_plane_intersection(origins, u_hat, interface.plane)


def static_display_homography(
    state: GeometryFeature, interface: Aperture, action: Action
) -> np.ndarray:
    """Plane-induced homography from reference pixels to the intervened view.

    Independent derivation of the static-display transition, used to validate
    :class:`DisplayTransition`.
    """
    ref_cam = state.camera
    new_cam = ref_cam.moved(action.delta_T())
    K = ref_cam.intrinsics.matrix()
    return plane_induced_homography(K, ref_cam.T_wc, new_cam.intrinsics.matrix(), new_cam.T_wc, interface.plane)
