"""H_R -- planar mirror reflection.

Two equivalent formulations of planar-mirror optics are implemented, and
``tests/unit/test_optics.py`` asserts they agree:

**Virtual-point formulation** (used by the transition).  The virtual image of a
physical point ``Y`` is ``reflect(Y)``; render it with the *real* camera.

**Virtual-camera formulation** (used as an independent cross-check).  Render the
*physical* points with the reflected camera ``reflect_pose(Pi, T_wc)``.  This is
the formulation used by mirror-aware rendering work such as Gaussian splatting
with virtual camera optimisation.

The scientifically important consequence, and one the preliminary benchmark
encodes rather than hides: **a static planar mirror reflecting a static scene
produces a static virtual scene.**  Its content parallax is therefore identical
to a real scene seen through an opening of the same shape, and ``(H_D, H_R)`` is
*not* separable by content motion alone.  The mirror becomes identifiable only
through the virtual image of observer-attached structure, which moves when the
observer moves.  Whether any allowed action makes that virtual image visible is
exactly the action-set-dependent identifiability question.
"""

from __future__ import annotations

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.geometry.planes import Aperture, Plane, reflect_points, reflect_pose
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.optics.base import HypothesisWorld, OpticalTransition, reference_rays


class MirrorTransition(OpticalTransition):
    """The apparent content is the virtual image of a real scene in a planar mirror."""

    mechanism = OpticalMechanism.REFLECTION

    def build_world(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> HypothesisWorld:
        X_direct, _, _ = reference_rays(state)
        markers = np.zeros((0, 3)) if markers_cam is None else np.asarray(markers_cam, dtype=np.float64)
        return HypothesisWorld(
            content_points=X_direct,  # the virtual image is a static world structure
            reference_camera=state.camera,
            interface=hypothesis.interface,
            markers_cam=markers,
            reflects_observer=True,  # the decisive, motion-dependent mirror cue
            axial_shift=0.0,
            contact_on_interface=True,  # you touch the mirror, not the reflected room
            mechanism=OpticalMechanism.REFLECTION,
        )


def physical_source_points(plane: Plane, virtual_points: np.ndarray) -> np.ndarray:
    """Recover the physical scene that produces ``virtual_points`` in ``plane``.

    Reflection is an involution, so this is the same operation as forming the
    virtual image.  Provided as a named function because the physical/virtual
    distinction is easy to get backwards when reading the code.
    """
    return reflect_points(plane, virtual_points)


def virtual_camera_pose(plane: Plane, T_wc: np.ndarray) -> np.ndarray:
    """The reflected ("virtual") camera pose for a planar mirror."""
    return reflect_pose(plane, T_wc)


def mirror_aperture_visibility(camera, interface: Aperture, points: np.ndarray) -> np.ndarray:
    """Public wrapper around the aperture test, for figures and diagnostics."""
    from intervene3d.optics.base import _aperture_visibility

    return _aperture_visibility(camera, points, interface, allow_on_plane=False)
