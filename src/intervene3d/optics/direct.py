"""H_D -- direct physical geometry.

The reference camera transformation is the ordinary rigid one, ``X' = R X + t``,
applied through :meth:`intervene3d.geometry.camera.Camera.moved`.  The content
sits exactly where a direct reading of the reference view places it, so this
mechanism is the "null hypothesis" against which the optical mechanisms are
compared.
"""

from __future__ import annotations

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.optics.base import HypothesisWorld, OpticalTransition, reference_rays


class DirectTransition(OpticalTransition):
    """The content really is where it appears to be."""

    mechanism = OpticalMechanism.DIRECT

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
            content_points=X_direct,
            reference_camera=state.camera,
            # An occluding opening, not an optical surface: contact geometry
            # stays on the content (contact_on_interface=False below).
            interface=hypothesis.interface,
            markers_cam=markers,
            reflects_observer=False,
            axial_shift=0.0,
            contact_on_interface=False,
            mechanism=OpticalMechanism.DIRECT,
        )
