"""H_M -- mixed optical mechanism (partially reflective transparent pane).

PRELIMINARY AND NOT VALIDATED.  The research plan is explicit that mixed optics
should only be implemented after the simpler mechanisms are validated, and that
the first paper should not depend on it.  The implementation here is the
smallest thing that is internally consistent:

* the transmitted content follows :class:`~intervene3d.optics.transmission.TransmissionTransition`
  with the pane's slab parameters;
* the pane simultaneously reflects observer-attached structure, like a mirror,
  which is what makes a partially reflective window recognisable in practice;
* ``reflectance`` currently modulates only the rendered image (see
  :mod:`intervene3d.data.synthetic.dataset_writer`), not the landmark geometry.

Excluded from the Phase 1 experiment by design.  See
``docs/RESEARCH_SPEC_AUDIT.md`` for the list of components carrying this status.
"""

from __future__ import annotations

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.optics.base import HypothesisWorld, OpticalTransition, reference_rays
from intervene3d.optics.transmission import paraxial_axial_shift


class MixedTransition(OpticalTransition):
    """Superposition of transmission through, and reflection off, one pane."""

    mechanism = OpticalMechanism.MIXED

    def build_world(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> HypothesisWorld:
        delta = paraxial_axial_shift(
            hypothesis.params.get("thickness", 0.0), hypothesis.params.get("refractive_index", 1.0)
        )
        X_direct, r, u_hat = reference_rays(state)
        physical = state.camera.center[None, :] + (r[:, None] + delta) * u_hat
        markers = np.zeros((0, 3)) if markers_cam is None else np.asarray(markers_cam, dtype=np.float64)
        return HypothesisWorld(
            content_points=physical,
            reference_camera=state.camera,
            interface=hypothesis.interface,
            markers_cam=markers,
            reflects_observer=True,
            axial_shift=delta,
            contact_on_interface=True,
            mechanism=OpticalMechanism.MIXED,
        )
