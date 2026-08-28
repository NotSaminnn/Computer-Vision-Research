r"""H_T -- transmission through a plane-parallel transparent slab.

Model and its assumptions
-------------------------
The default model is the **paraxial axial shift**.  For a slab of thickness
``d`` and refractive index ``n`` viewed near normal incidence, content behind
the slab appears displaced towards the observer along the viewing ray by

.. math::  \Delta = d\,\bigl(1 - 1/n\bigr).

Consequences used by the implementation:

* Anchoring at the reference view, if the perceived range of a content landmark
  is ``r``, the **physical** content lies at range ``r + Delta`` along the same
  reference ray.
* Because the apparent point is displaced along the *current* viewing ray, its
  projected pixel coincides with the projection of the physical point from any
  camera.  The transition therefore projects the physical points, and reports a
  perceived depth scaled by ``(range - Delta) / range``.

Assumptions -- and where the model stops being valid
----------------------------------------------------
* Plane-parallel slab of uniform index; no curvature, no wedge.
* Near-normal incidence.  The exact lateral ray displacement is
  ``s = d sin(theta) (1 - cos(theta) / (n cos(theta_t)))`` with
  ``sin(theta_t) = sin(theta)/n``; :func:`slab_lateral_displacement` implements
  it and :func:`paraxial_validity_angle` reports the incidence angle at which
  the paraxial approximation's error exceeds a tolerance.  Beyond that angle the
  results of this module should not be trusted.
* Single interface pair; no internal reflections, no dispersion, no absorption.
* Units are metres throughout; ``n`` is dimensionless and ``>= 1``.

Because the axial shift is small for realistic window glass
(``d = 6 mm``, ``n = 1.5`` gives ``Delta = 2 mm``), ``H_T`` is *near*
``H_D`` and is expected to be resolvable only under large baselines -- which is
scientifically the right behaviour and a useful source of genuinely
non-identifiable benchmark cases.
"""

from __future__ import annotations

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.optics.base import HypothesisWorld, OpticalTransition, reference_rays


def paraxial_axial_shift(thickness: float, refractive_index: float) -> float:
    """``Delta = d (1 - 1/n)`` -- the apparent axial displacement, in metres."""
    if refractive_index < 1.0:
        raise ValueError("refractive_index must be >= 1")
    return float(thickness) * (1.0 - 1.0 / float(refractive_index))


def slab_lateral_displacement(
    thickness: float, refractive_index: float, incidence_rad: np.ndarray | float
) -> np.ndarray:
    """Exact lateral ray displacement through a plane-parallel slab (metres).

    ``s = d sin(theta) [1 - cos(theta) / (n cos(theta_t))]`` with Snell's law
    ``sin(theta_t) = sin(theta) / n``.  Provided for validation of the paraxial
    model; not used by the default transition.
    """
    theta = np.asarray(incidence_rad, dtype=np.float64)
    n = float(refractive_index)
    sin_t = np.sin(theta) / n
    cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, 1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        s = float(thickness) * np.sin(theta) * (1.0 - np.cos(theta) / (n * cos_t))
    return s


def paraxial_validity_angle(
    refractive_index: float, *, relative_tolerance: float = 0.05
) -> float:
    """Largest incidence angle (radians) at which the paraxial axial shift is
    within ``relative_tolerance`` of the exact along-axis displacement.

    The exact axial displacement of the apparent source for incidence ``theta``
    is ``d (1 - tan(theta_t)/tan(theta))``; the paraxial limit of that expression
    as ``theta -> 0`` is ``d (1 - 1/n)``.
    """
    n = float(refractive_index)
    paraxial = 1.0 - 1.0 / n
    angles = np.radians(np.linspace(0.1, 89.0, 4000))
    sin_t = np.sin(angles) / n
    cos_t = np.sqrt(np.clip(1.0 - sin_t**2, 0.0, 1.0))
    tan_t = sin_t / cos_t
    exact = 1.0 - tan_t / np.tan(angles)
    rel_err = np.abs(exact - paraxial) / max(paraxial, 1e-12)
    ok = np.nonzero(rel_err <= relative_tolerance)[0]
    return float(angles[ok[-1]]) if ok.size else 0.0


class TransmissionTransition(OpticalTransition):
    """The apparent content is seen through a plane-parallel transparent slab."""

    mechanism = OpticalMechanism.TRANSMISSION

    def build_world(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> HypothesisWorld:
        delta = paraxial_axial_shift(
            hypothesis.params["thickness"], hypothesis.params["refractive_index"]
        )
        X_direct, r, u_hat = reference_rays(state)
        physical = state.camera.center[None, :] + (r[:, None] + delta) * u_hat
        markers = np.zeros((0, 3)) if markers_cam is None else np.asarray(markers_cam, dtype=np.float64)
        return HypothesisWorld(
            content_points=physical,
            reference_camera=state.camera,
            interface=hypothesis.interface,
            markers_cam=markers,
            reflects_observer=False,
            axial_shift=delta,
            contact_on_interface=True,  # you touch the pane, not the room behind it
            mechanism=OpticalMechanism.TRANSMISSION,
        )
