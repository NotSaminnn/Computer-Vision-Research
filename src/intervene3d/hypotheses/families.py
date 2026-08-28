"""Convenience constructors for the standard hypothesis family."""

from __future__ import annotations

import numpy as np

from intervene3d.geometry.planes import Aperture, Plane
from intervene3d.hypotheses.base import Hypothesis, HypothesisSet, OpticalMechanism


def direct_hypothesis(interface: Aperture | None = None, name: str = "H_D") -> Hypothesis:
    """H_D -- the apparent geometry really is where it appears to be.

    ``interface`` is optional and, when given, denotes a purely **occluding
    opening** (a doorway or window frame) rather than an optical surface.  In the
    matched benchmark every variant shares the same opening, so aperture clipping
    cannot by itself reveal the mechanism.
    """
    return Hypothesis(OpticalMechanism.DIRECT, interface, {}, name)


def mirror_hypothesis(interface: Aperture, name: str = "H_R") -> Hypothesis:
    """H_R -- the content is the virtual image of a real scene in a planar mirror."""
    return Hypothesis(OpticalMechanism.REFLECTION, interface, {}, name)


def display_hypothesis(
    interface: Aperture,
    *,
    display_mode: str = "static",
    name: str = "H_E",
) -> Hypothesis:
    """H_E -- the content is painted on a planar emissive/reflective surface.

    ``display_mode="static"``
        A poster or an ordinary monitor: the screen texture does not change with
        the observer.  Under lateral motion the content moves with the *screen
        plane*, not with its apparent depth -- this is the parallax signature.

    ``display_mode="view_tracked"``
        A perspective-correct / head-tracked display: the screen re-renders for
        the current observer pose, so within the screen aperture the observation
        is *identical* to H_D.  This is the deliberately non-identifiable case.
    """
    return Hypothesis(OpticalMechanism.EMISSIVE, interface, {"display_mode": display_mode}, name)


def glass_hypothesis(
    interface: Aperture,
    *,
    thickness: float = 0.01,
    refractive_index: float = 1.5,
    name: str = "H_T",
) -> Hypothesis:
    """H_T -- the content is viewed through a plane-parallel transparent slab.

    ``thickness`` in metres, ``refractive_index`` dimensionless.  The apparent
    axial shift under the paraxial approximation is ``d (1 - 1/n)``; see
    :mod:`intervene3d.optics.transmission` for the assumptions and their limits.
    """
    return Hypothesis(
        OpticalMechanism.TRANSMISSION,
        interface,
        {"thickness": float(thickness), "refractive_index": float(refractive_index)},
        name,
    )


def mixed_hypothesis(
    interface: Aperture,
    *,
    reflectance: float = 0.3,
    thickness: float = 0.01,
    refractive_index: float = 1.5,
    name: str = "H_M",
) -> Hypothesis:
    """H_M -- a partially reflective transparent pane (reflection + transmission).

    PRELIMINARY.  Implemented as a linear superposition of the H_T and H_R
    landmark responses weighted by ``reflectance``.  Not validated against a
    physically based renderer; excluded from the Phase 1 experiment by design.
    """
    return Hypothesis(
        OpticalMechanism.MIXED,
        interface,
        {
            "reflectance": float(reflectance),
            "thickness": float(thickness),
            "refractive_index": float(refractive_index),
        },
        name,
    )


def phase1_hypothesis_set(interface: Aperture) -> HypothesisSet:
    """The three-way family the research spec prescribes for Phase 1.

    ``{direct, static display, planar mirror}`` sharing one optical interface
    pose -- exactly the "prove the problem exists" configuration.
    """
    return HypothesisSet(
        (
            direct_hypothesis(interface),
            display_hypothesis(interface, display_mode="static"),
            mirror_hypothesis(interface),
        )
    )


def default_interface(distance: float = 2.0, half_width: float = 0.6, half_height: float = 0.45) -> Aperture:
    """A camera-facing interface plane ``distance`` metres down the world ``+y`` axis.

    Only used by tests and quick interactive checks; the synthetic generator
    builds interfaces from its own configuration.
    """
    plane = Plane(np.array([0.0, distance, 0.0]), np.array([0.0, -1.0, 0.0]))
    return Aperture.from_plane(plane, half_width, half_height)
