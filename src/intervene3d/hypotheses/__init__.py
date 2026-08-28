"""Explicit optical-hypothesis abstraction.

Hypothesis family (research spec section 6):

===========  ==========================================================
``H_D``      direct physical geometry
``H_R``      reflection (planar mirror)
``H_T``      transmission / refraction (planar glass)
``H_E``      emissive / display-induced geometry
``H_M``      mixed optical mechanism
===========  ==========================================================

**"Unidentifiable" is deliberately NOT a member of this family.**  It is an
inference outcome about the triple ``(H, A, O)`` -- see
:mod:`intervene3d.inference.result`.
"""

from intervene3d.hypotheses.base import (
    MECHANISM_LABELS,
    Hypothesis,
    HypothesisSet,
    OpticalMechanism,
)
from intervene3d.hypotheses.families import (
    direct_hypothesis,
    display_hypothesis,
    glass_hypothesis,
    mirror_hypothesis,
    mixed_hypothesis,
    phase1_hypothesis_set,
)

__all__ = [
    "Hypothesis",
    "HypothesisSet",
    "OpticalMechanism",
    "MECHANISM_LABELS",
    "direct_hypothesis",
    "display_hypothesis",
    "mirror_hypothesis",
    "glass_hypothesis",
    "mixed_hypothesis",
    "phase1_hypothesis_set",
]
