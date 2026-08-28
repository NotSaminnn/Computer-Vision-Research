r"""MCRB -- Minimum Causal Resolving Baseline.

Naming
------
A literature check (August 2026, recorded in ``docs/LITERATURE_CROSS_RESEARCH.md``)
found no established metric under this name, so the research specification's
name is retained.

The analytical result (static planar display vs. real 3-D scene)
----------------------------------------------------------------
Under a lateral observer translation of baseline ``B``, a real point at depth
``Z`` shifts in the image by

.. math:: \Delta u \approx \frac{fB}{Z}

so two scene points at depths ``Z_1 != Z_2`` produce a **differential** parallax

.. math:: |\Delta u_1 - \Delta u_2| = fB\left|\frac{1}{Z_1} - \frac{1}{Z_2}\right|.

A static planar display cannot produce any differential parallax: after
compensating the screen-plane homography all of its content lies on one plane.
Requiring the differential parallax to exceed the perceptual threshold ``delta``
gives

.. math:: B_{\min} > \frac{\delta}{f\left|1/Z_1 - 1/Z_2\right|}.

Assumptions -- the equation is **only** valid when all of these hold
--------------------------------------------------------------------
* Pure lateral translation perpendicular to the optical axis; no rotation.
* Pinhole camera, no distortion; ``f`` is the horizontal focal length in **pixels**.
* Small-baseline / first-order parallax: ``B << Z``.
* Depths ``Z_1, Z_2`` in **metres**, measured along the optical axis from the
  reference camera, both strictly positive and finite.
* ``delta`` is a perceptual displacement threshold in **pixels**, and is the
  *same* threshold as the identifiability ``epsilon`` when the composite
  distance is dominated by its motion term.
* The competing pair is exactly ``(direct, static planar display)``.  It is
  **not** valid for view-tracked displays (no baseline resolves them), for
  mirrors (the virtual scene has genuine parallax), or for transmission.
* The display's content is assumed to be exactly matched at the reference view.

:func:`mcrb_analytic` refuses to return a value outside these conditions rather
than silently extrapolating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism

#: Hypothesis-pair mechanisms for which the analytical MCRB is defined.
ANALYTIC_APPLICABLE_PAIR = frozenset({OpticalMechanism.DIRECT, OpticalMechanism.EMISSIVE})


@dataclass(frozen=True)
class MCRBResult:
    """A resolving-baseline estimate plus everything needed to interpret it."""

    value: float | None  # metres; None when no baseline in range resolves the pair
    method: str  # "analytic" | "numeric"
    applicable: bool
    delta_px: float
    focal_px: float | None = None
    z_near: float | None = None
    z_far: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mcrb": self.value,
            "method": self.method,
            "applicable": self.applicable,
            "delta_px": self.delta_px,
            "focal_px": self.focal_px,
            "z_near": self.z_near,
            "z_far": self.z_far,
            "note": self.note,
        }


def analytic_pair_is_applicable(h_i: Hypothesis, h_j: Hypothesis) -> bool:
    """True only for ``(direct, static display)``."""
    mechs = {h_i.mechanism, h_j.mechanism}
    if mechs != ANALYTIC_APPLICABLE_PAIR:
        return False
    display = h_i if h_i.mechanism is OpticalMechanism.EMISSIVE else h_j
    return display.params.get("display_mode") == "static"


def differential_parallax(focal_px: float, baseline_m: float, z_near: float, z_far: float) -> float:
    """``f B |1/Z1 - 1/Z2|`` in pixels."""
    return abs(focal_px * baseline_m * (1.0 / z_near - 1.0 / z_far))


def mcrb_analytic(
    focal_px: float,
    z_near: float,
    z_far: float,
    delta_px: float,
    *,
    h_i: Hypothesis | None = None,
    h_j: Hypothesis | None = None,
) -> MCRBResult:
    """Closed-form MCRB for the direct-vs-static-display pair.

    ``z_near`` / ``z_far`` are the extreme content depths of the scene: the
    largest available depth *difference* gives the smallest resolving baseline,
    so this is the optimistic (i.e. minimum) baseline consistent with the theory.
    """
    if h_i is not None and h_j is not None and not analytic_pair_is_applicable(h_i, h_j):
        return MCRBResult(
            None,
            "analytic",
            False,
            delta_px,
            note=(
                "analytic MCRB is defined only for the (direct, static display) pair; "
                f"got ({h_i.symbol}, {h_j.symbol})"
            ),
        )
    if not (math.isfinite(z_near) and math.isfinite(z_far)) or z_near <= 0 or z_far <= 0:
        return MCRBResult(None, "analytic", False, delta_px, note="non-positive or non-finite depths")
    if focal_px <= 0 or delta_px <= 0:
        return MCRBResult(None, "analytic", False, delta_px, note="focal length and delta must be positive")

    inv_diff = abs(1.0 / z_near - 1.0 / z_far)
    if inv_diff < 1e-12:
        return MCRBResult(
            None,
            "analytic",
            False,
            delta_px,
            focal_px=focal_px,
            z_near=z_near,
            z_far=z_far,
            note="fronto-planar content: no depth spread, so no baseline produces differential parallax",
        )
    value = delta_px / (focal_px * inv_diff)
    return MCRBResult(
        float(value),
        "analytic",
        True,
        delta_px,
        focal_px=focal_px,
        z_near=z_near,
        z_far=z_far,
        note="B_min = delta / (f |1/Z1 - 1/Z2|)",
    )


def mcrb_numeric(
    baselines: np.ndarray, separability: np.ndarray, epsilon: float
) -> MCRBResult:
    """Smallest sampled ``|B|`` at which separability first reaches ``epsilon``.

    The operational, mechanism-agnostic definition used to label the benchmark.
    Linear interpolation between the bracketing samples gives sub-sample
    resolution.  Returns ``value=None`` when no sampled baseline resolves the
    pair -- which is the *correct* answer for a genuinely non-identifiable case,
    not a failure.
    """
    b = np.abs(np.asarray(baselines, dtype=np.float64)).ravel()
    s = np.asarray(separability, dtype=np.float64).ravel()
    if b.shape != s.shape:
        raise ValueError("baselines and separability must have the same shape")
    order = np.argsort(b)
    b, s = b[order], s[order]

    hits = np.nonzero(s >= epsilon)[0]
    if hits.size == 0:
        return MCRBResult(
            None,
            "numeric",
            True,
            epsilon,
            note=f"no sampled baseline up to {b.max():.4f} m reaches epsilon={epsilon:g}",
        )
    k = int(hits[0])
    if k == 0:
        return MCRBResult(float(b[0]), "numeric", True, epsilon, note="resolved at the smallest sampled baseline")
    s0, s1 = s[k - 1], s[k]
    if s1 - s0 <= 1e-12:
        value = float(b[k])
    else:
        t = (epsilon - s0) / (s1 - s0)
        value = float(b[k - 1] + t * (b[k] - b[k - 1]))
    return MCRBResult(value, "numeric", True, epsilon, note="linear interpolation between bracketing samples")


def mcrb_absolute_error(predicted: float | None, ground_truth: float | None) -> float | None:
    """``MAE_MCRB = |B_hat_min - B_GT_min|`` for one sample.

    Returns ``None`` when either side is undefined (an unresolvable case); such
    samples are excluded from the aggregate rather than imputed.
    """
    if predicted is None or ground_truth is None:
        return None
    return abs(float(predicted) - float(ground_truth))
