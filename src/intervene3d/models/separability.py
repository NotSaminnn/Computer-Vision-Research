r"""Separability: how different are two hypotheses' predicted consequences?

The composite distance from the research specification is

.. math::
    D = \lambda_g D_{\text{geometry}} + \lambda_f D_{\text{feature}}
        + \lambda_m D_{\text{motion}} + \lambda_o D_{\text{occlusion}}

**The preliminary codebase does not pretend all four terms are meaningful.**
Only two are enabled by default and both are validated by unit tests:

``D_motion`` (pixels, weight 1.0)
    RMS image-space displacement between the two predicted landmark sets over
    jointly visible landmarks.  This is the parallax signal the Minimum Causal
    Resolving Baseline theory is written in, so the units of the composite
    distance are pixels and the epsilon threshold is the same perceptual
    threshold ``delta`` used by MCRB.

``D_occlusion`` (a **count**, weight 4.0 px per disagreeing landmark)
    The number of landmarks whose predicted *visibility* differs.  This carries
    the aperture-clipping and virtual-observer-marker cues -- for a mirror it is
    the only cue there is.

    It is a count rather than a fraction on purpose: a fraction would be diluted
    by the total landmark count, so whether a single decisive disagreement (a
    virtual marker appearing where another hypothesis predicts nothing) crosses
    the threshold would depend on how many landmarks the scene happens to have.
    The weight converts one unambiguous appearance/disappearance into pixel-
    equivalent evidence; the default 4.0 places a single disagreement clearly
    above the 1 px perceptual threshold.  It is a modelling choice, not a
    measured constant, and is recorded in every run manifest.

``D_geometry`` (metres, weight 0.0 by default)
    RMS 3-D distance between back-projected landmarks.  Meaningful, but partly
    redundant with ``D_motion`` at the preliminary stage, so it is off by default.

``D_feature`` (weight 0.0, NOT IMPLEMENTED)
    Reserved for a real geometry-foundation-model feature distance.  Raises if a
    non-zero weight is requested, rather than silently contributing zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.hypotheses.base import Hypothesis, HypothesisSet
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.interventions.actions import Action
from intervene3d.models.interfaces import TransitionModel


@dataclass(frozen=True)
class DistanceWeights:
    """Weights of the composite distance ``D``."""

    motion: float = 1.0
    occlusion: float = 4.0
    geometry: float = 0.0
    feature: float = 0.0

    def __post_init__(self) -> None:
        if self.feature != 0.0:
            raise NotImplementedError(
                "D_feature requires a real geometry-foundation-model feature space; "
                "it is not implemented in the preliminary codebase. Set lambda_feature=0."
            )
        if min(self.motion, self.occlusion, self.geometry) < 0.0:
            raise ValueError("distance weights must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "lambda_motion": self.motion,
            "lambda_occlusion": self.occlusion,
            "lambda_geometry": self.geometry,
            "lambda_feature": self.feature,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> DistanceWeights:
        payload = dict(payload or {})
        return cls(
            motion=float(payload.get("lambda_motion", 1.0)),
            occlusion=float(payload.get("lambda_occlusion", 4.0)),
            geometry=float(payload.get("lambda_geometry", 0.0)),
            feature=float(payload.get("lambda_feature", 0.0)),
        )


@dataclass(frozen=True)
class DistanceBreakdown:
    """Per-term decomposition, kept so figures can show *why* two hypotheses differ."""

    motion: float
    occlusion: float  # a count of visibility disagreements, not a fraction
    geometry: float
    feature: float
    total: float
    n_joint_visible: int

    def to_dict(self) -> dict[str, float]:
        return {
            "d_motion": self.motion,
            "d_occlusion": self.occlusion,
            "d_geometry": self.geometry,
            "d_feature": self.feature,
            "d_total": self.total,
            "n_joint_visible": self.n_joint_visible,
        }


def feature_distance(
    a: GeometryFeature, b: GeometryFeature, weights: DistanceWeights | None = None
) -> DistanceBreakdown:
    """Composite distance between two predicted (or one predicted, one observed) features."""
    weights = weights or DistanceWeights()
    if len(a) != len(b):
        raise ValueError(f"feature length mismatch: {len(a)} vs {len(b)}")

    joint = a.visible & b.visible
    n_joint = int(np.count_nonzero(joint))

    if n_joint:
        d_uv = a.uv[joint] - b.uv[joint]
        d_motion = float(np.sqrt(np.mean(np.sum(d_uv**2, axis=1))))
    else:
        d_motion = 0.0

    d_occlusion = float(np.count_nonzero(a.visible != b.visible))

    d_geometry = 0.0
    if weights.geometry > 0.0 and n_joint:
        pa = a.points_camera()[joint]
        pb = b.points_camera()[joint]
        ok = np.all(np.isfinite(pa), axis=1) & np.all(np.isfinite(pb), axis=1)
        if np.any(ok):
            d_geometry = float(np.sqrt(np.mean(np.sum((pa[ok] - pb[ok]) ** 2, axis=1))))

    total = (
        weights.motion * d_motion
        + weights.occlusion * d_occlusion
        + weights.geometry * d_geometry
    )
    return DistanceBreakdown(d_motion, d_occlusion, d_geometry, 0.0, total, n_joint)


class GeometrySeparabilityEstimator:
    """Deterministic geometry-space separability estimator.

    Distributions ``p(O'|H,a)`` are treated as deterministic point predictions in
    the preliminary version, so ``D`` reduces to a distance between predicted
    features.  A stochastic version (needed once the learned transition emits
    uncertainty) would replace :func:`feature_distance` here and nothing else.
    """

    name = "geometry_distance"

    def __init__(self, transition: TransitionModel, weights: DistanceWeights | None = None) -> None:
        self.transition = transition
        self.weights = weights or DistanceWeights()

    def predict_all(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> list[GeometryFeature]:
        """The ``H x a`` row of predicted consequences for one action."""
        return [
            self.transition.predict(state, h, action, markers_cam=markers_cam) for h in hypotheses
        ]

    def pairwise(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
        predictions: list[GeometryFeature] | None = None,
    ) -> np.ndarray:
        """Symmetric ``(K, K)`` matrix of ``Delta_ij(a)`` with a zero diagonal."""
        preds = predictions or self.predict_all(state, hypotheses, action, markers_cam=markers_cam)
        k = len(hypotheses)
        out = np.zeros((k, k), dtype=np.float64)
        for i, j in hypotheses.pairs():
            d = feature_distance(preds[i], preds[j], self.weights).total
            out[i, j] = out[j, i] = d
        return out

    def pairwise_over_actions(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> np.ndarray:
        """``(len(actions), K, K)`` tensor of separability."""
        return np.stack(
            [self.pairwise(state, hypotheses, a, markers_cam=markers_cam) for a in actions], axis=0
        )

    def breakdown(
        self,
        state: GeometryFeature,
        h_i: Hypothesis,
        h_j: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> DistanceBreakdown:
        """Per-term decomposition for one hypothesis pair and one action."""
        fi = self.transition.predict(state, h_i, action, markers_cam=markers_cam)
        fj = self.transition.predict(state, h_j, action, markers_cam=markers_cam)
        return feature_distance(fi, fj, self.weights)
