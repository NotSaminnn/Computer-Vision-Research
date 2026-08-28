"""Oracle ground truth: resolvability labels and resolving baselines.

Nothing here is assumed.  The resolvability label is *derived* by running the
analytical optics over the declared action set and applying the epsilon
criterion, so the benchmark's non-identifiable cases are non-identifiable as a
consequence of geometry, not because they were declared to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from intervene3d.data.synthetic.optical_variants import content_depth_range
from intervene3d.data.types import CHANNEL_CONTENT, Observation
from intervene3d.geometry.planes import homography_residual
from intervene3d.hypotheses.base import HypothesisSet
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.metrics.mcrb import (
    MCRBResult,
    analytic_pair_is_applicable,
    mcrb_analytic,
    mcrb_numeric,
)
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.separability import GeometrySeparabilityEstimator


@dataclass
class SceneGroundTruth:
    """Every oracle quantity attached to one scene variant."""

    true_index: int
    resolvable: bool
    identifiability_score: float
    identifiability_matrix: np.ndarray
    separability_by_action: np.ndarray
    hardest_competitor_index: int
    oracle_best_action: str
    oracle_action_utility: np.ndarray
    mcrb_operational: MCRBResult
    mcrb_analytic: MCRBResult
    mcrb_compensated: MCRBResult
    z_near: float
    z_far: float
    lateral_separability: np.ndarray
    lateral_residual: np.ndarray
    baselines: np.ndarray

    def to_metadata(self, hypotheses: HypothesisSet) -> dict[str, Any]:
        return {
            "resolvable": bool(self.resolvable),
            "identifiability_score": float(self.identifiability_score),
            "hardest_competitor": hypotheses[self.hardest_competitor_index].name,
            "oracle_best_action": self.oracle_best_action,
            "mcrb": self.mcrb_operational.value,
            "mcrb_method": self.mcrb_operational.method,
            "mcrb_analytic": self.mcrb_analytic.value,
            "mcrb_analytic_applicable": self.mcrb_analytic.applicable,
            "mcrb_compensated": self.mcrb_compensated.value,
            "z_near": self.z_near,
            "z_far": self.z_far,
        }


def compute_ground_truth(
    reference: Observation,
    hypotheses: HypothesisSet,
    true_index: int,
    actions: ActionSpace,
    estimator: GeometrySeparabilityEstimator,
    identifiability: EpsilonIdentifiabilityEstimator,
    *,
    markers_cam: np.ndarray,
    baselines: np.ndarray,
    lateral_space: ActionSpace,
) -> SceneGroundTruth:
    """Run the oracle over the whole action set and derive every GT quantity."""
    feature = reference.feature
    sep = estimator.pairwise_over_actions(feature, hypotheses, actions, markers_cam=markers_cam)
    I = identifiability.identifiability_matrix(sep)

    competitors = [j for j in range(len(hypotheses)) if j != true_index]
    pair_scores = np.array([I[true_index, j] for j in competitors])
    hardest_local = int(np.argmin(pair_scores))
    hardest = competitors[hardest_local]
    score = float(pair_scores[hardest_local])
    resolvable = bool(score >= identifiability.epsilon)

    utility = sep[:, true_index, hardest]
    best_idx = int(np.argmax(utility))

    # --- resolving baselines over a pure lateral sweep -----------------------
    lat = estimator.pairwise_over_actions(feature, hypotheses, lateral_space, markers_cam=markers_cam)
    lateral_sep = lat[:, true_index, hardest]
    mcrb_op = mcrb_numeric(baselines, lateral_sep, identifiability.epsilon)

    residual = _homography_residual_curve(
        feature, hypotheses, true_index, hardest, lateral_space, estimator, markers_cam
    )
    mcrb_comp = mcrb_numeric(baselines, np.nan_to_num(residual, nan=0.0), identifiability.epsilon)
    mcrb_comp = MCRBResult(
        mcrb_comp.value,
        "numeric_homography_compensated",
        mcrb_comp.applicable,
        identifiability.epsilon,
        note="smallest lateral baseline whose non-planar (homography-compensated) residual reaches epsilon",
    )

    z_near, z_far = content_depth_range(reference)
    fx = feature.camera.intrinsics.fx
    if analytic_pair_is_applicable(hypotheses[true_index], hypotheses[hardest]):
        mcrb_an = mcrb_analytic(
            fx, z_near, z_far, identifiability.epsilon, h_i=hypotheses[true_index], h_j=hypotheses[hardest]
        )
    else:
        mcrb_an = mcrb_analytic(
            fx, z_near, z_far, identifiability.epsilon,
            h_i=hypotheses[true_index], h_j=hypotheses[hardest],
        )

    return SceneGroundTruth(
        true_index=true_index,
        resolvable=resolvable,
        identifiability_score=score,
        identifiability_matrix=I,
        separability_by_action=sep,
        hardest_competitor_index=hardest,
        oracle_best_action=actions[best_idx].name,
        oracle_action_utility=utility,
        mcrb_operational=mcrb_op,
        mcrb_analytic=mcrb_an,
        mcrb_compensated=mcrb_comp,
        z_near=z_near,
        z_far=z_far,
        lateral_separability=lateral_sep,
        lateral_residual=residual,
        baselines=np.asarray(baselines, dtype=np.float64),
    )


def _homography_residual_curve(
    feature,
    hypotheses: HypothesisSet,
    i: int,
    j: int,
    lateral_space: ActionSpace,
    estimator: GeometrySeparabilityEstimator,
    markers_cam: np.ndarray,
) -> np.ndarray:
    """RMS residual, per baseline, of the best planar warp between two predictions.

    A static planar display can reproduce *any* planar warp of the reference
    image.  Compensating that warp therefore isolates the differential parallax
    the Minimum Causal Resolving Baseline theory is written about, which is what
    makes ``mcrb_compensated`` directly comparable with :func:`mcrb_analytic`.
    """
    out = np.full(len(lateral_space), np.nan)
    for a_idx, action in enumerate(lateral_space):
        fi = estimator.transition.predict(feature, hypotheses[i], action, markers_cam=markers_cam)
        fj = estimator.transition.predict(feature, hypotheses[j], action, markers_cam=markers_cam)
        m = (fi.channel == CHANNEL_CONTENT) & fi.visible & fj.visible
        if int(np.count_nonzero(m)) >= 4:
            out[a_idx] = homography_residual(fj.uv[m], fi.uv[m])
    return out
