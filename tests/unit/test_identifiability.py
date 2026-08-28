"""Separability, action-set identifiability and the epsilon decision."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.data.types import CHANNEL_CONTENT, GeometryFeature
from intervene3d.geometry.camera import Camera, CameraIntrinsics
from intervene3d.hypotheses.families import direct_hypothesis, display_hypothesis
from intervene3d.interventions.action_space import ActionSpace, ActionSpaceConfig
from intervene3d.interventions.actions import Action, ActionKind, null_action
from intervene3d.models.identifiability import EpsilonIdentifiabilityEstimator
from intervene3d.models.separability import DistanceWeights, feature_distance

INTR = CameraIntrinsics(fx=250.0, fy=250.0, cx=160.0, cy=120.0, width=320, height=240)


def _feature(uv, visible, depth=None):
    uv = np.asarray(uv, dtype=float)
    n = uv.shape[0]
    return GeometryFeature(
        uv,
        np.full(n, 2.0) if depth is None else np.asarray(depth, dtype=float),
        np.asarray(visible, dtype=bool),
        np.full(n, CHANNEL_CONTENT, dtype=np.int8),
        Camera(INTR, np.eye(4)),
    )


# -------------------------------------------------------------- the distance
def test_identical_features_have_zero_distance():
    f = _feature([[10.0, 10.0], [20.0, 20.0]], [True, True])
    assert feature_distance(f, f).total == 0.0


def test_motion_term_is_rms_pixel_displacement():
    a = _feature([[0.0, 0.0], [0.0, 0.0]], [True, True])
    b = _feature([[3.0, 4.0], [0.0, 0.0]], [True, True])
    d = feature_distance(a, b, DistanceWeights(motion=1.0, occlusion=0.0))
    assert np.isclose(d.motion, np.sqrt((25.0 + 0.0) / 2))
    assert np.isclose(d.total, d.motion)


def test_occlusion_term_counts_disagreements_not_fractions():
    """A count, so a decisive appearance is not diluted by the landmark count."""
    a = _feature([[0.0, 0.0]] * 50, [True] * 50)
    b = _feature([[0.0, 0.0]] * 50, [True] * 49 + [False])
    d = feature_distance(a, b, DistanceWeights(motion=0.0, occlusion=4.0))
    assert d.occlusion == 1.0
    assert np.isclose(d.total, 4.0)
    # The same single disagreement in a 5-landmark scene is worth the same.
    a2 = _feature([[0.0, 0.0]] * 5, [True] * 5)
    b2 = _feature([[0.0, 0.0]] * 5, [True] * 4 + [False])
    assert feature_distance(a2, b2, DistanceWeights(motion=0.0, occlusion=4.0)).total == 4.0


def test_distance_ignores_landmarks_not_jointly_visible():
    a = _feature([[0.0, 0.0], [100.0, 100.0]], [True, True])
    b = _feature([[0.0, 0.0], [900.0, 900.0]], [True, False])
    d = feature_distance(a, b, DistanceWeights(motion=1.0, occlusion=0.0))
    assert d.motion == 0.0 and d.n_joint_visible == 1


def test_distance_is_symmetric():
    a = _feature([[0.0, 0.0], [5.0, 1.0]], [True, True])
    b = _feature([[2.0, 1.0], [5.0, 3.0]], [True, False])
    assert np.isclose(feature_distance(a, b).total, feature_distance(b, a).total)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        feature_distance(_feature([[0.0, 0.0]], [True]), _feature([[0.0, 0.0]] * 2, [True, True]))


def test_feature_distance_term_is_not_implemented():
    with pytest.raises(NotImplementedError):
        DistanceWeights(feature=0.5)


def test_negative_weights_rejected():
    with pytest.raises(ValueError):
        DistanceWeights(motion=-1.0)


# ------------------------------------------------------ identifiability logic
def test_identifiability_is_the_max_over_actions():
    sep = np.zeros((4, 2, 2))
    sep[:, 0, 1] = sep[:, 1, 0] = [0.1, 5.0, 2.0, 0.0]
    est = EpsilonIdentifiabilityEstimator(epsilon=1.0)
    I = est.identifiability_matrix(sep)
    assert np.isclose(I[0, 1], 5.0)


def test_epsilon_decision_boundary():
    est = EpsilonIdentifiabilityEstimator(epsilon=1.0)

    def scene(value):
        sep = np.zeros((1, 2, 2))
        sep[0, 0, 1] = sep[0, 1, 0] = value
        return est.evaluate(sep)

    assert scene(5.0)[2] is True
    assert scene(1.0)[2] is True     # >= epsilon resolves
    assert scene(0.999)[2] is False  # < epsilon does not
    assert scene(0.0)[2] is False


def test_scene_score_is_the_worst_pair():
    """A scene resolves only if EVERY competing pair can be told apart."""
    sep = np.zeros((1, 3, 3))
    sep[0] = np.array([[0.0, 40.0, 0.2], [40.0, 0.0, 35.0], [0.2, 35.0, 0.0]])
    est = EpsilonIdentifiabilityEstimator(epsilon=1.0)
    _, score, resolvable = est.evaluate(sep)
    assert np.isclose(score, 0.2)
    assert resolvable is False


def test_weighted_score_ignores_hypotheses_the_posterior_has_ruled_out():
    I = np.array([[0.0, 40.0, 0.2], [40.0, 0.0, 35.0], [0.2, 35.0, 0.0]])
    est = EpsilonIdentifiabilityEstimator(epsilon=1.0)
    assert np.isclose(est.weighted_score(I, np.array([1 / 3, 1 / 3, 1 / 3])), 0.2)
    # Once H_R (index 2) is ruled out, the hard pair no longer applies.
    assert np.isclose(est.weighted_score(I, np.array([0.5, 0.5, 1e-6])), 40.0)


def test_weighted_score_falls_back_for_a_point_mass_posterior():
    I = np.array([[0.0, 0.2], [0.2, 0.0]])
    est = EpsilonIdentifiabilityEstimator(epsilon=1.0)
    assert np.isclose(est.weighted_score(I, np.array([1.0, 0.0])), 0.2)


def test_invalid_epsilon_and_shape_rejected():
    with pytest.raises(ValueError):
        EpsilonIdentifiabilityEstimator(epsilon=0.0)
    with pytest.raises(ValueError, match=r"\(A, K, K\)"):
        EpsilonIdentifiabilityEstimator().identifiability_matrix(np.zeros((2, 3)))


# ----------------------------------------------- known separable / non-separable
def test_known_non_separable_case_direct_vs_view_tracked_display(base_scene, reference_feature, estimator):
    """A perspective-correct display is geometrically identical to a real scene."""
    from intervene3d.hypotheses.base import HypothesisSet

    interface = base_scene.content.interface
    hs = HypothesisSet((direct_hypothesis(interface),
                        display_hypothesis(interface, display_mode="view_tracked")))
    space = ActionSpace.from_config(ActionSpaceConfig())
    sep = estimator.pairwise_over_actions(
        reference_feature, hs, space, markers_cam=base_scene.content.observer_markers_cam
    )
    _, score, resolvable = EpsilonIdentifiabilityEstimator(epsilon=1.0).evaluate(sep)
    assert score == 0.0, "a view-tracked display must be indistinguishable from a direct scene"
    assert resolvable is False


def test_known_separable_case_direct_vs_static_display(base_scene, reference_feature, estimator):
    from intervene3d.hypotheses.base import HypothesisSet

    interface = base_scene.content.interface
    hs = HypothesisSet((direct_hypothesis(interface),
                        display_hypothesis(interface, display_mode="static")))
    space = ActionSpace.from_config(ActionSpaceConfig())
    sep = estimator.pairwise_over_actions(
        reference_feature, hs, space, markers_cam=base_scene.content.observer_markers_cam
    )
    _, score, resolvable = EpsilonIdentifiabilityEstimator(epsilon=1.0).evaluate(sep)
    assert resolvable is True and score > 1.0


def test_separability_is_zero_under_the_null_action(base_scene, reference_feature, estimator, hypotheses):
    """Doing nothing cannot separate matched hypotheses -- that is the premise."""
    sep = estimator.pairwise(
        reference_feature, hypotheses, null_action(), markers_cam=base_scene.content.observer_markers_cam
    )
    assert np.allclose(sep, 0.0), "matched variants must be indistinguishable without intervention"


def test_separability_grows_with_baseline(base_scene, reference_feature, estimator):
    from intervene3d.hypotheses.base import HypothesisSet

    interface = base_scene.content.interface
    hs = HypothesisSet((direct_hypothesis(interface),
                        display_hypothesis(interface, display_mode="static")))
    values = []
    for b in (0.02, 0.05, 0.10, 0.20):
        a = Action(f"b{b}", ActionKind.TRANSLATE_X, np.array([b, 0.0, 0.0]), np.zeros(3))
        values.append(
            estimator.pairwise(reference_feature, hs, a,
                               markers_cam=base_scene.content.observer_markers_cam)[0, 1]
        )
    assert all(np.diff(values) > 0), f"separability should increase with baseline, got {values}"


def test_weighted_score_uses_the_map_row_once_the_posterior_has_committed():
    """A hard pair between two already-eliminated hypotheses must not force abstention."""
    # H_1 and H_2 are mutually hard (0.2) but both are separable from H_0 (40, 35).
    I = np.array([[0.0, 40.0, 35.0], [40.0, 0.0, 0.2], [35.0, 0.2, 0.0]])
    est = EpsilonIdentifiabilityEstimator(epsilon=1.0)
    committed = np.array([1.0 - 2e-9, 1e-9, 1e-9])
    # The MAP claim (H_0) IS justified: it is separable from both alternatives.
    assert np.isclose(est.weighted_score(I, committed), 35.0)
    # A global minimum over all pairs would have returned 0.2 and abstained.
    assert est.evaluate(I[None])[1] == 0.2
