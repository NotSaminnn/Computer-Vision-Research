"""Transition models, encoders, the learned residual and the discriminative baselines."""

from __future__ import annotations

import numpy as np

from intervene3d.data.types import CHANNEL_CONTENT
from intervene3d.hypotheses.families import display_hypothesis
from intervene3d.interventions.actions import Action, ActionKind
from intervene3d.models.baselines import (
    MultinomialLogisticRegression,
    reference_features,
    response_features,
    train_baseline,
)
from intervene3d.models.encoders import GroundTruthEncoder, MockEncoder, build_geometry_encoder
from intervene3d.models.learned_transition import INPUT_DIM, OUTPUT_DIM, ResidualMLP, build_inputs
from intervene3d.models.transition import (
    AnalyticalTransitionModel,
    HybridTransitionModel,
    LearnedOnlyTransitionModel,
    NoHypothesisConditioningTransition,
)

ACTION = Action("lat", ActionKind.TRANSLATE_X, np.array([0.15, 0.0, 0.0]), np.zeros(3))


# ----------------------------------------------------------------- encoders
def test_ground_truth_encoder_is_the_identity(base_scene, hypotheses):
    from intervene3d.data.synthetic import reference_observation

    obs = reference_observation(base_scene.content, hypotheses[0])
    assert GroundTruthEncoder().encode(obs) is obs.feature


def test_mock_encoder_is_reproducible_and_adds_noise(base_scene, hypotheses):
    from intervene3d.data.synthetic import reference_observation

    obs = reference_observation(base_scene.content, hypotheses[0])
    a = MockEncoder(pixel_noise_std=0.5, seed=3).encode(obs)
    b = MockEncoder(pixel_noise_std=0.5, seed=3).encode(obs)
    m = a.visible & b.visible
    assert np.allclose(a.uv[m], b.uv[m])
    assert not np.allclose(a.uv[m], obs.feature.uv[m])


def test_mock_encoder_dropout_removes_landmarks(base_scene, hypotheses):
    from intervene3d.data.synthetic import reference_observation

    obs = reference_observation(base_scene.content, hypotheses[0])
    noisy = MockEncoder(dropout=1.0, seed=0).encode(obs)
    assert not noisy.visible.any()


def test_encoder_factory_names():
    assert build_geometry_encoder(None).name == "ground_truth"
    assert build_geometry_encoder({"name": "mock", "pixel_noise_std": 0.1}).name == "mock"


# --------------------------------------------------------------- transitions
def test_no_hypothesis_conditioning_makes_every_hypothesis_predict_alike(
    base_scene, hypotheses, reference_feature
):
    """The ablation must genuinely destroy the causal information."""
    model = NoHypothesisConditioningTransition()
    markers = base_scene.content.observer_markers_cam
    preds = [model.predict(reference_feature, h, ACTION, markers_cam=markers) for h in hypotheses]
    for p in preds[1:]:
        m = preds[0].visible & p.visible
        assert np.allclose(preds[0].uv[m], p.uv[m], atol=1e-12)
        assert np.array_equal(preds[0].visible, p.visible)


def test_analytical_transition_distinguishes_mechanisms(base_scene, hypotheses, reference_feature):
    model = AnalyticalTransitionModel()
    markers = base_scene.content.observer_markers_cam
    preds = [model.predict(reference_feature, h, ACTION, markers_cam=markers) for h in hypotheses]
    m = preds[0].visible & preds[1].visible & (preds[0].channel == CHANNEL_CONTENT)
    assert np.max(np.abs(preds[0].uv[m] - preds[1].uv[m])) > 1e-3


def test_untrained_residual_reduces_the_hybrid_to_the_analytical_model(
    base_scene, hypotheses, reference_feature
):
    """The MLP's output layer starts at zero, so the hybrid begins exactly analytical."""
    mlp = ResidualMLP(hidden_dim=8, seed=0)
    hybrid = HybridTransitionModel(mlp)
    analytical = AnalyticalTransitionModel()
    markers = base_scene.content.observer_markers_cam
    a = hybrid.predict(reference_feature, hypotheses[0], ACTION, markers_cam=markers)
    b = analytical.predict(reference_feature, hypotheses[0], ACTION, markers_cam=markers)
    m = a.visible & b.visible
    assert np.allclose(a.uv[m], b.uv[m], atol=1e-12)


def test_learned_only_has_no_optical_prior(base_scene, hypotheses, reference_feature):
    """Untrained, it just copies the reference view: all the physics must be learned."""
    model = LearnedOnlyTransitionModel(ResidualMLP(hidden_dim=8, seed=0))
    pred = model.predict(reference_feature, hypotheses[0], ACTION,
                         markers_cam=base_scene.content.observer_markers_cam)
    m = pred.visible & reference_feature.visible
    assert np.allclose(pred.uv[m], reference_feature.uv[m], atol=1e-12)


# ------------------------------------------------------------ learned residual
def test_residual_mlp_shapes_and_determinism(base_scene, hypotheses, reference_feature):
    x = build_inputs(reference_feature, hypotheses[0], ACTION)
    n_content = int(np.count_nonzero(reference_feature.channel == CHANNEL_CONTENT))
    assert x.shape == (n_content, INPUT_DIM)
    mlp = ResidualMLP(hidden_dim=8, seed=1)
    assert mlp.predict(x).shape == (n_content, OUTPUT_DIM)
    assert np.allclose(ResidualMLP(hidden_dim=8, seed=1).predict(x), mlp.predict(x))


def test_residual_mlp_learns_a_simple_mapping():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, INPUT_DIM))
    y = np.stack([x[:, 0] * 2.0, x[:, 1] * -1.5, x[:, 2] * 0.5], axis=1)
    mlp = ResidualMLP(hidden_dim=32, seed=0, learning_rate=0.02)
    history = mlp.fit(x, y, epochs=200, seed=0)
    assert history[-1] < history[0] * 0.2, f"loss did not fall: {history[0]:.4f} -> {history[-1]:.4f}"


def test_residual_mlp_serialisation_round_trip():
    rng = np.random.default_rng(1)
    mlp = ResidualMLP(hidden_dim=8, seed=2)
    mlp.fit(rng.normal(size=(50, INPUT_DIM)), rng.normal(size=(50, OUTPUT_DIM)), epochs=5, seed=0)
    restored = ResidualMLP.from_state_dict(mlp.state_dict())
    x = rng.normal(size=(7, INPUT_DIM))
    assert np.allclose(restored.predict(x), mlp.predict(x))


def test_hypothesis_conditioning_flag_changes_the_input():
    from intervene3d.data.types import GeometryFeature
    from intervene3d.geometry.camera import Camera, CameraIntrinsics

    intr = CameraIntrinsics(200.0, 200.0, 100.0, 75.0, 200, 150)
    f = GeometryFeature(
        np.array([[10.0, 10.0]]), np.array([2.0]), np.array([True]),
        np.array([CHANNEL_CONTENT], dtype=np.int8), Camera(intr, np.eye(4)),
    )
    from intervene3d.geometry.planes import Aperture, Plane

    ap = Aperture.from_plane(Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0])), 0.5, 0.3)
    hyp = display_hypothesis(ap)
    with_cond = build_inputs(f, hyp, ACTION, hypothesis_conditioning=True)
    without = build_inputs(f, hyp, ACTION, hypothesis_conditioning=False)
    assert not np.allclose(with_cond, without)


# --------------------------------------------------------- discriminative base
def test_reference_features_are_identical_across_matched_variants(base_scene, hypotheses):
    """The core claim: no function of the reference view can separate the variants."""
    from intervene3d.data.synthetic import reference_observation

    vectors = [reference_features(reference_observation(base_scene.content, h).feature) for h in hypotheses]
    for v in vectors[1:]:
        assert np.allclose(v, vectors[0], atol=1e-12)


def test_response_features_do_separate_the_variants(base_scene, hypotheses):
    from intervene3d.data.synthetic import reference_observation, simulate

    vectors = []
    for h in hypotheses:
        ref = reference_observation(base_scene.content, h)
        obs = simulate(base_scene.content, h, ACTION, reference_feature=ref.feature)
        vectors.append(response_features(ref.feature, obs.feature))
    assert not np.allclose(vectors[0], vectors[1], atol=1e-6)


def test_logistic_regression_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.5, (60, 2)), rng.normal(2, 0.5, (60, 2))])
    y = np.array([0] * 60 + [1] * 60)
    model = MultinomialLogisticRegression(n_features=2, n_classes=2, epochs=300)
    model.fit(x, y)
    assert (model.predict(x) == y).mean() > 0.95


def test_untrained_classifier_returns_uniform_never_a_silent_guess():
    model = MultinomialLogisticRegression(n_features=3, n_classes=4)
    probs = model.predict_proba(np.zeros((5, 3)))
    assert np.allclose(probs, 0.25)


def test_classifier_on_identical_features_cannot_beat_chance():
    """Formal statement of the benchmark's premise."""
    x = np.ones((30, 4))
    y = np.array([0, 1, 2] * 10)
    baseline = train_baseline("sf", x, y, ["a", "b", "c"], uses_response=False, epochs=200)
    probs = baseline.predict_proba(x)
    assert np.allclose(probs, probs[0], atol=1e-9), "identical inputs must give identical outputs"
    accuracy = float((np.argmax(probs, axis=1) == y).mean())
    assert abs(accuracy - 1 / 3) < 1e-9
