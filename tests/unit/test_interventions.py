"""Action and action-space behaviour: bounds, validity, candidate generation."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.geometry.se3 import is_rotation, rotation_angle
from intervene3d.interventions.action_space import ActionSpace, ActionSpaceConfig
from intervene3d.interventions.actions import Action, ActionKind, null_action


def test_null_action_is_the_identity():
    a = null_action()
    assert a.is_null
    assert np.allclose(a.delta_T(), np.eye(4))
    assert a.translation_magnitude == 0.0 and a.rotation_magnitude == 0.0


def test_delta_T_is_a_valid_se3_transform():
    a = Action("compound", ActionKind.COMPOUND, np.array([0.1, -0.05, 0.2]), np.radians([4.0, -2.0, 1.0]))
    T = a.delta_T()
    assert is_rotation(T[:3, :3])
    assert np.allclose(T[:3, 3], a.translation)
    assert np.isclose(rotation_angle(T[:3, :3]), a.rotation_magnitude)


def test_translation_magnitude_is_the_baseline():
    a = Action("lat", ActionKind.TRANSLATE_X, np.array([0.3, 0.4, 0.0]), np.zeros(3))
    assert np.isclose(a.translation_magnitude, 0.5)


def test_candidate_generation_respects_bounds_and_kinds():
    cfg = ActionSpaceConfig(
        translation_steps=(0.1, 0.2),
        rotation_steps_deg=(5.0,),
        enabled_kinds=("translate_x", "yaw"),
        max_translation=0.25,
        max_rotation_deg=10.0,
    )
    space = ActionSpace.from_config(cfg)
    # 1 null + 2 steps x 2 signs (translate_x) + 1 step x 2 signs (yaw)
    assert len(space) == 1 + 4 + 2
    assert {a.kind for a in space} == {ActionKind.NONE, ActionKind.TRANSLATE_X, ActionKind.YAW}
    space.validate()


def test_out_of_bounds_steps_are_dropped_not_silently_clipped():
    cfg = ActionSpaceConfig(
        translation_steps=(0.1, 0.9), rotation_steps_deg=(), enabled_kinds=("translate_x",),
        max_translation=0.25, max_rotation_deg=0.0, include_null=False,
    )
    space = ActionSpace.from_config(cfg)
    assert len(space) == 2  # only +/-0.1 survives; 0.9 is dropped, not shrunk
    assert max(a.translation_magnitude for a in space) <= 0.25


def test_validate_rejects_an_action_exceeding_the_bounds():
    cfg = ActionSpaceConfig(max_translation=0.1, max_rotation_deg=5.0)
    too_far = Action("far", ActionKind.TRANSLATE_X, np.array([0.5, 0.0, 0.0]), np.zeros(3))
    with pytest.raises(ValueError, match="exceeds"):
        ActionSpace((too_far,), cfg)
    too_much_rotation = Action("spin", ActionKind.YAW, np.zeros(3), np.radians([30.0, 0, 0]))
    with pytest.raises(ValueError, match="exceeds"):
        ActionSpace((too_much_rotation,), cfg)


def test_duplicate_action_names_are_rejected():
    a = Action("dup", ActionKind.TRANSLATE_X, np.array([0.1, 0, 0]), np.zeros(3))
    b = Action("dup", ActionKind.TRANSLATE_Y, np.array([0, 0.1, 0]), np.zeros(3))
    with pytest.raises(ValueError, match="unique"):
        ActionSpace((a, b), ActionSpaceConfig())


def test_empty_action_space_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ActionSpace((), ActionSpaceConfig())


def test_unsupported_kind_is_rejected():
    with pytest.raises(ValueError):
        ActionSpace.from_config(ActionSpaceConfig(enabled_kinds=("teleport",)))


def test_max_translation_action_and_non_null():
    space = ActionSpace.from_config(ActionSpaceConfig())
    assert np.isclose(space.max_translation_action.translation_magnitude, 0.30)
    assert all(not a.is_null for a in space.non_null())


def test_lateral_sweep_is_ordered_and_within_bounds():
    baselines = np.linspace(0.0, 0.3, 7)
    space = ActionSpace.lateral_sweep(baselines)
    assert len(space) == len(baselines)
    assert space[0].is_null
    mags = [a.translation_magnitude for a in space]
    assert np.allclose(mags, baselines)


def test_scaled_and_perturbed():
    a = Action("lat", ActionKind.TRANSLATE_X, np.array([0.2, 0.0, 0.0]), np.zeros(3))
    assert np.isclose(a.scaled(0.5).translation_magnitude, 0.1)
    rng = np.random.default_rng(0)
    noisy = a.perturbed(rng, translation_std=0.01, rotation_std=0.001)
    assert not np.allclose(noisy.translation, a.translation)
    assert np.linalg.norm(noisy.translation - a.translation) < 0.1


def test_serialisation_round_trip():
    space = ActionSpace.from_config(ActionSpaceConfig())
    restored = ActionSpace.from_dict(space.to_dict())
    assert restored.names == space.names
    for a, b in zip(space, restored, strict=True):
        assert np.allclose(a.delta_T(), b.delta_T())
