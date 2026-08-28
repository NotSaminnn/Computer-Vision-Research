"""Hypothesis representation: validation, equality, labels, serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.geometry.planes import Aperture, Plane
from intervene3d.hypotheses.base import Hypothesis, HypothesisSet, OpticalMechanism
from intervene3d.hypotheses.families import (
    direct_hypothesis,
    display_hypothesis,
    glass_hypothesis,
    mirror_hypothesis,
    mixed_hypothesis,
    phase1_hypothesis_set,
)


@pytest.fixture
def interface() -> Aperture:
    return Aperture.from_plane(Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0])), 0.5, 0.35)


def test_symbols_and_labels(interface):
    assert direct_hypothesis().symbol == "H_D"
    assert mirror_hypothesis(interface).symbol == "H_R"
    assert glass_hypothesis(interface).symbol == "H_T"
    assert display_hypothesis(interface).symbol == "H_E"
    assert mixed_hypothesis(interface).symbol == "H_M"
    assert display_hypothesis(interface).label == "Display"


def test_unidentifiable_is_not_a_mechanism():
    """Non-identifiability is an inference outcome, never a light-transport class."""
    values = {m.value for m in OpticalMechanism}
    assert values == {"direct", "reflection", "transmission", "emissive", "mixed"}
    for bad in ("unidentifiable", "unresolved", "abstain", "unknown"):
        with pytest.raises(ValueError):
            OpticalMechanism(bad)


def test_optical_mechanisms_require_an_interface():
    for mech, params in (
        (OpticalMechanism.REFLECTION, {}),
        (OpticalMechanism.EMISSIVE, {"display_mode": "static"}),
        (OpticalMechanism.TRANSMISSION, {"thickness": 0.01, "refractive_index": 1.5}),
        (OpticalMechanism.MIXED, {"reflectance": 0.3}),
    ):
        with pytest.raises(ValueError, match="requires an optical interface"):
            Hypothesis(mech, None, params)


def test_direct_may_carry_an_occluding_opening(interface):
    """H_D's interface is an opening, not an optical surface -- needed for matching."""
    assert direct_hypothesis(interface).interface is interface
    assert direct_hypothesis().interface is None


def test_missing_required_parameters_are_rejected(interface):
    with pytest.raises(ValueError, match="missing required parameters"):
        Hypothesis(OpticalMechanism.TRANSMISSION, interface, {"thickness": 0.01})
    with pytest.raises(ValueError, match="missing required parameters"):
        Hypothesis(OpticalMechanism.EMISSIVE, interface, {})


def test_parameter_ranges_are_validated(interface):
    with pytest.raises(ValueError, match="refractive_index"):
        glass_hypothesis(interface, refractive_index=0.5)
    with pytest.raises(ValueError, match="thickness"):
        glass_hypothesis(interface, thickness=-1.0)
    with pytest.raises(ValueError, match="display_mode"):
        display_hypothesis(interface, display_mode="holographic")
    with pytest.raises(ValueError, match="reflectance"):
        mixed_hypothesis(interface, reflectance=1.5)


def test_equality_and_hashing(interface):
    a = glass_hypothesis(interface, thickness=0.01, refractive_index=1.5)
    b = glass_hypothesis(interface, thickness=0.01, refractive_index=1.5)
    c = glass_hypothesis(interface, thickness=0.02, refractive_index=1.5)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert a != direct_hypothesis()
    assert a != "not a hypothesis"
    assert len({a, b, c}) == 2


def test_display_modes_are_distinct_hypotheses(interface):
    static = display_hypothesis(interface, display_mode="static")
    tracked = display_hypothesis(interface, display_mode="view_tracked")
    assert static != tracked


def test_serialisation_round_trip(interface):
    for h in (
        direct_hypothesis(interface),
        mirror_hypothesis(interface),
        display_hypothesis(interface, display_mode="view_tracked"),
        glass_hypothesis(interface, thickness=0.008, refractive_index=1.52),
        mixed_hypothesis(interface, reflectance=0.4),
    ):
        assert Hypothesis.from_dict(h.to_dict()) == h


def test_hypothesis_set_requires_two_and_unique_names(interface):
    with pytest.raises(ValueError, match="at least two"):
        HypothesisSet((direct_hypothesis(),))
    with pytest.raises(ValueError, match="unique"):
        HypothesisSet((direct_hypothesis(interface), mirror_hypothesis(interface, name="H_D")))


def test_hypothesis_set_indexing_and_prior(interface):
    hs = phase1_hypothesis_set(interface)
    assert hs.names == ["H_D", "H_E", "H_R"]
    assert hs.labels == ["Direct", "Display", "Mirror"]
    assert hs.index_of_name("H_R") == 2
    assert hs.index_of_mechanism("emissive") == 1
    assert np.allclose(hs.uniform_prior(), 1 / 3)
    assert hs.pairs() == [(0, 1), (0, 2), (1, 2)]
    with pytest.raises(KeyError):
        hs.index_of_name("H_X")
    assert HypothesisSet.from_dict(hs.to_dict()).names == hs.names
