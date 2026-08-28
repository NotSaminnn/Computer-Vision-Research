"""Belief update: normalisation, numerical stability, posterior ordering."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.models.belief import (
    LikelihoodBeliefUpdater,
    log_normalise,
    normalised_entropy,
    posterior_entropy,
)


def test_posterior_is_normalised():
    up = LikelihoodBeliefUpdater(beta=1.0)
    for errors in ([0.0, 1.0, 2.0], [5.0, 5.0, 5.0], [0.0, 100.0, 200.0]):
        p = up.update(np.array([1 / 3, 1 / 3, 1 / 3]), np.array(errors))
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p > 0)


def test_lower_error_yields_higher_posterior():
    up = LikelihoodBeliefUpdater(beta=1.0, floor=0.0)
    p = up.update(np.array([1 / 3, 1 / 3, 1 / 3]), np.array([3.0, 1.0, 2.0]))
    assert np.argmax(p) == 1
    assert p[1] > p[2] > p[0]


def test_equal_errors_preserve_the_prior():
    up = LikelihoodBeliefUpdater(beta=2.0, floor=0.0)
    prior = np.array([0.6, 0.3, 0.1])
    assert np.allclose(up.update(prior, np.array([4.0, 4.0, 4.0])), prior)


def test_non_identifiable_pair_stays_balanced():
    """Two hypotheses predicting identically must remain equally probable."""
    up = LikelihoodBeliefUpdater(beta=1.0, floor=0.0)
    p = up.update(np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.5, 0.5, 40.0]))
    assert np.isclose(p[0], p[1])
    assert p[2] < 1e-10
    assert np.isclose(p[0], 0.5, atol=1e-9)


def test_numerical_stability_with_huge_errors():
    up = LikelihoodBeliefUpdater(beta=50.0)
    p = up.update(np.array([0.5, 0.5]), np.array([0.0, 1e6]))
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1.0)
    assert p[0] > p[1]


def test_non_finite_errors_do_not_produce_nan():
    up = LikelihoodBeliefUpdater(beta=1.0)
    p = up.update(np.array([0.5, 0.5]), np.array([1.0, np.nan]))
    assert np.all(np.isfinite(p)) and np.isclose(p.sum(), 1.0)


def test_floor_keeps_every_hypothesis_alive():
    """A hypothesis must stay recoverable: the system has to be able to say 'still competitive'."""
    up = LikelihoodBeliefUpdater(beta=10.0, floor=1e-4)
    p = up.update(np.array([0.5, 0.5]), np.array([0.0, 1e4]))
    assert p[1] >= 1e-4


def test_updates_compose_multiplicatively():
    up = LikelihoodBeliefUpdater(beta=1.0, floor=0.0)
    prior = np.array([0.5, 0.5])
    once = up.update(up.update(prior, np.array([0.0, 1.0])), np.array([0.0, 1.0]))
    twice = up.update(prior, np.array([0.0, 2.0]))
    assert np.allclose(once, twice, atol=1e-9)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        LikelihoodBeliefUpdater(beta=-1.0)
    with pytest.raises(ValueError):
        LikelihoodBeliefUpdater(floor=1.5)
    up = LikelihoodBeliefUpdater()
    with pytest.raises(ValueError, match="must match"):
        up.update(np.array([0.5, 0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        up.update(np.array([-0.5, 1.5]), np.array([1.0, 1.0]))


def test_beta_zero_ignores_the_evidence():
    up = LikelihoodBeliefUpdater(beta=0.0, floor=0.0)
    prior = np.array([0.7, 0.3])
    assert np.allclose(up.update(prior, np.array([0.0, 100.0])), prior)


def test_log_normalise_handles_all_infinite_input():
    p = log_normalise(np.array([-np.inf, -np.inf]))
    assert np.allclose(p, 0.5)


def test_entropy_bounds():
    assert np.isclose(posterior_entropy(np.array([1.0, 0.0, 0.0])), 0.0)
    assert np.isclose(posterior_entropy(np.full(4, 0.25)), np.log(4))
    assert np.isclose(normalised_entropy(np.full(4, 0.25)), 1.0)
    assert np.isclose(normalised_entropy(np.array([1.0, 0.0])), 0.0)
    assert normalised_entropy(np.array([1.0])) == 0.0
