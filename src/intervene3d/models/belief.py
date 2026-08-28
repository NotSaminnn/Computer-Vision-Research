r"""Belief updating over optical hypotheses.

Prediction error for hypothesis ``k`` under the executed action ``a``:

.. math:: e_k = D\bigl(F_{t+1},\, \hat F_{t+1}^{k,a}\bigr)

Multiplicative update:

.. math:: p_{t+1}(H_k) \propto \exp(-\beta e_k)\, p_t(H_k)

Numerical stability
-------------------
The update is performed entirely in log space with a log-sum-exp normalisation,
and the errors are shifted by their minimum before exponentiation, so
``beta * e`` can be arbitrarily large without overflow.  A ``floor`` keeps every
posterior strictly positive, which prevents a single unlucky observation from
irrecoverably killing a hypothesis -- important because the whole point of the
system is to be able to report that two hypotheses remain competitive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def log_normalise(log_weights: np.ndarray) -> np.ndarray:
    """Stable ``softmax`` of log-weights."""
    log_weights = np.asarray(log_weights, dtype=np.float64)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return np.full(log_weights.shape, 1.0 / log_weights.size)
    m = np.max(log_weights[finite])
    shifted = np.where(finite, log_weights - m, -np.inf)
    w = np.exp(shifted)
    total = float(np.sum(w))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(log_weights.shape, 1.0 / log_weights.size)
    return w / total


@dataclass
class LikelihoodBeliefUpdater:
    """Exponential-error likelihood update.

    Parameters
    ----------
    beta:
        Inverse temperature, in units of ``1 / D``.  With the default distance
        (pixels) ``beta = 1.0`` means a one-pixel prediction error costs one nat.
    floor:
        Minimum posterior mass per hypothesis after normalisation.
    """

    beta: float = 1.0
    floor: float = 1e-6
    name: str = "likelihood"

    def __post_init__(self) -> None:
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if not 0.0 <= self.floor < 1.0:
            raise ValueError("floor must lie in [0, 1)")

    def log_likelihood(self, errors: np.ndarray) -> np.ndarray:
        """``-beta e_k``, shifted so the best hypothesis has log-likelihood 0."""
        e = np.asarray(errors, dtype=np.float64)
        e = np.where(np.isfinite(e), e, np.max(e[np.isfinite(e)]) if np.any(np.isfinite(e)) else 0.0)
        return -self.beta * (e - np.min(e))

    def update(self, prior: np.ndarray, errors: np.ndarray) -> np.ndarray:
        """One Bayesian step.  ``prior`` need not be normalised."""
        prior = np.asarray(prior, dtype=np.float64)
        errors = np.asarray(errors, dtype=np.float64)
        if prior.shape != errors.shape:
            raise ValueError(f"prior {prior.shape} and errors {errors.shape} must match")
        if np.any(prior < 0.0):
            raise ValueError("prior probabilities must be non-negative")

        with np.errstate(divide="ignore"):
            log_prior = np.log(np.clip(prior, 1e-300, None))
        posterior = log_normalise(log_prior + self.log_likelihood(errors))

        if self.floor > 0.0:
            k = posterior.size
            if self.floor * k >= 1.0:  # pragma: no cover - guarded by __post_init__ in practice
                return np.full(k, 1.0 / k)
            posterior = self.floor + (1.0 - self.floor * k) * posterior
        return posterior / float(np.sum(posterior))


def posterior_entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats; 0 for a point mass, ``log K`` for uniform."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def normalised_entropy(p: np.ndarray) -> float:
    """Entropy scaled to ``[0, 1]``; the *prediction* uncertainty ``U_prediction``."""
    k = int(np.asarray(p).size)
    if k <= 1:
        return 0.0
    return float(posterior_entropy(p) / np.log(k))
