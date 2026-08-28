r"""Intervention selection -- which visual experiment to run.

Strategies (the ``interventions`` axis of the ablation framework):

``max_separability`` (Intervene3D)
    ``a* = argmax_a  sum_{i<j} p_i p_j Delta_ij(a)`` -- belief-weighted expected
    hypothesis separability.  The distinguishing feature is *what* the
    information is about: the competing optical explanation, not the state.
``entropy_nbv``
    A deliberately generic next-best-view proxy: pick the action producing the
    largest predicted change from the reference view under the current MAP
    hypothesis, i.e. "look where the scene changes most".  It is hypothesis-
    *blind*, which is exactly the property the ablation tests.  It is a proxy,
    not a reimplementation of any published NBV method.
``random`` / ``max_baseline`` / ``fixed`` / ``null``
    Control strategies: an arbitrary allowed action, the largest allowed
    baseline, one configured action (the passive multi-view baseline), and no
    action at all (the single-frame baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from intervene3d.data.types import GeometryFeature
from intervene3d.hypotheses.base import HypothesisSet
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.interventions.actions import Action, null_action
from intervene3d.models.separability import GeometrySeparabilityEstimator, feature_distance


def _pair_weighted_utility(sep: np.ndarray, beliefs: np.ndarray) -> np.ndarray:
    """``sum_{i<j} p_i p_j Delta_ij(a)`` for each action."""
    p = np.asarray(beliefs, dtype=np.float64)
    w = np.outer(p, p)
    np.fill_diagonal(w, 0.0)
    w = w / 2.0  # count each unordered pair once
    return np.einsum("akl,kl->a", sep, w)


@dataclass
class MaxSeparabilitySelector:
    """Intervene3D's own selector."""

    estimator: GeometrySeparabilityEstimator
    allow_null: bool = False
    name: str = "max_separability"

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        sep = self.estimator.pairwise_over_actions(state, hypotheses, actions, markers_cam=markers_cam)
        utility = _pair_weighted_utility(sep, beliefs)
        return _argmax_action(actions, utility, self.allow_null), utility


@dataclass
class EntropyNBVSelector:
    """Hypothesis-blind "largest predicted change" next-best-view proxy."""

    estimator: GeometrySeparabilityEstimator
    allow_null: bool = False
    name: str = "entropy_nbv"

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        map_idx = int(np.argmax(beliefs))
        h = hypotheses[map_idx]
        utility = np.zeros(len(actions))
        for a_idx, action in enumerate(actions):
            pred = self.estimator.transition.predict(state, h, action, markers_cam=markers_cam)
            utility[a_idx] = feature_distance(pred, state, self.estimator.weights).total
        return _argmax_action(actions, utility, self.allow_null), utility


@dataclass
class RandomSelector:
    """Uniformly random allowed action."""

    seed: int = 0
    allow_null: bool = False
    name: str = "random"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        pool = list(actions) if self.allow_null else actions.non_null()
        idx = int(self._rng.integers(0, len(pool)))
        chosen = pool[idx]
        utility = np.array([1.0 if a.name == chosen.name else 0.0 for a in actions])
        return chosen, utility


@dataclass
class MaxBaselineSelector:
    """Always take the largest allowed translation."""

    name: str = "max_baseline"

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        utility = np.array([a.translation_magnitude for a in actions])
        return actions[int(np.argmax(utility))], utility


@dataclass
class FixedActionSelector:
    """Always execute one configured action -- the passive multi-view control."""

    action_name: str
    name: str = "fixed"

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        idx = actions.index_of(self.action_name)
        utility = np.zeros(len(actions))
        utility[idx] = 1.0
        return actions[idx], utility


@dataclass
class NullSelector:
    """Never move -- the single-frame control."""

    name: str = "null"

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]:
        return null_action(), np.zeros(len(actions))


def _argmax_action(actions: ActionSpace, utility: np.ndarray, allow_null: bool) -> Action:
    """Argmax with deterministic tie-breaking (lowest index wins)."""
    util = np.asarray(utility, dtype=np.float64).copy()
    if not allow_null:
        for i, a in enumerate(actions):
            if a.is_null:
                util[i] = -np.inf
    if not np.any(np.isfinite(util)):  # pragma: no cover - action space always has a non-null action
        return actions[0]
    return actions[int(np.argmax(util))]


def build_selector(
    name: str,
    estimator: GeometrySeparabilityEstimator,
    *,
    seed: int = 0,
    fixed_action: str | None = None,
    allow_null: bool = False,
):
    """Factory used by the baseline / ablation framework."""
    if name == "max_separability":
        return MaxSeparabilitySelector(estimator, allow_null=allow_null)
    if name == "entropy_nbv":
        return EntropyNBVSelector(estimator, allow_null=allow_null)
    if name == "random":
        return RandomSelector(seed=seed, allow_null=allow_null)
    if name == "max_baseline":
        return MaxBaselineSelector()
    if name == "fixed":
        if not fixed_action:
            raise ValueError("selector 'fixed' requires fixed_action")
        return FixedActionSelector(fixed_action)
    if name == "null":
        return NullSelector()
    raise ValueError(f"unknown selector {name!r}")
