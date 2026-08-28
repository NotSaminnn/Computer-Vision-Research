"""Abstract model interfaces.

The preliminary system is deliberately assembled from six small, swappable
components rather than one large network:

===========================  ==========================================================
``GeometryEncoder``          image / scene -> ``F_t``
``TransitionModel``          ``(F_t, H_k, a) -> F_hat_{t+1}``
``SeparabilityEstimator``    ``(F_hat^i, F_hat^j) -> Delta_ij``
``BeliefUpdater``            ``(p_t, e_k) -> p_{t+1}``
``InterventionSelector``     ``(beliefs, Delta(a)) -> a*``
``IdentifiabilityEstimator`` ``Delta matrix -> I_A, resolvable``
===========================  ==========================================================

Everything downstream of these interfaces (experiments, metrics, figures) is
written against the protocols, so replacing the analytical transition with a
learned one, or the mock encoder with MoGe/VGGT, requires no restructuring.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from intervene3d.data.types import GeometryFeature, Observation
from intervene3d.hypotheses.base import Hypothesis, HypothesisSet
from intervene3d.interventions.action_space import ActionSpace
from intervene3d.interventions.actions import Action


@runtime_checkable
class GeometryEncoder(Protocol):
    """``F_t = E(I_t)`` -- turn an observation into the geometry representation."""

    name: str

    def encode(self, observation: Observation) -> GeometryFeature: ...


@runtime_checkable
class TransitionModel(Protocol):
    """``F_hat_{t+1} = W(F_t, H_k, a)`` -- hypothesis-conditioned prediction."""

    name: str

    def predict(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> GeometryFeature: ...


@runtime_checkable
class SeparabilityEstimator(Protocol):
    """``Delta_ij(a) = D(p(O'|H_i,a), p(O'|H_j,a))``."""

    name: str

    def pairwise(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> np.ndarray: ...


@runtime_checkable
class BeliefUpdater(Protocol):
    """``p_{t+1}(H_k) ~ exp(-beta e_k) p_t(H_k)``."""

    name: str

    def update(self, prior: np.ndarray, errors: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class InterventionSelector(Protocol):
    """Choose which visual experiment to run."""

    name: str

    def select(
        self,
        state: GeometryFeature,
        hypotheses: HypothesisSet,
        actions: ActionSpace,
        beliefs: np.ndarray,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> tuple[Action, np.ndarray]: ...


@runtime_checkable
class IdentifiabilityEstimator(Protocol):
    """``I_A(H_i,H_j) = max_a Delta_ij(a)`` plus the epsilon decision."""

    name: str
    epsilon: float

    def evaluate(self, separability_by_action: np.ndarray) -> tuple[np.ndarray, float, bool]: ...
