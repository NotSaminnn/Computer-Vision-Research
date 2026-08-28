"""Transition models -- ``F_hat_{t+1} = W(F_t, H_k, a)``.

Four variants, all satisfying :class:`~intervene3d.models.interfaces.TransitionModel`:

``AnalyticalTransitionModel``
    Pure analytical optics (the default).
``NoHypothesisConditioningTransition``
    Ablation: every hypothesis is predicted with the *direct* transition, so the
    world model merely forecasts geometry and carries no causal information.
``HybridTransitionModel``
    Analytical optics plus a small learned residual (``analytical + learned``).
``LearnedOnlyTransitionModel``
    Ablation: the learned residual on top of the identity, i.e. no analytical
    optical prior at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from intervene3d.data.types import CHANNEL_CONTENT, GeometryFeature
from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.interventions.actions import Action
from intervene3d.models.learned_transition import ResidualMLP, build_inputs
from intervene3d.optics.registry import get_transition


@dataclass
class AnalyticalTransitionModel:
    """Dispatch to the analytical optical transition for the hypothesis' mechanism."""

    name: str = "analytical"

    def predict(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> GeometryFeature:
        return get_transition(hypothesis.mechanism).predict(
            state, hypothesis, action, markers_cam=markers_cam
        )

    def predict_contact_depth(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> np.ndarray:
        return get_transition(hypothesis.mechanism).predict_contact_depth(
            state, hypothesis, action, markers_cam=markers_cam
        )

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass
class NoHypothesisConditioningTransition:
    """Ablation: predict every hypothesis with the direct transition.

    The optical interface is retained (so aperture geometry still applies) but
    the *mechanism* is ignored.  If Intervene3D still worked under this ablation,
    the causal hypothesis conditioning would be doing no work.
    """

    name: str = "no_hypothesis_conditioning"

    def predict(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> GeometryFeature:
        direct_like = Hypothesis(OpticalMechanism.DIRECT, None, {}, "H_D")
        world = get_transition(OpticalMechanism.DIRECT).build_world(
            state, direct_like, markers_cam=markers_cam
        )
        # Keep the aperture so visibility is comparable across hypotheses.
        world = type(world)(
            content_points=world.content_points,
            reference_camera=world.reference_camera,
            interface=hypothesis.interface,
            markers_cam=world.markers_cam,
            reflects_observer=False,
            axial_shift=0.0,
            contact_on_interface=False,
            mechanism=OpticalMechanism.DIRECT,
        )
        from intervene3d.optics.base import project_world

        return project_world(world, action)

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass
class HybridTransitionModel:
    """Analytical optics + learned per-landmark residual."""

    mlp: ResidualMLP
    hypothesis_conditioning: bool = True
    residual_scale: float = 1.0
    name: str = "hybrid"
    _base: AnalyticalTransitionModel = field(default_factory=AnalyticalTransitionModel, repr=False)

    def predict(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> GeometryFeature:
        base = self._base.predict(state, hypothesis, action, markers_cam=markers_cam)
        return _apply_residual(
            base, state, hypothesis, action, self.mlp, self.hypothesis_conditioning, self.residual_scale
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "hypothesis_conditioning": self.hypothesis_conditioning,
            "residual_scale": self.residual_scale,
            "hidden_dim": self.mlp.hidden_dim,
        }


@dataclass
class LearnedOnlyTransitionModel:
    """Ablation: learned residual on top of the identity -- no optical prior.

    The base prediction copies the reference feature into the new camera's frame
    without any optics, so every bit of transition behaviour must come from the
    MLP.  Visibility is inherited from the reference view because a purely
    learned landmark predictor has no aperture model.
    """

    mlp: ResidualMLP
    hypothesis_conditioning: bool = True
    residual_scale: float = 1.0
    name: str = "learned_only"

    def predict(
        self,
        state: GeometryFeature,
        hypothesis: Hypothesis,
        action: Action,
        *,
        markers_cam: np.ndarray | None = None,
    ) -> GeometryFeature:
        cam = state.camera.moved(action.delta_T())
        base = GeometryFeature(state.uv.copy(), state.depth.copy(), state.visible.copy(), state.channel, cam)
        return _apply_residual(
            base, state, hypothesis, action, self.mlp, self.hypothesis_conditioning, self.residual_scale
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "hypothesis_conditioning": self.hypothesis_conditioning,
            "residual_scale": self.residual_scale,
            "hidden_dim": self.mlp.hidden_dim,
        }


def _apply_residual(
    base: GeometryFeature,
    state: GeometryFeature,
    hypothesis: Hypothesis,
    action: Action,
    mlp: ResidualMLP,
    hypothesis_conditioning: bool,
    scale: float,
) -> GeometryFeature:
    """Add the MLP residual to the content channel of ``base``."""
    x = build_inputs(state, hypothesis, action, hypothesis_conditioning=hypothesis_conditioning)
    if x.shape[0] == 0:
        return base
    residual = mlp.predict(x) * scale
    mask = base.channel == CHANNEL_CONTENT
    uv = base.uv.copy()
    depth = base.depth.copy()
    idx = np.nonzero(mask)[0]
    uv[idx, 0] = uv[idx, 0] + residual[:, 0]
    uv[idx, 1] = uv[idx, 1] + residual[:, 1]
    depth[idx] = depth[idx] + residual[:, 2]
    return GeometryFeature(uv, depth, base.visible, base.channel, base.camera)
