"""Mechanism -> transition lookup."""

from __future__ import annotations

from intervene3d.hypotheses.base import Hypothesis, OpticalMechanism
from intervene3d.optics.base import OpticalTransition
from intervene3d.optics.direct import DirectTransition
from intervene3d.optics.display import DisplayTransition
from intervene3d.optics.mirror import MirrorTransition
from intervene3d.optics.mixed import MixedTransition
from intervene3d.optics.transmission import TransmissionTransition

_TRANSITIONS: dict[OpticalMechanism, OpticalTransition] = {
    OpticalMechanism.DIRECT: DirectTransition(),
    OpticalMechanism.REFLECTION: MirrorTransition(),
    OpticalMechanism.EMISSIVE: DisplayTransition(),
    OpticalMechanism.TRANSMISSION: TransmissionTransition(),
    OpticalMechanism.MIXED: MixedTransition(),
}


def get_transition(mechanism: OpticalMechanism | str) -> OpticalTransition:
    """Return the analytical transition implementing ``mechanism``."""
    return _TRANSITIONS[OpticalMechanism(mechanism)]


def transition_for(hypothesis: Hypothesis) -> OpticalTransition:
    return get_transition(hypothesis.mechanism)


def available_mechanisms() -> list[OpticalMechanism]:
    return list(_TRANSITIONS)
