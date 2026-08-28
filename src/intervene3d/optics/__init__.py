"""Analytical optical transition layer.

    transition = get_transition(hypothesis.mechanism)
    F_next = transition.predict(state=F_t, hypothesis=H_k, action=a)

Physics stays explicit and inspectable; the learned component (see
:mod:`intervene3d.models.learned_transition`) is a *residual* on top of this,
never a replacement for it.
"""

from intervene3d.optics.base import (
    HypothesisWorld,
    OpticalTransition,
    contact_depth_for_world,
    project_world,
    reference_rays,
)
from intervene3d.optics.direct import DirectTransition
from intervene3d.optics.display import DisplayTransition, screen_points, static_display_homography
from intervene3d.optics.mirror import MirrorTransition, virtual_camera_pose
from intervene3d.optics.mixed import MixedTransition
from intervene3d.optics.registry import available_mechanisms, get_transition, transition_for
from intervene3d.optics.transmission import (
    TransmissionTransition,
    paraxial_axial_shift,
    paraxial_validity_angle,
    slab_lateral_displacement,
)

__all__ = [
    "HypothesisWorld",
    "OpticalTransition",
    "contact_depth_for_world",
    "project_world",
    "reference_rays",
    "DirectTransition",
    "DisplayTransition",
    "MirrorTransition",
    "MixedTransition",
    "TransmissionTransition",
    "screen_points",
    "static_display_homography",
    "virtual_camera_pose",
    "paraxial_axial_shift",
    "paraxial_validity_angle",
    "slab_lateral_displacement",
    "available_mechanisms",
    "get_transition",
    "transition_for",
]
