"""Causal variants of a base scene, and the simulator that renders them.

Every variant shares one base scene and one optical interface pose, so the
reference view is *pixel-identical* across variants by construction.  Only the
physical mechanism -- and hence the response to intervention and the contact
geometry -- differs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from intervene3d.data.types import CHANNEL_CONTENT, GeometryFeature, Observation, SceneContent
from intervene3d.geometry.planes import Aperture
from intervene3d.hypotheses.base import Hypothesis, HypothesisSet, OpticalMechanism
from intervene3d.hypotheses.families import (
    direct_hypothesis,
    display_hypothesis,
    glass_hypothesis,
    mirror_hypothesis,
    mixed_hypothesis,
)
from intervene3d.interventions.actions import Action, null_action
from intervene3d.optics.base import contact_depth_for_world, project_world
from intervene3d.optics.registry import get_transition


def build_hypothesis(
    mechanism: str,
    interface: Aperture,
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    display_mode: str | None = None,
) -> Hypothesis:
    """Instantiate the hypothesis for one mechanism using the generator config."""
    mech = OpticalMechanism(mechanism)
    if mech is OpticalMechanism.DIRECT:
        return direct_hypothesis(interface)
    if mech is OpticalMechanism.REFLECTION:
        return mirror_hypothesis(interface)
    if mech is OpticalMechanism.EMISSIVE:
        return display_hypothesis(interface, display_mode=display_mode or config["display"]["mode"])
    if mech is OpticalMechanism.TRANSMISSION:
        g = config["glass"]
        return glass_hypothesis(
            interface,
            thickness=float(rng.uniform(*g["thickness"])),
            refractive_index=float(rng.uniform(*g["refractive_index"])),
        )
    if mech is OpticalMechanism.MIXED:
        g = config["glass"]
        return mixed_hypothesis(
            interface,
            reflectance=0.35,
            thickness=float(rng.uniform(*g["thickness"])),
            refractive_index=float(rng.uniform(*g["refractive_index"])),
        )
    raise ValueError(f"unsupported mechanism {mechanism!r}")  # pragma: no cover


def build_hypothesis_set(
    interface: Aperture, config: dict[str, Any], rng: np.random.Generator
) -> HypothesisSet:
    """The competing hypothesis family ``H`` for one base scene.

    The *same* parameter draw is used for the hypothesis the agent reasons with
    and for the variant the simulator renders, so a mechanism is never rejected
    merely because the agent guessed the wrong slab thickness.  Estimating those
    interface parameters is explicitly out of scope for the preliminary codebase
    (see ``docs/RESEARCH_SPEC_AUDIT.md``, "assumptions").
    """
    prob = float(config["display"].get("view_tracked_probability", 0.0))
    display_mode = "view_tracked" if rng.random() < prob else config["display"]["mode"]
    return HypothesisSet(
        tuple(build_hypothesis(m, interface, config, rng, display_mode=display_mode) for m in config["mechanisms"])
    )


def reference_observation(content: SceneContent, hypothesis: Hypothesis) -> Observation:
    """Render the reference view ``C_0`` for one causal variant.

    The image geometry is rendered from the variant's **own** world, so the
    simulator stays physically consistent: if a mirror would show the observer's
    reflection at ``C_0``, this function shows it.  The matched-counterfactual
    property is therefore *earned*, not imposed -- the scene generator rejects any
    marker configuration whose virtual image is visible at the reference view
    (see :func:`~intervene3d.data.synthetic.scene_generator.generate_base_scene`),
    exactly as a careful experimenter would choose a reference viewpoint from
    which their own reflection is not in frame.

    The one quantity that *is* imposed is the perceived **depth** of the content,
    which is set to the apparent (direct) reading for every mechanism.  That
    encodes the empirical finding that a single-frame geometry model reports the
    illusory depth: shown a corridor on a screen it sees the corridor, not the
    screen plane.  Without this the depth channel alone would betray a display to
    a single-frame classifier, and the premise would be an artefact.  The true
    surface is available only as ``contact_depth``, which is ground truth.
    """
    seed = _seed_feature(content)
    direct = direct_hypothesis(content.interface)
    direct_world = get_transition(OpticalMechanism.DIRECT).build_world(
        seed, direct, markers_cam=content.observer_markers_cam
    )
    apparent = project_world(direct_world, null_action())

    true_world = get_transition(hypothesis.mechanism).build_world(
        apparent, hypothesis, markers_cam=content.observer_markers_cam
    )
    rendered = project_world(true_world, null_action())

    content_mask = rendered.channel == CHANNEL_CONTENT
    depth = rendered.depth.copy()
    depth[content_mask] = apparent.depth[content_mask]
    feature = GeometryFeature(rendered.uv, depth, rendered.visible, rendered.channel, rendered.camera)

    contact = contact_depth_for_world(true_world, null_action())
    return Observation(feature=feature, contact_depth=contact, mechanism=hypothesis.mechanism.value)


def observer_reflection_visible_at_reference(content: SceneContent) -> bool:
    """Would a mirror on this interface show the observer's own reflection at ``C_0``?

    Used by the generator to reject such scenes: if the reflection is already in
    frame, the variants are not matched at the reference view and a single-frame
    classifier could separate them without any intervention.
    """
    if content.n_markers == 0:
        return False
    from intervene3d.data.types import CHANNEL_MARKER
    from intervene3d.hypotheses.families import mirror_hypothesis

    seed = _seed_feature(content)
    mirror = mirror_hypothesis(content.interface)
    world = get_transition(mirror.mechanism).build_world(
        seed, mirror, markers_cam=content.observer_markers_cam
    )
    feature = project_world(world, null_action())
    return bool(np.any(feature.visible[feature.channel == CHANNEL_MARKER]))


def simulate(
    content: SceneContent,
    hypothesis: Hypothesis,
    action: Action,
    *,
    reference_feature: GeometryFeature | None = None,
) -> Observation:
    """Render the observation obtained by executing ``action`` from ``C_0``.

    The simulator uses the same analytical optics as the transition model.  That
    validates the inference pipeline, **not** the optics against an independent
    renderer -- see ``docs/RESEARCH_SPEC_AUDIT.md`` for this limitation and the
    two independent cross-checks (mirror virtual camera, display homography) that
    do validate the optics.
    """
    ref = reference_feature if reference_feature is not None else reference_observation(content, hypothesis).feature
    world = get_transition(hypothesis.mechanism).build_world(
        ref, hypothesis, markers_cam=content.observer_markers_cam
    )
    feature = project_world(world, action)
    contact = contact_depth_for_world(world, action)
    return Observation(feature=feature, contact_depth=contact, mechanism=hypothesis.mechanism.value)


def _seed_feature(content: SceneContent) -> GeometryFeature:
    """A bootstrap feature holding the true apparent content, used to seed world building."""
    cam = content.reference_camera
    uv_c, z_c, vis_c = cam.project(content.points)
    corners = content.interface.corners()
    uv_f, z_f, vis_f = cam.project(corners)
    k = content.n_markers
    uv = np.concatenate([uv_c, uv_f, np.full((k, 2), np.nan)], axis=0)
    depth = np.concatenate([z_c, z_f, np.full(k, np.nan)])
    visible = np.concatenate([vis_c, vis_f, np.zeros(k, dtype=bool)])
    return GeometryFeature(uv, depth, visible, content.channel_vector(), cam)


def content_depth_range(observation: Observation) -> tuple[float, float]:
    """``(Z_min, Z_max)`` over visible content landmarks -- the MCRB inputs."""
    f = observation.feature
    m = (f.channel == CHANNEL_CONTENT) & f.visible & np.isfinite(f.depth)
    if not np.any(m):
        return float("nan"), float("nan")
    d = f.depth[m]
    return float(np.min(d)), float(np.max(d))
