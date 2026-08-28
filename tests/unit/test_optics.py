"""Analytical optics.

The two most important tests here validate each mechanism against an
*independent derivation*, not against itself:

* the planar mirror, via the virtual-camera formulation;
* the static display, via the plane-induced homography.
"""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.data.types import CHANNEL_CONTENT
from intervene3d.geometry.camera import Camera, CameraIntrinsics, look_at, project_points
from intervene3d.geometry.planes import (
    Aperture,
    Plane,
    apply_homography,
    fit_homography_dlt,
    homography_residual,
    reflect_points,
    reflect_pose,
    signed_distance,
)
from intervene3d.geometry.se3 import is_rotation
from intervene3d.hypotheses.families import (
    direct_hypothesis,
    display_hypothesis,
    glass_hypothesis,
    mirror_hypothesis,
)
from intervene3d.interventions.actions import Action, ActionKind, null_action
from intervene3d.optics import (
    get_transition,
    paraxial_axial_shift,
    paraxial_validity_angle,
    slab_lateral_displacement,
    static_display_homography,
)

INTR = CameraIntrinsics(fx=250.0, fy=250.0, cx=160.0, cy=120.0, width=320, height=240)


# ------------------------------------------------------------------- planes
def test_signed_distance_and_reflection_are_consistent():
    plane = Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    pts = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.5]])
    d = signed_distance(plane, pts)
    reflected = reflect_points(plane, pts)
    assert np.allclose(signed_distance(plane, reflected), -d)
    assert np.allclose(reflect_points(plane, reflected), pts)  # involution


def test_reflect_pose_stays_a_proper_rotation():
    plane = Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    T = look_at([0.1, -0.2, 0.0], [0.0, 0.0, 2.0])
    virtual = reflect_pose(plane, T)
    assert is_rotation(virtual[:3, :3])
    # The virtual camera centre is the mirror image of the real one.
    assert np.allclose(virtual[:3, 3], reflect_points(plane, T[:3, 3][None, :])[0])


def test_mirror_virtual_camera_matches_virtual_point_formulation():
    """Two independent derivations of planar-mirror optics must agree.

    Formulation A (used by the transition): project the *virtual* points with the
    *real* camera.
    Formulation B (independent): project the *physical* points with the
    *reflected* camera, then undo the horizontal image flip a mirror introduces.
    """
    plane = Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    cam = Camera(INTR, look_at([0.0, 0.0, 0.0], [0.0, 0.0, 2.0]))
    rng = np.random.default_rng(0)
    virtual_points = np.stack(
        [rng.uniform(-0.5, 0.5, 40), rng.uniform(-0.4, 0.4, 40), rng.uniform(2.6, 5.0, 40)], axis=1
    )
    physical_points = reflect_points(plane, virtual_points)

    uv_a, depth_a, _ = cam.project(virtual_points)

    virtual_cam = Camera(INTR, reflect_pose(plane, cam.T_wc))
    pts_cam = virtual_cam.world_to_camera(physical_points)
    pts_cam[:, 0] *= -1.0  # undo the mirror's left-right image flip
    uv_b, depth_b, _ = project_points(INTR, pts_cam)

    assert np.allclose(uv_a, uv_b, atol=1e-9)
    assert np.allclose(depth_a, depth_b, atol=1e-9)


def test_static_display_matches_the_plane_induced_homography():
    """The static-display transition must equal a plane-induced homography."""
    from intervene3d.config import validate_synthetic_config
    from intervene3d.data.synthetic import (
        build_hypothesis_set,
        generate_base_scene,
        reference_observation,
    )

    cfg = validate_synthetic_config({"scene": {"num_content_landmarks": 20}})
    base = generate_base_scene(np.random.default_rng([9, 1]), cfg, 0)
    hyps = build_hypothesis_set(base.content.interface, cfg, np.random.default_rng([9, 1, 7]))
    ref = reference_observation(base.content, hyps[0]).feature

    display = display_hypothesis(base.content.interface, display_mode="static")
    action = Action("lateral", ActionKind.TRANSLATE_X, np.array([0.12, 0.0, 0.0]), np.zeros(3))
    predicted = get_transition(display.mechanism).predict(
        ref, display, action, markers_cam=base.content.observer_markers_cam
    )

    H = static_display_homography(ref, base.content.interface, action)
    m = (ref.channel == CHANNEL_CONTENT) & ref.visible & predicted.visible
    warped = apply_homography(H, ref.uv[m])
    assert np.allclose(warped, predicted.uv[m], atol=1e-6)


def test_homography_fit_recovers_a_known_homography():
    rng = np.random.default_rng(3)
    src = rng.uniform(0, 320, (16, 2))
    H = np.array([[1.08, 0.04, 9.0], [-0.02, 0.96, -5.0], [1e-4, 3e-5, 1.0]])
    dst = apply_homography(H, src)
    assert np.max(np.abs(apply_homography(fit_homography_dlt(src, dst), src) - dst)) < 1e-8
    assert homography_residual(src, dst) < 1e-8


def test_homography_fit_needs_four_points():
    with pytest.raises(ValueError):
        fit_homography_dlt(np.zeros((3, 2)), np.zeros((3, 2)))


# ------------------------------------------------------------------ aperture
def test_aperture_contains_and_corners():
    plane = Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    ap = Aperture.from_plane(plane, 0.5, 0.3)
    assert np.allclose(signed_distance(plane, ap.corners()), 0.0, atol=1e-12)
    assert ap.contains(ap.corners()).all()
    outside = ap.plane.point + 1.5 * ap.u_axis
    assert not ap.contains(outside[None, :])[0]
    assert np.isclose(ap.area, 4 * 0.5 * 0.3)


# -------------------------------------------------------------- transmission
def test_paraxial_axial_shift_formula():
    assert np.isclose(paraxial_axial_shift(0.012, 1.5), 0.012 * (1 - 1 / 1.5))
    assert paraxial_axial_shift(0.02, 1.0) == 0.0  # no index contrast, no shift


def test_paraxial_axial_shift_rejects_invalid_index():
    with pytest.raises(ValueError):
        paraxial_axial_shift(0.01, 0.9)


def test_slab_lateral_displacement_limits():
    # Normal incidence produces no lateral displacement.
    assert np.isclose(float(slab_lateral_displacement(0.01, 1.5, 0.0)), 0.0)
    # Displacement grows with incidence angle over the usable range.
    angles = np.radians([5.0, 20.0, 40.0, 60.0])
    s = slab_lateral_displacement(0.01, 1.5, angles)
    assert np.all(np.diff(np.abs(s)) > 0)


def test_paraxial_validity_angle_is_reported_and_finite():
    angle = paraxial_validity_angle(1.5, relative_tolerance=0.05)
    assert 0.0 < angle < np.pi / 2
    # A tighter tolerance must give a smaller validity range.
    assert paraxial_validity_angle(1.5, relative_tolerance=0.01) < angle


def test_transmission_reduces_to_direct_when_there_is_no_shift(base_scene, reference_feature):
    interface = base_scene.content.interface
    glass = glass_hypothesis(interface, thickness=0.0, refractive_index=1.5)
    direct = direct_hypothesis(interface)
    action = Action("lateral", ActionKind.TRANSLATE_X, np.array([0.2, 0.0, 0.0]), np.zeros(3))
    markers = base_scene.content.observer_markers_cam
    a = get_transition(glass.mechanism).predict(reference_feature, glass, action, markers_cam=markers)
    b = get_transition(direct.mechanism).predict(reference_feature, direct, action, markers_cam=markers)
    m = a.visible & b.visible
    assert np.allclose(a.uv[m], b.uv[m], atol=1e-9)


# ------------------------------------------------- matched-counterfactual core
def test_all_mechanisms_are_pixel_identical_at_the_reference_view(base_scene, hypotheses):
    from intervene3d.data.synthetic import reference_observation

    refs = [reference_observation(base_scene.content, h) for h in hypotheses]
    f0 = refs[0].feature
    for r in refs[1:]:
        assert np.array_equal(r.feature.visible, f0.visible)
        assert np.allclose(r.feature.uv[f0.visible], f0.uv[f0.visible], atol=1e-12)


def test_contact_geometry_differs_by_mechanism(base_scene, hypotheses):
    """Same appearance, different first physical surface."""
    from intervene3d.data.synthetic import reference_observation

    contacts = {}
    for h in hypotheses:
        obs = reference_observation(base_scene.content, h)
        m = (obs.feature.channel == CHANNEL_CONTENT) & obs.feature.visible
        contacts[h.mechanism.value] = float(np.nanmean(obs.contact_depth[m]))
    # A display and a mirror put the contact surface on the interface plane;
    # a direct scene puts it on the content, which is further away.
    assert contacts["direct"] > contacts["emissive"] + 1e-6
    assert contacts["direct"] > contacts["reflection"] + 1e-6
    assert np.isclose(contacts["emissive"], contacts["reflection"], rtol=1e-6)


def test_null_action_reproduces_the_reference_view(base_scene, hypotheses, reference_feature):
    markers = base_scene.content.observer_markers_cam
    direct = [h for h in hypotheses if h.mechanism.value == "direct"][0]
    predicted = get_transition(direct.mechanism).predict(
        reference_feature, direct, null_action(), markers_cam=markers
    )
    m = reference_feature.visible & predicted.visible
    assert np.allclose(predicted.uv[m], reference_feature.uv[m], atol=1e-9)


def test_mirror_predicts_moving_virtual_observer_markers(base_scene, hypotheses, reference_feature):
    """The mirror's decisive cue: virtual images of observer-attached structure move."""
    from intervene3d.data.types import CHANNEL_MARKER

    interface = base_scene.content.interface
    mirror = mirror_hypothesis(interface)
    markers = np.array([[0.05, 0.0, -0.02], [-0.05, 0.02, -0.02]])
    a1 = Action("a1", ActionKind.TRANSLATE_X, np.array([0.0, 0.0, 0.0]), np.zeros(3))
    a2 = Action("a2", ActionKind.TRANSLATE_X, np.array([0.15, 0.0, 0.0]), np.zeros(3))
    t = get_transition(mirror.mechanism)
    f1 = t.predict(reference_feature, mirror, a1, markers_cam=markers)
    f2 = t.predict(reference_feature, mirror, a2, markers_cam=markers)
    mm = (f1.channel == CHANNEL_MARKER)
    assert mm.sum() == 2
    # A direct hypothesis predicts no such landmark at all.
    direct = direct_hypothesis(interface)
    fd = get_transition(direct.mechanism).predict(reference_feature, direct, a2, markers_cam=markers)
    assert not fd.visible[fd.channel == CHANNEL_MARKER].any()
    # If the mirror does show them, they must move with the observer.
    both = f1.visible[mm] & f2.visible[mm]
    if both.any():
        assert np.max(np.abs(f1.uv[mm][both] - f2.uv[mm][both])) > 1e-6


def test_no_scene_reveals_a_mirror_at_the_reference_view(synth_config):
    """The matched property must be *earned* by geometry, not imposed.

    If the observer's reflection were already in frame at C_0, a mirror would be
    recognisable with no intervention and the benchmark's premise would collapse.
    The generator rejection-samples the marker rig to prevent it.
    """
    from intervene3d.data.synthetic import (
        build_hypothesis_set,
        generate_base_scene,
        reference_observation,
    )
    from intervene3d.data.synthetic.optical_variants import observer_reflection_visible_at_reference

    for b in range(12):
        base = generate_base_scene(np.random.default_rng([31, b]), synth_config, b)
        assert not observer_reflection_visible_at_reference(base.content), (
            f"base scene {b}: the observer's reflection is visible at C_0"
        )
        hyps = build_hypothesis_set(base.content.interface, synth_config, np.random.default_rng([31, b, 7]))
        refs = [reference_observation(base.content, h) for h in hyps]
        f0 = refs[0].feature
        for r in refs[1:]:
            assert np.array_equal(r.feature.visible, f0.visible)
            assert np.allclose(r.feature.uv[f0.visible], f0.uv[f0.visible], atol=1e-12)
            # The perceived depth is the illusory one for every mechanism, so the
            # depth channel cannot betray a display to a single-frame classifier.
            m = (f0.channel == CHANNEL_CONTENT) & f0.visible
            assert np.allclose(r.feature.depth[m], f0.depth[m], atol=1e-12)


def test_separability_is_exactly_zero_under_the_null_action_for_every_scene(synth_config):
    """No intervention, no evidence -- for every scene, not just a lucky one."""
    from intervene3d.data.synthetic import (
        build_hypothesis_set,
        generate_base_scene,
        reference_observation,
    )
    from intervene3d.interventions.actions import null_action
    from intervene3d.models.separability import DistanceWeights, GeometrySeparabilityEstimator
    from intervene3d.models.transition import AnalyticalTransitionModel

    est = GeometrySeparabilityEstimator(
        AnalyticalTransitionModel(), DistanceWeights.from_dict(synth_config["identifiability"]["distance"])
    )
    for b in range(12):
        base = generate_base_scene(np.random.default_rng([37, b]), synth_config, b)
        hyps = build_hypothesis_set(base.content.interface, synth_config, np.random.default_rng([37, b, 7]))
        f0 = reference_observation(base.content, hyps[0]).feature
        sep = est.pairwise(f0, hyps, null_action(), markers_cam=base.content.observer_markers_cam)
        assert np.allclose(sep, 0.0), f"base scene {b} is separable without moving: {sep}"
