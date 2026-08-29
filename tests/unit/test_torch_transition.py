"""The Gate-7 geometry and its supervision.

These tests exist because of a real defect. The first version of ``warp_depth``
took the inverse of the relative pose, and the loader de-normalised the two focal
lengths by different image dimensions. Neither error raises, neither shows up in
a loss curve -- training proceeded, the loss fell convincingly, and the resulting
"learned residual" was a registration error rather than optics.

The decisive check is the last one: **rigid reprojection must beat the identity
warp.** If moving the camera correctly explains the next frame no better than
pretending it never moved, the pose is being applied wrongly, whatever the loss
curve says.
"""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.models.torch_transition import (
    TORCH_INPUT_DIM,
    TORCH_OUTPUT_DIM,
    build_pair_dataset,
    occlusion_mask,
    pair_supervision,
    sample_nearest,
    warp_depth,
)


def _K(fx: float = 200.0, w: int = 80, h: int = 60) -> np.ndarray:
    return np.array([[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]])


def _pose(tx=0.0, ty=0.0, tz=0.0, yaw=0.0) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T[:3, 3] = (tx, ty, tz)
    return T



def _render_next_frame(depth0: np.ndarray, K: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Z-buffered forward render of a rigid scene into the next camera.

    A plain "last write wins" scatter would leave *occluded* depths in the frame,
    which is not what a renderer produces and would make this fixture disagree
    with physics rather than with the code under test.
    """
    h, w = depth0.shape
    u, v, z = warp_depth(depth0, K, pose)
    out = np.full((h, w), np.inf)
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    ui = np.rint(u[ok]).astype(int)
    vi = np.rint(v[ok]).astype(int)
    ins = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    np.minimum.at(out, (vi[ins], ui[ins]), z[ok][ins])   # nearest surface wins
    return np.where(np.isfinite(out), out, np.nan)


# ------------------------------------------------------------------ the warp
def test_identity_pose_is_the_identity_warp():
    """No motion must reproject every pixel onto itself at its own depth."""
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 5.0, size=(60, 80))
    u, v, z = warp_depth(depth, _K(), np.eye(4))
    vs, us = np.mgrid[0:60, 0:80]
    assert np.allclose(u, us, atol=1e-9)
    assert np.allclose(v, vs, atol=1e-9)
    assert np.allclose(z, depth, atol=1e-9)


def test_pure_forward_translation_moves_depth_by_exactly_that_amount():
    """Advancing 0.5 m along the optical axis must reduce every z by 0.5 m."""
    depth = np.full((40, 40), 3.0)
    _, _, z = warp_depth(depth, _K(w=40, h=40), _pose(tz=-0.5))
    assert np.allclose(z, 2.5, atol=1e-9)


@pytest.mark.parametrize("pose", [
    _pose(tx=0.15),
    _pose(tx=0.1, ty=-0.05, tz=0.2),
    _pose(yaw=np.deg2rad(8.0)),
    _pose(tx=0.2, yaw=np.deg2rad(-5.0)),
])
def test_a_rigid_scene_reprojects_with_zero_residual(pose):
    """THE test the original code failed.

    Construct a rigid scene, reproject it exactly, and treat the reprojection as
    the observation. A correct warp then has *no* residual left to explain. An
    inverted or transposed pose produces a large residual here while still
    training happily on real data.
    """
    K = _K()
    # A smooth surface: random per-pixel depth would make every neighbouring
    # pixel mutually occluding and the fixture would test nothing but rounding.
    vs, us = np.mgrid[0:60, 0:80].astype(float)
    depth0 = 4.0 + 0.6 * np.sin(us / 14.0) + 0.4 * np.cos(vs / 11.0)
    u, v, z = warp_depth(depth0, K, pose)
    depth1 = _render_next_frame(depth0, K, pose)

    observed = sample_nearest(depth1, u, v)
    keep = occlusion_mask(u, v, z, depth0.shape)
    m = np.isfinite(observed) & np.isfinite(z) & keep
    assert np.count_nonzero(m) > 500, "the test scene barely reprojects; it proves nothing"
    residual = 1.0 / observed[m] - 1.0 / z[m]
    # Non-zero only through the nearest-neighbour rounding of the scatter.
    assert np.median(np.abs(residual)) < 1e-9
    assert np.sqrt(np.mean(residual**2)) < 5e-3


def test_out_of_frame_samples_are_nan_not_clamped():
    """A point that left the frame must not inherit a border pixel's depth."""
    image = np.arange(12, dtype=float).reshape(3, 4)
    got = sample_nearest(image, np.array([-1.0, 0.0, 3.0, 9.0]), np.array([0.0, 0.0, 2.0, 1.0]))
    assert np.isnan(got[0]) and np.isnan(got[3])
    assert got[1] == image[0, 0] and got[2] == image[2, 3]


def test_non_finite_depth_stays_non_finite():
    depth = np.full((10, 10), 2.0)
    depth[0, 0] = np.nan
    depth[1, 1] = 0.0
    _, _, z = warp_depth(depth, _K(w=10, h=10), _pose(tx=0.1))
    assert np.isnan(z[0, 0])
    assert np.isnan(z[1, 1])  # zero depth is unmeasured, not "at the camera"


# ----------------------------------------------------------- the supervision
class _Frame:
    """Minimal stand-in for an ExternalSample."""

    def __init__(self, depth, K, key="s0", seq="0"):
        self.depth = depth
        self.intrinsics = K
        self.key = key
        self.extra = {"sequence_id": seq}


def test_supervision_is_empty_for_a_perfectly_rigid_pair():
    K = _K()
    vs, us = np.mgrid[0:60, 0:80].astype(float)
    d0 = 4.0 + 0.6 * np.sin(us / 14.0) + 0.4 * np.cos(vs / 11.0)
    pose = _pose(tx=0.12)
    d1 = _render_next_frame(d0, K, pose)

    X, Y, stats = pair_supervision(_Frame(d0, K), _Frame(d1, K), pose, max_points=2000)
    assert X.shape[1] == TORCH_INPUT_DIM and Y.shape[1] == TORCH_OUTPUT_DIM
    assert X.shape[0] == Y.shape[0]
    assert stats["residual_rms_inv_m"] < 5e-3, "a rigid pair must leave essentially nothing to learn"


def test_missing_depth_or_intrinsics_yields_no_supervision_rather_than_zeros():
    K = _K()
    d = np.full((20, 20), 3.0)
    X, Y, _ = pair_supervision(_Frame(None, K), _Frame(d, K), np.eye(4))
    assert X.shape[0] == 0 and Y.shape[0] == 0
    X, Y, _ = pair_supervision(_Frame(d, None), _Frame(d, K), np.eye(4))
    assert X.shape[0] == 0 and Y.shape[0] == 0


def test_unmeasured_pixels_are_dropped_not_zero_filled():
    """A zero residual where there is no evidence is a fabricated label."""
    K = _K()
    vs, us = np.mgrid[0:60, 0:80].astype(float)
    d0 = 4.0 + 0.6 * np.sin(us / 14.0) + 0.4 * np.cos(vs / 11.0)
    d1 = d0.copy()
    d1[:30, :] = np.nan  # half the next frame was never measured
    X, _, stats = pair_supervision(_Frame(d0, K), _Frame(d1, K), np.eye(4), max_points=10_000)
    assert stats["valid_fraction"] < 0.6
    assert X.shape[0] <= stats["valid_pixels"]


# --------------------------------------------------- group / leakage tracking
def test_every_row_carries_the_group_of_the_pair_it_came_from():
    """Regression: group labels were reconstructed by position and could shift.

    ``build_pair_dataset`` skips pairs that yield nothing, so a caller that
    zipped a separately-accumulated group list against the surviving rows
    mislabelled everything after the first skipped pair -- and the "held-out
    sequence" split then leaked.
    """
    K = _K()
    vs, us = np.mgrid[0:30, 0:40].astype(float)

    def make(seq, usable=True):
        # Smooth, like a real surface: per-pixel noise makes every 2x2
        # neighbourhood mutually inconsistent and nothing survives sampling.
        d0 = 4.0 + 0.3 * np.sin(us / 9.0) + 0.2 * np.cos(vs / 7.0)
        d1 = d0 * 1.005 if usable else np.full_like(d0, np.nan)
        return _Frame(d0, K, seq=seq), _Frame(d1, K, seq=seq), np.eye(4)

    pairs = [make("A"), make("B", usable=False), make("C"), make("D", usable=False), make("E")]
    X, Y, report = build_pair_dataset(iter(pairs), max_pairs=10, max_points=200, seed=0)

    groups = report["row_groups"]
    assert len(groups) == X.shape[0], "one group label per row, always"
    assert set(groups) == {"A", "C", "E"}, "skipped pairs must not shift the labels"
    assert report["pairs_used"] == 3 and report["pairs_skipped"] == 2


def test_row_groups_survive_uneven_pair_sizes():
    """Pairs contributing different row counts must still map correctly."""
    K = _K()
    vs, us = np.mgrid[0:40, 0:40].astype(float)

    def make(seq, fraction):
        d0 = 4.0 + 0.3 * np.sin(us / 9.0) + 0.2 * np.cos(vs / 7.0)
        d1 = d0 * 1.002
        cut = int(40 * fraction)
        d1[cut:, :] = np.nan  # different usable fraction per pair
        return _Frame(d0, K, seq=seq), _Frame(d1, K, seq=seq), np.eye(4)

    pairs = [make("A", 0.9), make("B", 0.3), make("C", 0.6)]
    X, _, report = build_pair_dataset(iter(pairs), max_pairs=10, max_points=100_000, seed=0)
    groups = np.array(report["row_groups"])
    assert len(groups) == X.shape[0]
    counts = {g: int(np.count_nonzero(groups == g)) for g in ("A", "B", "C")}
    assert counts["A"] > counts["C"] > counts["B"], f"row counts do not track usable area: {counts}"


# ------------------------------------------------- the guard against B1, on real data
def _transphy3d_pairs(stride: int, n: int):
    """Real frame pairs, or a skip. Never a silent pass."""
    loaders = pytest.importorskip("intervene3d.data.external.loaders")
    try:
        reader = loaders.get_reader("transphy3d", variant="sample")
    except loaders.LoaderError as exc:
        pytest.skip(f"TransPhy3D sample not acquired: {exc}")
    try:
        import PIL  # noqa: F401
    except ImportError:
        pytest.skip('reading TransPhy3D needs Pillow: pip install -e ".[data]"')
    import itertools

    pairs = list(itertools.islice(reader.iter_pairs(stride=stride), n))
    if len(pairs) < n:
        pytest.skip("not enough real frame pairs available")
    return pairs


def _photometric_mae(s0, s1, K, T) -> float:
    """Project frame t's pixels into t+1 and compare RGB.

    Independent of the depth residual the model is trained on, so it cannot be
    satisfied by the same error that would corrupt that residual.
    """
    u, v, _ = warp_depth(s0.depth, K, T)
    h, w = s0.depth.shape
    ui, vi = np.rint(u), np.rint(v)
    ok = (
        np.isfinite(ui) & np.isfinite(vi) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        & np.isfinite(s0.depth) & (s0.depth > 1e-3)
    )
    if np.count_nonzero(ok) < 1000:
        return float("nan")
    a = s0.image[ok].astype(float)
    b = s1.image[vi[ok].astype(int), ui[ok].astype(int)].astype(float)
    return float(np.mean(np.abs(a - b)))


@pytest.mark.slow
def test_reprojection_beats_doing_nothing_on_real_frames():
    """THE guard. If moving the camera correctly explains the next frame no
    better than pretending it never moved, the pose convention is wrong.

    This is what caught the original defect: the shipped convention scored
    *worse* than the identity warp, while training happily and showing a falling
    loss. Both the extrinsics reading (world-to-camera, applied without a further
    inverse) and the square-pixel intrinsics are pinned by this test.
    """
    pairs = _transphy3d_pairs(stride=5, n=4)
    warped = np.nanmedian([_photometric_mae(a, b, a.intrinsics, T) for a, b, T in pairs])
    identity = np.nanmedian([_photometric_mae(a, b, a.intrinsics, np.eye(4)) for a, b, _ in pairs])
    assert np.isfinite(warped) and np.isfinite(identity)
    assert warped < identity, (
        f"rigid reprojection (MAE {warped:.3f}) is no better than not moving the camera "
        f"(MAE {identity:.3f}). The relative pose is being applied wrongly; any residual "
        "trained on this is a registration error, not optics."
    )
    assert identity / warped > 1.5, f"reprojection only {identity / warped:.2f}x better than nothing"


@pytest.mark.slow
def test_real_intrinsics_have_square_pixels():
    """A 4:3 pixel aspect would mean the normalised K was de-normalised per-axis."""
    s0, _, _ = _transphy3d_pairs(stride=5, n=1)[0]
    K = s0.intrinsics
    assert K is not None
    assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-6), (
        f"fx={K[0, 0]:.3f} != fy={K[1, 1]:.3f}; a Blender render has square pixels, so both "
        "focals normalise by the same factor and only the principal point is per-axis"
    )
    assert K[0, 2] == pytest.approx(320.0, abs=1.0)
    assert K[1, 2] == pytest.approx(240.0, abs=1.0)
