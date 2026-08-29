"""Gate 7: a PyTorch transition model trained on rendered multi-view sequences.

Why this exists
---------------
The NumPy :class:`~intervene3d.models.learned_transition.ResidualMLP` can only be
trained on the synthetic benchmark, where the analytical transition reproduces
the simulator exactly. The residual target is then **identically zero**, so the
hybrid model is equal to its base and nothing is learned. That is a property of
the benchmark, not a result.

TransPhy3D breaks the degeneracy. It is Blender/Cycles renderings of transparent
and reflective scenes with a **per-frame camera extrinsic**, so a frame pair
``(t, t+k)`` is a genuine ``(F_t, a) -> F_{t+1}`` example whose optics our
analytical model does *not* generate, so the residual is not degenerate. Note
TransPhy3D is Blender/Cycles rendering, not photography, and this network takes
no hypothesis input -- it is not the hypothesis-conditioned transition the
project claims as its mechanism.

What is predicted
-----------------
Per pixel-track, the residual between the observed next-frame geometry and what
rigid reprojection of the reference depth predicts::

    target = observed_depth(t+1)  -  warp(depth_t, T_rel, K)

``warp`` is exact rigid reprojection, so the network only ever has to explain
what rigid geometry cannot: transparency, reflection, and everything else that
makes a surface non-Lambertian. Predicting zero is exactly "the scene is a
rigid, opaque, directly-observed world" -- the ``H_D`` hypothesis. The learned
residual is the departure from it.

Honesty constraints, enforced in code
-------------------------------------
* Supervision comes only from pixels with **finite depth in both frames**, a
  valid reprojection inside the image, and which win a **z-buffer test** at their
  target pixel. Occluded or missing pixels are dropped, never zero-filled -- a
  zero residual where there is no evidence is a fabricated label, and it would
  teach the model that "no data" means "rigid". The occlusion test is a z-buffer
  rather than a threshold on the residual, because a magnitude threshold would
  delete the transparency and reflection this model exists to learn.
* Train / validation are split by **sequence** (shard), never by frame. Frames of
  one sequence are near-duplicates; splitting them would report a validation
  number the model had effectively already seen.
* The metric-depth scale is applied from each frame's own ``max_depth``. The
  loader refuses a frame whose scale is unknown rather than guessing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "the torch transition needs PyTorch, which is an optional extra:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "(cu128 is required for Blackwell / sm_120 cards such as the RTX 5070)"
        ) from exc
    return torch


def select_device(preferred: str = "auto") -> str:
    """Resolve ``auto`` honestly: never claim CUDA when it is unavailable."""
    torch = _torch()
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def device_report(device: str) -> dict[str, Any]:
    """What was actually computed on, recorded alongside every result."""
    torch = _torch()
    info: dict[str, Any] = {"device": device, "torch": torch.__version__}
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            name=props.name,
            compute_capability=f"sm_{props.major}{props.minor}",
            total_memory_gb=round(props.total_memory / 1e9, 2),
            multiprocessors=props.multi_processor_count,
            cuda=torch.version.cuda,
            arch_list=list(torch.cuda.get_arch_list()),
        )
    return info


# ------------------------------------------------------------------ geometry
def warp_depth(
    depth: np.ndarray, K: np.ndarray, T_rel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigidly reproject ``depth`` from frame *t* into frame *t+1*.

    ``T_rel`` maps a point **from camera t's frame into camera t+1's frame** and
    is applied as given -- no inverse is taken here. That convention is fixed by
    :meth:`~intervene3d.data.external.loaders.TransPhy3DReader.iter_pairs`, and
    it is the single easiest thing in this file to get backwards: an inverted
    pose still trains, still shows a falling loss, and measures nothing.
    ``tests/unit/test_torch_transition.py`` pins it with a synthetic rigid scene
    whose residual must be ~0, plus the check that reprojection beats the
    identity warp on real frames.

    Returns ``(u, v, z)`` in the target frame, all ``(H, W)``. This is the
    ``H_D`` prediction: what the next view would show if the scene were a rigid,
    opaque, directly-observed world. Non-finite entries mark pixels that cannot
    be reprojected, and they stay non-finite rather than becoming zeros.
    """
    h, w = depth.shape
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    with np.errstate(invalid="ignore", divide="ignore"):
        x = (us - cx) / fx * depth
        y = (vs - cy) / fy * depth
    pts = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)      # (H, W, 4)
    moved = pts @ np.asarray(T_rel, dtype=np.float64).T               # into frame t+1
    zt = moved[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        ut = moved[..., 0] / zt * fx + cx
        vt = moved[..., 1] / zt * fy + cy
    bad = ~np.isfinite(zt) | (zt <= 1e-6)
    ut = np.where(bad, np.nan, ut)
    vt = np.where(bad, np.nan, vt)
    zt = np.where(bad, np.nan, zt)
    return ut, vt, zt


def sample_bilinear_consistent(
    image: np.ndarray, u: np.ndarray, v: np.ndarray, *, rel_tol: float = 0.02
) -> np.ndarray:
    """Bilinear sample, but only where the 2x2 neighbourhood agrees on depth.

    Nearest-neighbour rounding injects ~3x more error than the optical residual
    it is meant to measure, so it cannot be used on the geometrically consistent
    pixels where the signal lives. Plain bilinear is worse still -- it blends
    across depth discontinuities and fabricates a surface that is in neither
    frame. Interpolating only within a consistent neighbourhood keeps the
    precision and refuses the fabrication: a straddling sample returns NaN.
    """
    h, w = image.shape
    u0 = np.floor(u)
    v0 = np.floor(v)
    ok = np.isfinite(u0) & np.isfinite(v0) & (u0 >= 0) & (u0 < w - 1) & (v0 >= 0) & (v0 < h - 1)
    out = np.full(u.shape, np.nan, dtype=np.float64)
    if not np.any(ok):
        return out
    x0 = u0[ok].astype(int)
    y0 = v0[ok].astype(int)
    fx = (u[ok] - x0)[:, None]
    fy = (v[ok] - y0)[:, None]
    q = np.stack([image[y0, x0], image[y0, x0 + 1], image[y0 + 1, x0], image[y0 + 1, x0 + 1]], axis=1)
    finite = np.all(np.isfinite(q), axis=1)   # an all-NaN row has no min/max
    lo = np.where(finite, np.nanmin(np.where(np.isfinite(q), q, np.inf), axis=1), np.nan)
    hi = np.where(finite, np.nanmax(np.where(np.isfinite(q), q, -np.inf), axis=1), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = (hi - lo) / np.maximum(lo, 1e-6)
    agree = np.isfinite(spread) & (spread <= rel_tol) & finite
    wts = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
    vals = np.sum(q * wts, axis=1)
    res = np.where(agree, vals, np.nan)
    out[ok] = res
    return out


def sample_nearest(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-neighbour lookup; out-of-bounds returns NaN rather than a clamp.

    Clamping would silently attribute a border pixel's depth to a point that left
    the frame, which is a fabricated observation.
    """
    h, w = image.shape
    ui = np.rint(u)
    vi = np.rint(v)
    inside = np.isfinite(ui) & np.isfinite(vi) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    out = np.full(u.shape, np.nan, dtype=np.float64)
    if np.any(inside):
        out[inside] = image[vi[inside].astype(int), ui[inside].astype(int)]
    return out



def occlusion_mask(u: np.ndarray, v: np.ndarray, z: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """True where a forward-warped pixel is the NEAREST one at its target pixel.

    Several source pixels can land on the same target pixel; only the closest is
    actually visible there. Without this test an occluded point is handed the
    occluder's depth, which is a fabricated observation -- precisely the thing
    :func:`pair_supervision` claims not to do.

    Deliberately a **z-buffer** test rather than a threshold on
    ``|z_rigid - z_obs|``. A magnitude threshold would also discard large genuine
    residuals, and large genuine residuals -- transparency, reflection -- are the
    entire signal this model exists to learn.
    """
    h, w = shape
    ui = np.rint(u)
    vi = np.rint(v)
    ok = np.isfinite(ui) & np.isfinite(vi) & np.isfinite(z) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    keep = np.zeros(u.shape, dtype=bool)
    if not np.any(ok):
        return keep
    flat = (vi[ok].astype(np.int64) * w + ui[ok].astype(np.int64))
    zz = z[ok]
    # Nearest-per-target: sort by (target, z) and take the first of each target.
    order = np.lexsort((zz, flat))
    first = np.ones(order.size, dtype=bool)
    first[1:] = flat[order][1:] != flat[order][:-1]
    winners = np.zeros(order.size, dtype=bool)
    winners[order[first]] = True
    keep[ok] = winners
    return keep


# --------------------------------------------------------------- supervision
#: (inv_depth, u_norm, v_norm, warped_inv_depth, translation(3), rotvec(3))
TORCH_INPUT_DIM = 10
TORCH_OUTPUT_DIM = 1  # residual inverse depth


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Axis-angle of a rotation, correct at 180 degrees.

    The naive form divides by ``sin(angle)`` and so returns "no rotation" for a
    half turn -- reachable here, since the camera turns exactly 3 deg/frame and
    ``--stride 60`` lands on 180 exactly.
    """
    angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.zeros(3)
    sin_a = np.sin(angle)
    if abs(sin_a) > 1e-6:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return axis / (2.0 * sin_a) * angle
    # angle ~ pi: recover the axis from the symmetric part, where it is stable.
    M = (R + np.eye(3)) / 2.0
    axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
    k = int(np.argmax(axis))
    if axis[k] > 1e-9:
        axis = M[:, k] / axis[k]
    n = np.linalg.norm(axis)
    return (axis / n * angle) if n > 1e-9 else np.zeros(3)


def pair_supervision(
    sample_t, sample_next, T_rel: np.ndarray, *, max_points: int = 4096,
    rng: np.random.Generator | None = None, consistency_tau: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build ``(X, Y)`` for one frame pair, in inverse-depth space.

    Inverse depth keeps a 0.5 m surface and a 20 m surface on comparable scales;
    an MSE on raw metres would be dominated entirely by the far field.
    """
    rng = rng or np.random.default_rng(0)
    d0 = sample_t.depth
    d1 = sample_next.depth
    K = sample_t.intrinsics
    if d0 is None or d1 is None or K is None:
        return (np.zeros((0, TORCH_INPUT_DIM)), np.zeros((0, TORCH_OUTPUT_DIM)),
                {"reason": "missing depth or K", "consistent": np.zeros(0, dtype=bool)})

    u, v, z_rigid = warp_depth(d0, K, T_rel)
    # Bilinear within a depth-consistent neighbourhood: nearest-neighbour
    # rounding is ~3x the optical residual being measured.
    z_obs = sample_bilinear_consistent(d1, u, v)

    # A pixel supervises only where BOTH frames measured it and the reprojection
    # landed in frame. Everything else is dropped, not zero-filled.
    visible = occlusion_mask(u, v, z_rigid, d0.shape)
    valid = (
        np.isfinite(d0) & (d0 > 1e-3)
        & np.isfinite(z_rigid) & (z_rigid > 1e-3)
        & np.isfinite(z_obs) & (z_obs > 1e-3)
        & visible
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return (np.zeros((0, TORCH_INPUT_DIM)), np.zeros((0, TORCH_OUTPUT_DIM)),
                {"reason": "no valid pixels", "consistent": np.zeros(0, dtype=bool)})

    # Occlusion and optical anomaly BOTH produce large |z_obs - z_rigid|, and no
    # threshold can separate them: gating on magnitude would delete exactly the
    # transparency and reflection this model exists to learn. So rows are TAGGED,
    # not dropped, and every metric is reported on both subsets separately.
    # Pooling them lets an occlusion detector masquerade as an optics model.
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(z_obs - z_rigid) / np.maximum(z_rigid, 1e-6)
    consistent_map = np.isfinite(rel) & (rel <= consistency_tau)

    idx = np.flatnonzero(valid.ravel())
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    consistent = consistent_map.ravel()[idx]

    h, w = d0.shape
    vs, us = np.divmod(idx, w)
    inv0 = 1.0 / d0.ravel()[idx]
    inv_rigid = 1.0 / z_rigid.ravel()[idx]
    inv_obs = 1.0 / z_obs.ravel()[idx]

    t_vec = T_rel[:3, 3]
    r_vec = _rotvec(T_rel[:3, :3])
    tail = np.concatenate([t_vec, r_vec])

    X = np.concatenate(
        [
            np.stack([inv0, us / w, vs / h, inv_rigid], axis=1),
            np.tile(tail, (idx.size, 1)),
        ],
        axis=1,
    )
    Y = (inv_obs - inv_rigid).reshape(-1, 1)
    stats = {
        "valid_pixels": n_valid,
        "sampled": int(idx.size),
        "valid_fraction": round(n_valid / d0.size, 4),
        "occluded_dropped": int(np.count_nonzero(~visible & np.isfinite(z_rigid))),
        "residual_rms_inv_m": float(np.sqrt(np.mean(Y**2))),
        "consistent": consistent,
        "consistent_fraction": round(float(np.mean(consistent)), 4),
        "consistency_tau": consistency_tau,
    }
    return X, Y, stats



def _group_of(sample: Any) -> str:
    """The sequence a frame belongs to -- the unit the split must respect.

    Prefers the publisher's own ``sequence_id`` over parsing the key.
    """
    extra = getattr(sample, "extra", None) or {}
    seq = extra.get("sequence_id")
    if seq:
        return str(seq)
    key = str(getattr(sample, "key", ""))
    return key.split("_")[0] if "_" in key else key


def build_pair_dataset(
    pairs: Iterator[tuple[Any, Any, np.ndarray]],
    *,
    max_pairs: int,
    max_points: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Accumulate supervision over many frame pairs."""
    rng = np.random.default_rng(seed)
    xs, ys, used, skipped = [], [], 0, 0
    valid_fractions: list[float] = []
    # The group label is emitted HERE, one per surviving row, rather than
    # reconstructed by the caller from a parallel list. Pairs that yield nothing
    # are skipped, and a positional reconstruction then mislabels every row after
    # the first skip -- which silently breaks the held-out-sequence split.
    row_groups: list[str] = []
    row_consistent: list[np.ndarray] = []
    for a, b, T in pairs:
        if used >= max_pairs:
            break
        X, Y, stats = pair_supervision(a, b, T, max_points=max_points, rng=rng)
        if X.shape[0] == 0:
            skipped += 1
            continue
        xs.append(X)
        ys.append(Y)
        row_groups.extend([_group_of(a)] * X.shape[0])
        row_consistent.append(np.asarray(stats["consistent"], dtype=bool))
        used += 1
        valid_fractions.append(stats["valid_fraction"])
    if not xs:
        raise RuntimeError("no usable frame pairs produced supervision")
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    assert len(row_groups) == X.shape[0]  # the invariant the split depends on
    consistent = np.concatenate(row_consistent) if row_consistent else np.zeros(0, dtype=bool)
    assert consistent.size == X.shape[0]
    cons_rms = float(np.sqrt(np.mean(Y[consistent] ** 2))) if consistent.any() else float("nan")
    return X, Y, {
        "row_consistent": consistent,
        "consistent_fraction": round(float(np.mean(consistent)), 4),
        "target_rms_inv_m_consistent": cons_rms,
        "pairs_used": used,
        "pairs_skipped": skipped,
        "n_samples": int(X.shape[0]),
        "n_groups": len(set(row_groups)),
        "row_groups": row_groups,
        "mean_valid_pixel_fraction": round(float(np.mean(valid_fractions)), 4),
        "target_rms_inv_m": float(np.sqrt(np.mean(Y**2))),
    }


# -------------------------------------------------------------------- model
@dataclass
class TorchResidualConfig:
    hidden_dim: int = 256
    depth: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8192
    epochs: int = 40
    device: str = "auto"
    amp: bool = True          # bf16 autocast; the 5070 does 50 TFLOP/s in bf16
    seed: int = 0


def build_mlp(cfg: TorchResidualConfig):
    """A plain MLP. The scientific content is the residual target, not the net."""
    torch = _torch()
    nn = torch.nn
    layers: list[Any] = [nn.Linear(TORCH_INPUT_DIM, cfg.hidden_dim), nn.GELU()]
    for _ in range(max(cfg.depth - 1, 0)):
        layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
    head = nn.Linear(cfg.hidden_dim, TORCH_OUTPUT_DIM)
    # Zero-init the head so the model starts exactly at the rigid prediction:
    # training can only improve on H_D, never begin somewhere arbitrary.
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


@dataclass
class TorchTrainingResult:
    state_dict: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


def train_torch_residual(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    groups: Sequence[str] | None = None,
    consistent: np.ndarray | None = None,
    cfg: TorchResidualConfig | None = None,
    val_fraction: float = 0.2,
) -> TorchTrainingResult:
    """Train the residual network, holding out whole **sequences** for validation."""
    torch = _torch()
    cfg = cfg or TorchResidualConfig()
    device = select_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    if groups is not None and len(set(groups)) > 1:
        uniq = sorted(set(groups))
        n_hold = max(1, int(round(len(uniq) * val_fraction)))
        held = set(rng.permutation(np.array(uniq, dtype=object))[:n_hold].tolist())
        val_mask = np.array([g in held for g in groups])
        split_note = f"held out {len(held)}/{len(uniq)} sequences: {sorted(held)}"
    else:
        # No grouping available: fall back to a row split and SAY SO, because the
        # resulting validation number is optimistic on correlated frames.
        val_mask = rng.random(X.shape[0]) < val_fraction
        split_note = "row-wise split (no sequence grouping available) -- optimistic"

    Xtr = torch.tensor(X[~val_mask], dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y[~val_mask], dtype=torch.float32, device=device)
    Xva = torch.tensor(X[val_mask], dtype=torch.float32, device=device)
    Yva = torch.tensor(Y[val_mask], dtype=torch.float32, device=device)

    model = build_mlp(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    use_amp = bool(cfg.amp and device.startswith("cuda"))

    history: list[dict[str, float]] = []
    n = Xtr.shape[0]
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.shape[0]
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model(Xva), Yva)) if Xva.shape[0] else float("nan")
        history.append({"epoch": epoch + 1, "train_mse": total / max(n, 1), "val_mse": val})
        if (epoch + 1) % max(cfg.epochs // 5, 1) == 0:
            LOGGER.info("  epoch %3d  train %.6g  val %.6g", epoch + 1, history[-1]["train_mse"], val)

    # Predicting zero residual IS the rigid H_D model, so a ratio below 1 means
    # the network beat the physical null hypothesis.
    #
    # Reported on THREE subsets, never pooled into one headline. Occlusion
    # boundaries carry orders of magnitude more residual than optics does, so a
    # pooled ratio is dominated by them: a network that only learns to predict
    # occlusion scores brilliantly while being worse than useless on the
    # geometrically consistent pixels where transparency and reflection live.
    # The `consistent` subset is the one that speaks to the optics claim.
    with torch.no_grad():
        zero_val = float(torch.mean(Yva**2)) if Xva.shape[0] else float("nan")
        model_val = history[-1]["val_mse"] if history else float("nan")
        best = min((h["val_mse"] for h in history if np.isfinite(h["val_mse"])), default=float("nan"))
        subsets: dict[str, Any] = {}
        if consistent is not None and Xva.shape[0]:
            cons_va = torch.tensor(np.asarray(consistent, dtype=bool)[val_mask], device=device)
            for label, sel in (("consistent", cons_va), ("occlusion_affected", ~cons_va)):
                if not bool(sel.any()):
                    subsets[label] = None
                    continue
                yy, xx = Yva[sel], Xva[sel]
                base = float(torch.mean(yy**2))
                mm = float(torch.nn.functional.mse_loss(model(xx), yy))
                subsets[label] = {
                    "n": int(sel.sum()),
                    "fraction": round(float(sel.float().mean()), 4),
                    "rigid_baseline_mse": base,
                    "model_mse": mm,
                    "ratio_vs_rigid": round(mm / base, 6) if base > 0 else None,
                }

    report = {
        "device": device_report(device),
        "config": cfg.__dict__,
        "n_train": int(Xtr.shape[0]),
        "n_val": int(Xva.shape[0]),
        "split_protocol": split_note,
        "train_seconds": round(time.time() - t0, 2),
        "history": history,
        "final_train_mse": history[-1]["train_mse"] if history else None,
        "final_val_mse": model_val,
        "best_val_mse": best,   # a heavy-tailed target + cosine LR can end above its own best
        "rigid_baseline_val_mse": zero_val,
        "val_mse_ratio_vs_rigid_POOLED": round(model_val / zero_val, 6) if zero_val and np.isfinite(zero_val) else None,
        "by_subset": subsets,
        "target": "residual inverse depth vs rigid reprojection (the H_D prediction)",
        "reporting_note": (
            "The pooled ratio is NOT the optics result. Occlusion boundaries dominate the "
            "residual by orders of magnitude, so `by_subset.consistent.ratio_vs_rigid` is the "
            "number that speaks to whether the model explains non-rigid optics; "
            "`by_subset.occlusion_affected` measures occlusion prediction."
        ),
    }
    return TorchTrainingResult(
        state_dict={k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()},
        report=report,
    )
