"""Gate 6: a real geometry foundation encoder.

Until now every ``F_t = E(I_t)`` in this repository was an oracle:
``GroundTruthEncoder`` returns the simulator's own state, and ``MockEncoder``
returns that plus noise. Neither has ever looked at a pixel. This module is the
first encoder that actually runs a vision model on an image.

What it is
----------
:class:`MonocularDepthEncoder` wraps a published monocular depth network through
``transformers`` and samples its prediction at the landmark pixels, returning a
:class:`~intervene3d.data.types.GeometryFeature` with the *same* layout as the
simulator so it is a drop-in replacement.

Three things it refuses to do, because each would produce a plausible wrong number
--------------------------------------------------------------------------------
1. **It never invents an image.** ``encode()`` raises if the observation carries
   no RGB. The synthetic benchmark stores landmark arrays and only a handful of
   preview renders, so most synthetic scenes genuinely cannot be encoded this
   way -- and silently falling back to the oracle would let a run claim a
   foundation encoder it never ran.
2. **It never claims metric depth.** Depth Anything V2 predicts *relative inverse
   depth*, defined up to an unknown scale and shift. :meth:`predict_inverse_depth`
   returns exactly that and says so. Metric values appear only through
   :func:`align_scale_shift`, which fits the two free parameters against a
   reference and reports the residual, so the alignment is visible rather than
   assumed.
3. **It never hides the licence.** The default checkpoint is Apache-2.0. The
   larger ones are **CC-BY-NC-4.0 (non-commercial)**, verified on 2026-08-29
   against the Hugging Face model API; selecting one records that restriction in
   :meth:`to_dict` so it reaches the run manifest.

Domain caveat, stated once and loudly: this repository's synthetic previews come
from a splat renderer that its own README calls "adequate for figures only".
Running a photo-trained network on those renders measures the domain gap, not
the network. The meaningful use of this encoder is on the acquired external
imagery (:mod:`intervene3d.data.external.loaders`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from intervene3d.data.types import GeometryFeature, Observation

LOGGER = logging.getLogger(__name__)

#: Licences verified 2026-08-29 against https://huggingface.co/api/models/<id>.
#: Only the Apache-2.0 checkpoint is a safe default; the others restrict use.
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "depth-anything/Depth-Anything-V2-Small-hf": {
        "licence": "apache-2.0",
        "commercial_use": True,
        "params_m": 24.8,
    },
    "depth-anything/Depth-Anything-V2-Base-hf": {
        "licence": "cc-by-nc-4.0",
        "commercial_use": False,
        "params_m": 97.5,
    },
    "depth-anything/Depth-Anything-V2-Large-hf": {
        "licence": "cc-by-nc-4.0",
        "commercial_use": False,
        "params_m": 335.3,
    },
}
DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"


class EncoderUnavailable(RuntimeError):
    """The encoder cannot run, and will not pretend otherwise."""


def align_scale_shift(
    prediction: np.ndarray, reference: np.ndarray, *, valid: np.ndarray | None = None
) -> tuple[float, float, dict[str, Any]]:
    """Least-squares fit of ``s * prediction + t ~ reference``.

    Relative inverse depth is only defined up to this pair, so any metric
    comparison must fit it explicitly. Returns ``(scale, shift, report)`` where
    the report carries the residual -- a bad fit means the comparison is
    meaningless and the caller should know rather than plot it.
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    r = np.asarray(reference, dtype=np.float64).ravel()
    m = np.isfinite(p) & np.isfinite(r)
    if valid is not None:
        m &= np.asarray(valid, dtype=bool).ravel()
    n = int(np.count_nonzero(m))
    if n < 2:
        raise EncoderUnavailable(
            f"scale/shift alignment needs at least 2 paired finite samples, got {n}"
        )
    A = np.stack([p[m], np.ones(n)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(A, r[m], rcond=None)
    residual = A @ np.array([scale, shift]) - r[m]
    denom = float(np.var(r[m]))
    return (
        float(scale),
        float(shift),
        {
            "n_paired": n,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "r2": float(1.0 - np.var(residual) / denom) if denom > 0 else None,
        },
    )


@dataclass
class MonocularDepthEncoder:
    """A published monocular depth network, run for real.

    ``encode`` samples the predicted map at each landmark's pixel and returns a
    ``GeometryFeature``. Because the network is relative, the returned ``depth``
    is in **inverse-depth units unless** ``align_to_reference`` is set, in which
    case the two free parameters are fitted against the observation's own
    landmark depths and the fit quality is recorded.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    device: str = "auto"
    align_to_reference: bool = False
    batch_size: int = 1
    name: str = "depth_anything_v2"
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _resolved_device: str = field(default="", init=False, repr=False)
    _last_alignment: dict[str, Any] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ setup
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise EncoderUnavailable(
                f"encoder {self.name!r} needs torch and transformers:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                "  pip install transformers\n"
                "(cu128 is required for Blackwell / sm_120 cards such as the RTX 5070)"
            ) from exc

        meta = CHECKPOINTS.get(self.checkpoint)
        if meta is None:
            LOGGER.warning(
                "checkpoint %r is not in the verified list %s; its licence has NOT been "
                "checked here and will be recorded as unknown",
                self.checkpoint, sorted(CHECKPOINTS),
            )
        elif not meta["commercial_use"]:
            LOGGER.warning(
                "checkpoint %r is %s -- NON-COMMERCIAL use only. This restriction is "
                "recorded in the run manifest.", self.checkpoint, meta["licence"],
            )

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device
        LOGGER.info("loading %s onto %s", self.checkpoint, device)
        self._processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint).to(device).eval()

    # -------------------------------------------------------------- inference
    def predict_inverse_depth(self, image: np.ndarray) -> np.ndarray:
        """Relative inverse depth for one RGB image, at the image's own resolution.

        Larger values are nearer. The result is defined up to an unknown scale
        and shift -- it is **not** metres.
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            # accept float images in [0, 1], which is what the simulator renders
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        pil = Image.fromarray(arr[..., :3])

        inputs = self._processor(images=pil, return_tensors="pt").to(self._resolved_device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        predicted = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(arr.shape[0], arr.shape[1]),
            mode="bicubic",
            align_corners=False,
        )
        return predicted.squeeze().float().cpu().numpy()

    # ------------------------------------------------------- encoder protocol
    def encode(self, observation: Observation) -> GeometryFeature:
        """``F_t = E(I_t)`` from pixels.

        Raises rather than falling back to the oracle when there is no image:
        a run must never be able to claim this encoder without having run it.
        """
        image = getattr(observation, "image", None)
        if image is None:
            raise EncoderUnavailable(
                f"encoder {self.name!r} needs Observation.image, which is None. The synthetic "
                "benchmark stores landmark arrays and only a few preview renders "
                "(synthetic.render.preview_limit), so most scenes carry no image. Either enable "
                "rendering for every scene or use this encoder on external imagery. Falling back "
                "to the oracle here would let the run claim an encoder it did not run."
            )
        feature = observation.feature
        inv = self.predict_inverse_depth(np.asarray(image))
        h, w = inv.shape

        uv = feature.uv
        cols = np.rint(uv[:, 0]).astype(float)
        rows = np.rint(uv[:, 1]).astype(float)
        inside = (
            np.isfinite(cols) & np.isfinite(rows)
            & (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        )
        sampled = np.full(uv.shape[0], np.nan)
        if np.any(inside):
            sampled[inside] = inv[rows[inside].astype(int), cols[inside].astype(int)]

        depth = sampled
        self._last_alignment = None
        if self.align_to_reference:
            # Fit the two free parameters against the observation's own depths,
            # in inverse space where the model actually operates.
            with np.errstate(divide="ignore", invalid="ignore"):
                ref_inv = np.where(feature.depth > 1e-6, 1.0 / feature.depth, np.nan)
            scale, shift, report = align_scale_shift(sampled, ref_inv, valid=feature.visible & inside)
            aligned_inv = scale * sampled + shift
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = np.where(aligned_inv > 1e-9, 1.0 / aligned_inv, np.nan)
            self._last_alignment = {"scale": scale, "shift": shift, **report}

        # A landmark the network could not be sampled at is not visible. Marking
        # it visible with a NaN depth would push NaN into every downstream metric.
        visible = feature.visible & inside & np.isfinite(depth)
        return GeometryFeature(uv.copy(), depth, visible, feature.channel, feature.camera)

    # ------------------------------------------------------------- provenance
    def to_dict(self) -> dict[str, Any]:
        meta = CHECKPOINTS.get(self.checkpoint, {})
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "licence": meta.get("licence", "UNVERIFIED"),
            "commercial_use_permitted": meta.get("commercial_use"),
            "params_m": meta.get("params_m"),
            "device": self._resolved_device or self.device,
            "align_to_reference": self.align_to_reference,
            "output_units": "metres (scale/shift aligned)" if self.align_to_reference
            else "relative inverse depth (NOT metric)",
            "last_alignment": self._last_alignment,
        }
