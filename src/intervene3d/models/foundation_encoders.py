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
        "family": "depth-anything-v2",
        "output": "inverse_relative",
    },
    "depth-anything/Depth-Anything-V2-Base-hf": {
        "licence": "cc-by-nc-4.0",
        "commercial_use": False,
        "params_m": 97.5,
        "family": "depth-anything-v2",
        "output": "inverse_relative",
    },
    "depth-anything/Depth-Anything-V2-Large-hf": {
        "licence": "cc-by-nc-4.0",
        "commercial_use": False,
        "params_m": 335.3,
        "family": "depth-anything-v2",
        "output": "inverse_relative",
    },
    # Fine-tuned specifically for transparent surfaces -- the fairest opponent
    # for this evaluation, since it targets the exact failure mode.
    "depth-anything/prompt-depth-anything-vits-transparent-hf": {
        "licence": "apache-2.0",
        "commercial_use": True,
        "params_m": 24.8,
        "family": "depth-anything-v2",
        "output": "inverse_relative",
    },
    # A SECOND ARCHITECTURE FAMILY. Every checkpoint above is Depth-Anything, so
    # any finding drawn from them alone is a statement about one lineage rather
    # than about monocular depth. DepthPro is a different design from a different
    # group.
    #
    # Its output convention was VERIFIED rather than assumed, and the assumption
    # would have been wrong. DepthPro is documented as a *metric* depth model, so
    # the obvious reading is that ``predicted_depth`` holds metres and must be
    # inverted here. Measured against stereo disparity on a real sample:
    #
    #     raw predicted_depth        corr = +0.951   <- already inverse depth
    #     1 / predicted_depth        corr = -0.947
    #     post_process_...(outputs)  corr = -0.947   <- this is the metric one
    #
    # The raw attribute is canonical inverse depth; the processor's
    # ``post_process_depth_estimation`` is what converts to metres. Inverting it
    # would have flipped near and far while still producing entirely plausible
    # numbers, and every AUROC computed from it would have been quietly wrong.
    # DEPTH ANYTHING 3 (2025/26). A separate lineage from V2 with its own package
    # (`depth-anything-3`), its own API, and -- verified against stereo disparity
    # on this data -- the OPPOSITE output convention to DepthPro:
    #
    #     DA3 raw `depth`   corr = -0.842   <- metric depth, must be inverted
    #     1 / depth         corr = +0.840
    #
    # Two published checkpoints in a row, two different conventions, both
    # documented as "depth models". This is why every entry here records the
    # convention and why each was measured rather than read off a model card.
    #
    # DA3 also emits a native per-pixel confidence map, which makes it the
    # fairest available opponent for the confidence baseline: the model's own
    # trained uncertainty rather than a test-time-augmentation proxy we built.
    "depth-anything/DA3-SMALL": {
        "licence": "apache-2.0", "commercial_use": True, "params_m": 24.8,
        "family": "depth-anything-3", "output": "metric_depth",
    },
    "depth-anything/DA3-BASE": {
        "licence": "apache-2.0", "commercial_use": True, "params_m": 97.5,
        "family": "depth-anything-3", "output": "metric_depth",
    },
    "depth-anything/DA3-LARGE": {
        "licence": "cc-by-nc-4.0", "commercial_use": False, "params_m": 410.9,
        "family": "depth-anything-3", "output": "metric_depth",
    },
    "depth-anything/DA3-GIANT": {
        "licence": "cc-by-nc-4.0", "commercial_use": False, "params_m": 1100.0,
        "family": "depth-anything-3", "output": "metric_depth",
    },
    "depth-anything/DA3NESTED-GIANT-LARGE": {
        "licence": "cc-by-nc-4.0", "commercial_use": False, "params_m": 1400.0,
        "family": "depth-anything-3", "output": "metric_depth",
    },
    # THREE FURTHER FAMILIES, each with its own package and its own conventions.
    # All three predict METRIC DEPTH and are inverted here; each was verified
    # against stereo disparity on real data before being added:
    #
    #     MoGe-2       raw -0.986   1/x +0.985
    #     UniDepth-v2  raw -0.527   1/x +0.526
    #     Metric3D     see _predict_metric3d
    #
    # UniDepth's correlation is markedly weaker than the others. That is recorded
    # rather than smoothed over: on this hardware its optimised attention kernel
    # is unavailable (xformers has no sm_120 build), so it runs a fallback path,
    # and its numbers should be read as a lower bound on the published model.
    "Ruicheng/moge-2-vitl-normal": {
        "licence": "mit", "commercial_use": True, "params_m": 330.9,
        "family": "moge-2", "output": "metric_depth",
    },
    "lpiccinelli/unidepth-v2-vitl14": {
        "licence": "cc-by-nc-4.0", "commercial_use": False, "params_m": 353.8,
        "family": "unidepth-v2", "output": "metric_depth",
    },
    "metric3d/vit_small": {
        "licence": "bsd-2-clause", "commercial_use": True, "params_m": 37.5,
        "family": "metric3d", "output": "metric_depth",
    },
    "metric3d/vit_large": {
        "licence": "bsd-2-clause", "commercial_use": True, "params_m": 343.0,
        "family": "metric3d", "output": "metric_depth",
    },
    "apple/DepthPro-hf": {
        "licence": "apple-amlr",
        "commercial_use": False,
        "params_m": 952.0,
        "family": "depth-pro",
        "output": "inverse_relative",
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
    def _family(self) -> str:
        return CHECKPOINTS.get(self.checkpoint, {}).get("family", "")

    def _is_da3(self) -> bool:
        return self._family() == "depth-anything-3"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        loader = {
            "depth-anything-3": self._load_da3,
            "moge-2": self._load_moge,
            "unidepth-v2": self._load_unidepth,
            "metric3d": self._load_metric3d,
        }.get(self._family())
        if loader is not None:
            loader()
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

    def _load_da3(self) -> None:
        """Depth Anything 3 ships its own package and does not go through transformers."""
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise EncoderUnavailable(
                f"checkpoint {self.checkpoint!r} needs the `depth-anything-3` package, which is "
                "NOT installed in the main environment -- installing it there once corrupted "
                "numpy's metadata while a job was running. It lives in the isolated `.venv-da3` "
                "instead, and is run from there:  "
                ".venv-da3/Scripts/python.exe scripts/<script>.py "
                f"--checkpoint {self.checkpoint}"
            ) from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device
        meta = CHECKPOINTS.get(self.checkpoint, {})
        if meta and not meta["commercial_use"]:
            LOGGER.warning("checkpoint %r is %s -- NON-COMMERCIAL use only.",
                           self.checkpoint, meta["licence"])
        LOGGER.info("loading %s (depth-anything-3) onto %s", self.checkpoint, device)
        self._model = DepthAnything3.from_pretrained(self.checkpoint).to(device).eval()

    def _predict_da3(self, arr: np.ndarray) -> np.ndarray:
        """DA3 predicts at its own working resolution; resize to the input's."""
        import cv2

        prediction = self._model.inference([arr[..., :3]])
        depth = np.asarray(prediction.depth)[0].astype(np.float64)
        if depth.shape != arr.shape[:2]:
            depth = cv2.resize(depth, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
        # DA3 returns metric depth; this method's contract is inverse depth.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(depth > 1e-6, 1.0 / depth, np.nan)

    # ------------------------------------------------- other model families
    def _resolve_device(self) -> str:
        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._resolved_device = device
        meta = CHECKPOINTS.get(self.checkpoint, {})
        if meta and not meta["commercial_use"]:
            LOGGER.warning("checkpoint %r is %s -- NON-COMMERCIAL use only.",
                           self.checkpoint, meta["licence"])
        return device

    def _needs_models_venv(self, package: str) -> EncoderUnavailable:
        return EncoderUnavailable(
            f"checkpoint {self.checkpoint!r} needs the {package!r} package, which lives in the "
            "isolated .venv-models environment (installing model packages into the main venv "
            "once corrupted numpy while a job was running). Run from there:  "
            f".venv-models/Scripts/python.exe scripts/<script>.py --checkpoint {self.checkpoint}"
        )

    def _load_moge(self) -> None:
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise self._needs_models_venv("moge") from exc
        device = self._resolve_device()
        LOGGER.info("loading %s (moge-2) onto %s", self.checkpoint, device)
        self._model = MoGeModel.from_pretrained(self.checkpoint).to(device).eval()

    def _load_unidepth(self) -> None:
        try:
            from unidepth.models import UniDepthV2
        except ImportError as exc:
            raise self._needs_models_venv("unidepth") from exc
        device = self._resolve_device()
        LOGGER.info("loading %s (unidepth-v2) onto %s", self.checkpoint, device)
        self._model = UniDepthV2.from_pretrained(self.checkpoint).to(device).eval()

    def _load_metric3d(self) -> None:
        """Metric3D ships no package; its code is vendored under third_party/."""
        import sys
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[3]
        for extra in (root / "third_party" / "mmcv_shim", root / "third_party" / "metric3d"):
            if not extra.exists():
                raise EncoderUnavailable(
                    f"Metric3D needs {extra}, which is missing. Clone it with:  "
                    "git clone --depth 1 https://github.com/YvanYin/Metric3D.git third_party/metric3d"
                )
            if str(extra) not in sys.path:
                sys.path.insert(0, str(extra))
        try:
            import hubconf
        except ImportError as exc:
            raise self._needs_models_venv("mmengine (for Metric3D)") from exc
        device = self._resolve_device()
        variant = self.checkpoint.split("/", 1)[1]
        LOGGER.info("loading metric3d %s onto %s", variant, device)
        self._model = getattr(hubconf, f"metric3d_{variant}")(pretrain=True).to(device).eval()

    def _predict_moge(self, arr: np.ndarray) -> np.ndarray:
        import torch

        t = torch.tensor(arr[..., :3] / 255.0, dtype=torch.float32)
        t = t.permute(2, 0, 1).to(self._resolved_device)
        with torch.no_grad():
            depth = self._model.infer(t)["depth"].cpu().numpy()
        return self._to_inverse(depth, arr.shape[:2])

    def _predict_unidepth(self, arr: np.ndarray) -> np.ndarray:
        import torch

        t = torch.from_numpy(np.ascontiguousarray(arr[..., :3]))
        t = t.permute(2, 0, 1).to(self._resolved_device)
        with torch.no_grad():
            depth = self._model.infer(t)["depth"].squeeze().cpu().numpy()
        return self._to_inverse(depth, arr.shape[:2])

    def _predict_metric3d(self, arr: np.ndarray) -> np.ndarray:
        """Metric3D's own preprocessing: keep-ratio resize, pad, normalise.

        The published recipe ends with a de-canonical rescale by ``fx / 1000``,
        which needs the camera's focal length. That step is deliberately skipped
        here: it is a single multiplicative factor on depth, and every consumer in
        this repository fits a scale and shift before comparing anything, so the
        factor is absorbed exactly. Applying it with a guessed focal length would
        introduce an error the alignment would then silently hide.
        """
        import cv2
        import torch

        input_size = (616, 1064)
        h0, w0 = arr.shape[:2]
        scale = min(input_size[0] / h0, input_size[1] / w0)
        rgb = cv2.resize(arr[..., :3], (int(w0 * scale), int(h0 * scale)),
                         interpolation=cv2.INTER_LINEAR)
        pad_value = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h, pad_w = input_size[0] - h, input_size[1] - w
        top, left = pad_h // 2, pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, top, pad_h - top, left, pad_w - left,
                                 cv2.BORDER_CONSTANT, value=pad_value)
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        x = ((x - mean) / std)[None].to(self._resolved_device)
        with torch.no_grad():
            depth, _confidence, _ = self._model.inference({"input": x})
        depth = depth.squeeze()
        depth = depth[top : depth.shape[0] - (pad_h - top),
                      left : depth.shape[1] - (pad_w - left)]
        depth = torch.nn.functional.interpolate(depth[None, None], (h0, w0),
                                                mode="bilinear").squeeze()
        return self._to_inverse(depth.float().cpu().numpy(), (h0, w0))

    @staticmethod
    def _to_inverse(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Metric depth -> inverse depth at the input resolution."""
        import cv2

        depth = np.asarray(depth, dtype=np.float64)
        if depth.shape != shape:
            depth = cv2.resize(depth.astype(np.float32), (shape[1], shape[0]),
                               interpolation=cv2.INTER_CUBIC).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(depth > 1e-6, 1.0 / depth, np.nan)

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
        custom = {
            "depth-anything-3": self._predict_da3,
            "moge-2": self._predict_moge,
            "unidepth-v2": self._predict_unidepth,
            "metric3d": self._predict_metric3d,
        }.get(self._family())
        if custom is not None:
            a = arr
            if a.dtype != np.uint8:
                a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            if a.ndim == 2:
                a = np.stack([a] * 3, axis=-1)
            return custom(a)
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
            outputs.predicted_depth.reshape(1, 1, *outputs.predicted_depth.shape[-2:]),
            size=(arr.shape[0], arr.shape[1]),
            mode="bicubic",
            align_corners=False,
        )
        out = predicted.squeeze().float().cpu().numpy()

        # Different families can return different quantities under the same
        # attribute name, so the convention is looked up rather than assumed. A
        # model that reports metres must be inverted to satisfy this method's
        # contract; returning metres unchanged would flip the near/far ordering
        # and quietly reverse every downstream comparison. Every checkpoint
        # currently listed returns inverse depth, each verified against stereo
        # disparity -- see the DepthPro entry for why that check is not optional.
        if CHECKPOINTS.get(self.checkpoint, {}).get("output") == "metric_depth":
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(out > 1e-6, 1.0 / out, np.nan)
        return out

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
            "family": meta.get("family", "UNVERIFIED"),
            "output_convention": meta.get("output", "inverse_relative"),
            "device": self._resolved_device or self.device,
            "align_to_reference": self.align_to_reference,
            "output_units": "metres (scale/shift aligned)" if self.align_to_reference
            else "relative inverse depth (NOT metric)",
            "last_alignment": self._last_alignment,
        }
