"""Gate 5: readers for the acquired external datasets.

Acquisition (``fetchers.py``) put bytes on disk. This module turns them into
Python objects, and it is the first code in the repository that reads anything
other than the synthetic simulator.

Four formats, because four publishers made four choices:

======================  ==================================================
``LayeredDepthReader``  parquet; ``image.png`` + ``tuples.json`` **ordinal**
                        multi-layer relations (pairs / triplets / quads with
                        an ``is_real`` flag). No metric depth at all.
``LayeredDepthSynReader``
                        parquet; ``image.png`` + ``depth_1..depth_8.png`` --
                        eight ray-ordered depth layers with true ground truth.
``TransPhy3DReader``    WebDataset ``.tar`` shards; per frame an ``image.png``,
                        ``depth.png`` + ``depth.json`` (a ``max_depth`` scale),
                        ``normal.png`` and ``metadata.json`` carrying **4x4
                        extrinsics and normalised 3x3 intrinsics**.  The only
                        acquired data with real observer motion.
``VisualIllusionReader``
                        one ``tar.gz``; ``test/{left,right}/*.png``,
                        ``test/disp/*.pfm`` float disparity, ``test/mask/*.jpg``,
                        and ``calib/*.yaml``.
======================  ==================================================

Design rules, all of which exist because the alternative silently corrupts a
result:

* **Nothing is fabricated.** A missing depth, an unreadable layer or an absent
  calibration yields ``None`` or a masked array -- never a zero, never an
  interpolated guess. ``LoaderError`` is raised for a genuinely broken file.
* **Metric scale is applied, not assumed.** TransPhy3D depth is a 16-bit PNG
  that means nothing without the per-frame ``max_depth``; the reader applies it
  and records that it did.
* **Every reader verifies against ``manifest.json``** on request, so a silently
  corrupted download cannot become a silently wrong number.
* **Readers are lazy.** These archives are 4-24 GB; nothing loads eagerly, and
  iteration streams.

``pyarrow`` is needed for the two parquet datasets and ``Pillow`` for image
decoding; both live in the optional ``data`` extra
(``pip install -e ".[data]"``). The core pipeline still needs neither.
"""

from __future__ import annotations

import io
import json
import logging
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.config.loader import repo_root
from intervene3d.reproducibility.hashing import sha256_file
from intervene3d.utils.io import load_json

LOGGER = logging.getLogger(__name__)


class LoaderError(RuntimeError):
    """A dataset is present but cannot be read as its format promises."""


class MissingDependency(LoaderError):
    """An optional reader dependency is absent.  Names the exact install."""


def _require(module: str, extra: str = "data"):
    """Import a module, or fail with the exact install command.

    ``importlib.import_module`` rather than ``__import__`` because the latter
    returns the top-level package, so ``__import__("PIL").Image`` raises
    ``AttributeError`` instead of importing the submodule.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MissingDependency(
            f"reading this dataset needs {module!r}, which is in the optional "
            f'{extra!r} extra: pip install -e ".[{extra}]"'
        ) from exc


# --------------------------------------------------------------------- samples
@dataclass(frozen=True)
class DepthLayers:
    """Ray-ordered depth layers for one image.

    ``layers`` is ``(L, H, W)``; ``valid`` marks where each layer carries a real
    measurement. A layer that the publisher did not provide is absent from
    ``layers`` entirely rather than being filled with zeros.
    """

    layers: np.ndarray
    valid: np.ndarray
    scale_note: str = ""

    @property
    def n_layers(self) -> int:
        return int(self.layers.shape[0])


@dataclass(frozen=True)
class ExternalSample:
    """One example from an external dataset, in this repository's vocabulary."""

    dataset: str
    key: str
    image: np.ndarray | None = None           # (H, W, 3) uint8
    depth: np.ndarray | None = None           # (H, W) float metres, NaN where unknown
    disparity: np.ndarray | None = None       # (H, W) float, NaN where unknown
    layers: DepthLayers | None = None
    intrinsics: np.ndarray | None = None      # (3, 3) pixels
    extrinsics: np.ndarray | None = None      # (4, 4) camera-from-world (world -> camera)
    ordinal_relations: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        bits = [f"{self.dataset}:{self.key}"]
        if self.image is not None:
            bits.append(f"image{tuple(self.image.shape)}")
        if self.depth is not None:
            finite = int(np.count_nonzero(np.isfinite(self.depth)))
            bits.append(f"depth{tuple(self.depth.shape)} {finite} finite")
        if self.disparity is not None:
            bits.append(f"disp{tuple(self.disparity.shape)}")
        if self.layers is not None:
            bits.append(f"{self.layers.n_layers} layers")
        if self.intrinsics is not None:
            bits.append("K")
        if self.extrinsics is not None:
            bits.append("T_wc")
        if self.ordinal_relations is not None:
            counts = count_relations(self.ordinal_relations)
            total = sum(counts.values())
            bits.append(f"{total} ordinal relations {counts}")
        return "  ".join(bits)



def count_relations(relations: dict[str, Any]) -> dict[str, int]:
    """Count ordinal relations in a LayeredDepth ``tuples.json``.

    The structure nests two levels -- ``{layer_all, layer_first} x {pairs, trips,
    quads}`` -- so a one-level ``len()`` sum silently reports zero.
    """
    counts: dict[str, int] = {}
    for group, kinds in (relations or {}).items():
        if not isinstance(kinds, dict):
            continue
        for kind, items in kinds.items():
            if isinstance(items, list) and items:
                counts[f"{group}.{kind}"] = len(items)
    return counts


# ---------------------------------------------------------------- base reader
class ExternalReader:
    """Common behaviour: locate the variant, verify it, refuse to guess."""

    dataset_key = ""

    def __init__(self, root: Path | str | None = None, *, variant: str | None = None) -> None:
        if root is None:
            if variant is None:
                raise LoaderError("pass either an explicit root or a variant name")
            root = repo_root() / "data" / "raw" / self.dataset_key / variant
        self.root = Path(root)
        if not self.root.exists():
            raise LoaderError(
                f"{self.dataset_key} variant not found at {self.root}. Acquire it with:\n"
                f"  bash scripts/download_datasets.sh --dataset {self.dataset_key} "
                f"--variant {variant or '<variant>'} --yes"
            )
        self.manifest_path = self.root / "manifest.json"

    # ------------------------------------------------------------- integrity
    def verify(self, *, full: bool = False) -> dict[str, Any]:
        """Check the download against its manifest before trusting a number."""
        if not self.manifest_path.exists():
            return {"verified": False, "reason": "no manifest.json; was this fetched by our tooling?"}
        manifest = load_json(self.manifest_path)
        recorded = manifest.get("files") or {}
        bad, missing = [], []
        for rel, digest in recorded.items():
            target = self.root / rel
            if not target.exists():
                missing.append(rel)
            elif full and sha256_file(target) != digest:
                bad.append(rel)
        return {
            "verified": not bad and not missing,
            "n_files": len(recorded),
            "missing": missing,
            "checksum_mismatches": bad,
            "checksums_recomputed": full,
            "revision": (manifest.get("source") or {}).get("revision_resolved"),
            "licence": manifest.get("licence_as_recorded"),
        }

    def provenance(self) -> dict[str, Any]:
        """What a result computed from this data must cite."""
        if not self.manifest_path.exists():
            return {"dataset": self.dataset_key, "root": str(self.root), "manifest": None}
        m = load_json(self.manifest_path)
        src = m.get("source") or {}
        return {
            "dataset": self.dataset_key,
            "variant": m.get("variant"),
            "root": str(self.root),
            "repo_id": src.get("repo_id"),
            "revision": src.get("revision_resolved"),
            "url": src.get("url"),
            "licence": m.get("licence_as_recorded"),
            "downloaded_utc": m.get("downloaded_utc"),
        }

    def __iter__(self) -> Iterator[ExternalSample]:  # pragma: no cover - overridden
        raise NotImplementedError


# ------------------------------------------------------------------- helpers
def _decode_image(payload: bytes) -> np.ndarray:
    Image = _require("PIL.Image")  # noqa: N806
    with Image.open(io.BytesIO(payload)) as im:
        return np.asarray(im.convert("RGB"))


def _decode_gray(payload: bytes) -> np.ndarray:
    Image = _require("PIL.Image")  # noqa: N806
    with Image.open(io.BytesIO(payload)) as im:
        return np.asarray(im)


def read_pfm(payload: bytes) -> np.ndarray:
    """Decode a PFM float map (3D Visual Illusion disparity).

    Returns a float32 array, top-down, with non-finite values preserved as NaN
    rather than replaced -- an invalid disparity is information, not noise.
    """
    stream = io.BytesIO(payload)
    header = stream.readline().rstrip()
    if header not in (b"Pf", b"PF"):
        raise LoaderError(f"not a PFM file (magic {header!r})")
    colour = header == b"PF"
    dims = stream.readline()
    while dims.startswith(b"#"):
        dims = stream.readline()
    match = re.match(rb"^(\d+)\s+(\d+)\s*$", dims)
    if not match:
        raise LoaderError(f"malformed PFM dimensions {dims!r}")
    width, height = int(match.group(1)), int(match.group(2))
    scale = float(stream.readline().rstrip())
    endian = "<" if scale < 0 else ">"
    count = width * height * (3 if colour else 1)
    data = np.frombuffer(stream.read(count * 4), dtype=endian + "f4", count=count)
    data = data.reshape((height, width, 3) if colour else (height, width))
    return np.flipud(data).astype(np.float32)


# ---------------------------------------------------------------- parquet base
class _ParquetReader(ExternalReader):
    """Shared streaming over the parquet shards a Hub dataset publishes."""

    def _shards(self) -> list[Path]:
        shards = sorted((self.root / "data").glob("*.parquet"))
        if not shards:
            raise LoaderError(f"no parquet shards under {self.root / 'data'}")
        return shards

    def _batches(self, columns: list[str] | None = None, batch_size: int = 8):
        pq = _require("pyarrow.parquet")

        for shard in self._shards():
            handle = pq.ParquetFile(shard)
            available = set(handle.schema_arrow.names)
            cols = [c for c in columns if c in available] if columns else None
            for batch in handle.iter_batches(batch_size=batch_size, columns=cols):
                yield shard, batch

    def __len__(self) -> int:
        import pyarrow.parquet as pq

        return sum(pq.ParquetFile(s).metadata.num_rows for s in self._shards())


# ------------------------------------------------------------- LayeredDepth
class LayeredDepthReader(_ParquetReader):
    """LayeredDepth real benchmark (CC0-1.0).

    **Ordinal, not metric.** Each example carries `tuples.json` with `layer_all`
    and `layer_first` relations: `pairs`, `trips` and `quads`, each point given
    as `[x, y, layer]` and each relation flagged `is_real`. There is no depth
    map, so any metric quoted against this dataset would be invented. Score it
    with ordinal agreement.
    """

    dataset_key = "layereddepth"

    def __init__(self, root: Path | str | None = None, *, variant: str = "validation") -> None:
        super().__init__(root, variant=variant)
        self.variant = variant

    def __iter__(self) -> Iterator[ExternalSample]:
        for _, batch in self._batches(["__key__", "image.png", "tuples.json"]):
            rows = batch.to_pylist()
            for i, row in enumerate(rows):
                img = row.get("image.png")
                tuples = row.get("tuples.json")
                if img is None:
                    raise LoaderError(f"{self.dataset_key}: row {i} has no image.png")
                yield ExternalSample(
                    dataset=self.dataset_key,
                    key=str(row.get("__key__") or i),
                    image=_decode_image(img["bytes"]),
                    ordinal_relations=tuples,
                    extra={
                        "variant": self.variant,
                        # Stated explicitly so nobody reaches for a depth metric.
                        "has_metric_depth": False,
                        "annotation": "ordinal multi-layer relations only",
                    },
                )


# --------------------------------------------------------- LayeredDepth-Syn
class LayeredDepthSynReader(_ParquetReader):
    """LayeredDepth-Syn (BSD-3-Clause): image + eight ray-ordered depth layers.

    The layer PNGs are 16-bit. The dataset card publishes no metric scale, so the
    values are returned **as stored** and `layers.scale_note` says so: treating
    raw 16-bit codes as metres would be a fabricated unit.
    """

    dataset_key = "layereddepth_syn"
    N_LAYERS = 8

    def __init__(self, root: Path | str | None = None, *, variant: str = "validation") -> None:
        super().__init__(root, variant=variant)
        self.variant = variant

    def __iter__(self) -> Iterator[ExternalSample]:
        cols = ["__key__", "image.png"] + [f"depth_{i}.png" for i in range(1, self.N_LAYERS + 1)]
        for _, batch in self._batches(cols, batch_size=4):
            for i, row in enumerate(batch.to_pylist()):
                img = row.get("image.png")
                if img is None:
                    raise LoaderError(f"{self.dataset_key}: row {i} has no image.png")
                planes, valid, present = [], [], []
                for layer in range(1, self.N_LAYERS + 1):
                    cell = row.get(f"depth_{layer}.png")
                    if cell is None:
                        continue  # absent layers are omitted, never zero-filled
                    arr = _decode_gray(cell["bytes"]).astype(np.float32)
                    planes.append(arr)
                    valid.append(arr > 0)
                    # Record WHICH layers these are. Compacting silently renumbers
                    # them, and the whole point of this dataset is that the layers
                    # are ray-ordered: planes[2] must not quietly mean layer 4.
                    present.append(layer)
                if not planes:
                    raise LoaderError(f"{self.dataset_key}: row {i} has no depth layers")
                yield ExternalSample(
                    dataset=self.dataset_key,
                    key=str(row.get("__key__") or i),
                    image=_decode_image(img["bytes"]),
                    layers=DepthLayers(
                        layers=np.stack(planes),
                        valid=np.stack(valid),
                        scale_note=(
                            "raw 16-bit PNG values as stored; the dataset card publishes no metric "
                            "scale factor, so these are NOT metres"
                        ),
                    ),
                    extra={
                        "variant": self.variant,
                        "n_layers_present": len(planes),
                        "layer_indices": present,
                    },
                )


# ----------------------------------------------------------------- TransPhy3D
class TransPhy3DReader(ExternalReader):
    """TransPhy3D (Apache-2.0): WebDataset shards of rendered video sequences.

    The important property for this project: every frame carries a **4x4 camera
    extrinsic**, so consecutive frames form a genuine ``(F_t, a) -> F_{t+1}``
    example with a known relative pose. :meth:`iter_pairs` yields exactly that.

    Depth is a 16-bit PNG normalised by a per-frame ``max_depth`` in the
    companion ``depth.json``; the reader applies it and records the conversion.
    Intrinsics are normalised (principal point at 0.5), so
    :meth:`ExternalSample.intrinsics` is scaled to pixels using the image size.
    """

    dataset_key = "transphy3d"

    def __init__(self, root: Path | str | None = None, *, variant: str = "sample") -> None:
        super().__init__(root, variant=variant)
        self.variant = variant

    def shards(self) -> list[Path]:
        shards = sorted(self.root.rglob("*.tar"))
        if not shards:
            raise LoaderError(f"no .tar shards under {self.root}")
        return shards

    @staticmethod
    def _group(names: list[str]) -> dict[str, dict[str, str]]:
        """Group WebDataset members by their sample stem."""
        grouped: dict[str, dict[str, str]] = {}
        for name in names:
            stem, _, suffix = name.partition(".")
            grouped.setdefault(stem, {})[suffix] = name
        return grouped

    def _sample_from(self, tar: tarfile.TarFile, stem: str, members: dict[str, str]) -> ExternalSample:
        def read(suffix: str) -> bytes | None:
            name = members.get(suffix)
            if name is None:
                return None
            handle = tar.extractfile(name)
            return handle.read() if handle else None

        image_bytes = read("image.png")
        if image_bytes is None:
            raise LoaderError(f"{self.dataset_key}: sample {stem} has no image.png")
        image = _decode_image(image_bytes)
        h, w = image.shape[:2]

        depth = None
        scale_note = ""
        depth_bytes = read("depth.png")
        meta_depth = read("depth.json")
        if depth_bytes is not None:
            raw = _decode_gray(depth_bytes).astype(np.float64)
            if meta_depth is not None:
                max_depth = float(json.loads(meta_depth.decode())["max_depth"])
                # Bit depth from the DTYPE, never from the values: a 16-bit frame
                # whose codes all happen to fall below 256 would otherwise be
                # scaled 257x too large, silently.
                decoded = _decode_gray(depth_bytes)
                full = float(np.iinfo(decoded.dtype).max) if np.issubdtype(decoded.dtype, np.integer) else 1.0
                depth = raw / full * max_depth
                scale_note = f"raw/{full:.0f} * max_depth({max_depth:.4f} m)"
                if max_depth <= 0.0:
                    # A degenerate frame, not an empty one. Say so rather than
                    # letting it vanish into an anonymous "skipped" count.
                    LOGGER.warning(
                        "%s: sample %s has max_depth=%s -- the depth frame is unusable",
                        self.dataset_key, stem, max_depth,
                    )
                    depth = np.full_like(raw, np.nan)
            else:
                # No scale means no metres. Refusing beats inventing a unit.
                raise LoaderError(
                    f"{self.dataset_key}: sample {stem} has depth.png but no depth.json, so its "
                    "metric scale is unknown; refusing to return unscaled values as depth"
                )

        intrinsics = extrinsics = None
        meta_bytes = read("metadata.json")
        if meta_bytes is not None:
            meta = json.loads(meta_bytes.decode())
            mats = meta.get("camera_matrices") or {}
            if mats.get("intrinsics") is not None:
                K = np.asarray(mats["intrinsics"], dtype=np.float64).copy()
                # The published K is normalised. BOTH focal lengths are normalised
                # by the same factor (the render has square pixels), so scaling
                # row 1 by H instead of W manufactures fy != fx -- for a 640x480
                # frame that is fy=454.3 against fx=605.7, a 4:3 pixel aspect no
                # Blender render has. Only the principal point is per-axis.
                # Verified photometrically on 2026-08-29: square K reprojects
                # better than the per-axis scaling.
                fx = K[0, 0] * w
                fy = K[1, 1] * w
                cx = K[0, 2] * w
                cy = K[1, 2] * h
                intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            if mats.get("extrinsics") is not None:
                extrinsics = np.asarray(mats["extrinsics"], dtype=np.float64)

        normal = read("normal.png")
        return ExternalSample(
            dataset=self.dataset_key,
            key=stem,
            image=image,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            extra={
                "variant": self.variant,
                "depth_scale": scale_note,
                "normal": _decode_image(normal) if normal else None,
                "sequence_id": stem.split("_")[0] if "_" in stem else stem,
            },
        )

    def __iter__(self) -> Iterator[ExternalSample]:
        for shard in self.shards():
            with tarfile.open(shard) as tar:
                grouped = self._group([m.name for m in tar.getmembers() if m.isfile()])
                for stem in sorted(grouped):
                    yield self._sample_from(tar, stem, grouped[stem])

    def iter_pairs(self, *, stride: int = 1, limit: int | None = None) -> Iterator[tuple[ExternalSample, ExternalSample, np.ndarray]]:
        """Yield ``(frame_t, frame_{t+stride}, T_rel)`` within a single shard.

        ``T_rel`` maps a point from **camera t's frame into camera t+1's frame**
        -- it is the action that separates the two observations, expressed so it
        can be applied directly with no further inversion. Pairs never straddle a shard, because each shard is
        one rendered sequence and a cross-shard pair would relate unrelated
        scenes by a meaningless transform.
        """
        emitted = 0
        for shard in self.shards():
            with tarfile.open(shard) as tar:
                grouped = self._group([m.name for m in tar.getmembers() if m.isfile()])
                stems = sorted(grouped)
                for a, b in zip(stems, stems[stride:], strict=False):
                    if limit is not None and emitted >= limit:
                        return
                    s0 = self._sample_from(tar, a, grouped[a])
                    s1 = self._sample_from(tar, b, grouped[b])
                    if s0.extrinsics is None or s1.extrinsics is None:
                        continue  # no pose, no action, no supervision
                    # metadata `extrinsics` is WORLD-TO-CAMERA, so moving a point
                    # from camera t into camera t+1 is E1 @ inv(E0). Establishing
                    # this empirically mattered: the camera-to-world reading
                    # reprojects WORSE than not moving the camera at all.
                    T_rel = s1.extrinsics @ np.linalg.inv(s0.extrinsics)
                    emitted += 1
                    yield s0, s1, T_rel


# ---------------------------------------------------------- 3D Visual Illusion
class VisualIllusionReader(ExternalReader):
    """3D Visual Illusion, real split (Apache-2.0).

    455 rectified stereo pairs under ``real/test/``: ``left``/``right`` PNGs,
    ``disp/*.pfm`` float disparity from a ZED rig cross-calibrated to an Intel
    RealSense L515, and ``mask/*.jpg``.

    The archive is a single 8.1 GB ``tar.gz``, which is not seekable, so
    iteration is a **single streaming pass**; random access would decompress the
    whole file per lookup. Call :meth:`extract_test_split` once if you need
    indexed access.
    """

    dataset_key = "visual_illusion_3d"

    def __init__(self, root: Path | str | None = None, *, variant: str = "real") -> None:
        super().__init__(root, variant=variant)
        self.variant = variant
        archives = sorted(self.root.glob("*.tar.gz"))
        if not archives:
            raise LoaderError(f"no .tar.gz under {self.root}")
        self.archive = archives[0]

    def calibration(self) -> dict[str, Any]:
        """The three calibration yamls, parsed."""
        yaml = _require("yaml")
        out: dict[str, Any] = {}
        expected = 3  # L515_calib, ZED_calib, L515_ZEDleft
        with tarfile.open(self.archive, "r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.startswith("real/calib/") and member.name.endswith(".yaml"):
                    handle = tar.extractfile(member)
                    if handle:
                        out[Path(member.name).stem] = yaml.safe_load(handle.read().decode())
                    if len(out) >= expected:
                        break  # the archive is 8.1 GB; do not stream the rest
        if not out:
            raise LoaderError("no calibration yaml found in the archive")
        return out

    #: Members are ``real/test/<kind>/<scene>/<frame>.<ext>``.
    _KINDS = ("left", "right", "disp", "mask")

    @staticmethod
    def _sample_key(parts: list[str]) -> str | None:
        """``<scene>/<frame>`` -- the identity of one stereo sample.

        Keying on the bare filename stem instead collides catastrophically: the
        archive holds 455 frames spread over 83 scene directories but reuses only
        28 frame names (``frame_0000``...``frame_0027``), so a stem key yields 28
        samples that each splice a left image from one scene onto a right image
        and disparity from another. The scene directory is not decoration.
        """
        if len(parts) < 5:
            return None
        return "/".join(parts[3:-1]) + "/" + Path(parts[-1]).stem

    def __iter__(self) -> Iterator[ExternalSample]:
        """Stream the test split, joining left/right/disp/mask per scene+frame."""
        pending: dict[str, dict[str, Any]] = {}
        decode = {
            "left": _decode_image, "right": _decode_image,
            "disp": read_pfm, "mask": _decode_gray,
        }
        seen_test = False
        with tarfile.open(self.archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if not member.name.startswith("real/test/"):
                    # The archive is 8.1 GB and not seekable; once past the test
                    # split there is nothing left to find.
                    if seen_test:
                        break
                    continue
                seen_test = True
                parts = member.name.split("/")
                kind = parts[2]
                key = self._sample_key(parts)
                if key is None or kind not in self._KINDS:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                slot = pending.setdefault(key, {})
                slot[kind] = decode[kind](handle.read())

                if set(self._KINDS) <= slot.keys():
                    yield self._emit(key, pending.pop(key))

        # Masks stream last in this archive, so anything still pending simply
        # never got all four kinds. Emit what exists rather than dropping it
        # silently, and say which kinds were missing.
        for key in sorted(pending):
            slot = pending[key]
            if {"left", "right", "disp"} <= slot.keys():
                yield self._emit(key, slot)
            else:
                LOGGER.warning(
                    "%s: sample %s is incomplete (have %s); skipped",
                    self.dataset_key, key, sorted(slot),
                )

    def _emit(self, key: str, slot: dict[str, Any]) -> ExternalSample:
        # A non-positive disparity is "unmeasured", not "at infinity".
        disp = slot["disp"]
        disp = np.where(np.isfinite(disp) & (disp > 0), disp, np.nan)
        scene = key.rsplit("/", 1)[0]
        return ExternalSample(
            dataset=self.dataset_key,
            key=key,
            image=slot["left"],
            disparity=disp,
            extra={
                "variant": self.variant,
                "scene": scene,
                "right": slot["right"],
                "mask": slot.get("mask"),
                "note": "rectified stereo; disparity in pixels, NaN where unmeasured",
            },
        )

    def extract_test_split(self, dest: Path | str | None = None) -> Path:
        """One-off extraction of ``real/test/`` for indexed access."""
        dest = Path(dest or (self.root / "extracted"))
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self.archive, "r|gz") as tar:
            for member in tar:
                if member.isfile() and member.name.startswith("real/test/"):
                    target = dest / Path(member.name).relative_to("real")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle = tar.extractfile(member)
                    if handle:
                        target.write_bytes(handle.read())
        return dest


READERS: dict[str, type[ExternalReader]] = {
    "layereddepth": LayeredDepthReader,
    "layereddepth_syn": LayeredDepthSynReader,
    "transphy3d": TransPhy3DReader,
    "visual_illusion_3d": VisualIllusionReader,
}


def get_reader(dataset: str, **kwargs) -> ExternalReader:
    """Reader for a registered dataset key."""
    if dataset not in READERS:
        raise LoaderError(f"no loader for {dataset!r}; available: {sorted(READERS)}")
    return READERS[dataset](**kwargs)
