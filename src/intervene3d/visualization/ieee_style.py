"""Centralised IEEE-style plotting configuration.

Every figure in the repository is produced through this module.  Individual
experiments never touch ``matplotlib.rcParams``, so typography, sizing and
colour stay consistent and a style change is a one-line edit rather than a
repository-wide sweep.

Design constraints (research plan section 24):

* single- and double-column IEEE layouts, sized in inches;
* readable fonts at print size;
* vector PDF output by default, high-resolution raster alongside;
* consistent typography and mathematical notation;
* **grayscale-safe**: every series is distinguished by marker *and* linestyle as
  well as colour, and the categorical palette is monotonic in luminance;
* no chartjunk.

``SciencePlots``' ``["science", "ieee", "no-latex"]`` is used when available.
The ``no-latex`` variant matters: the plain ``science`` style sets
``text.usetex=True``, which fails on machines without a LaTeX installation.  A
self-contained fallback keeps figure generation working either way, and which
path was taken is reported by :func:`style_provenance` and recorded in run
manifests, so a figure's appearance is never a mystery.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

#: IEEE column widths in inches.
IEEE_SINGLE_COLUMN_IN = 3.5
IEEE_DOUBLE_COLUMN_IN = 7.16
GOLDEN = 1.618

#: Grayscale-safe categorical palette, ordered by increasing luminance contrast.
PALETTE = ("#1b1b1b", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)))

#: Stable colours for the optical mechanisms, used across every figure.
MECHANISM_COLORS = {
    "direct": "#0072B2",
    "emissive": "#D55E00",
    "reflection": "#009E73",
    "transmission": "#CC79A7",
    "mixed": "#E69F00",
    "abstain": "#7f7f7f",
}
MECHANISM_LABELS = {
    "direct": r"$H_D$ direct",
    "emissive": r"$H_E$ display",
    "reflection": r"$H_R$ mirror",
    "transmission": r"$H_T$ glass",
    "mixed": r"$H_M$ mixed",
    "abstain": "abstain",
}

_FALLBACK_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.prop_cycle": plt.cycler(color=list(PALETTE)),
}

_HAS_SCIENCEPLOTS: bool | None = None


def _sciencePlots_available() -> bool:
    global _HAS_SCIENCEPLOTS
    if _HAS_SCIENCEPLOTS is None:
        try:
            import scienceplots  # noqa: F401

            _HAS_SCIENCEPLOTS = all(s in plt.style.available for s in ("science", "ieee", "no-latex"))
        except Exception:  # noqa: BLE001  # pragma: no cover
            _HAS_SCIENCEPLOTS = False
    return bool(_HAS_SCIENCEPLOTS)


def style_provenance() -> dict[str, Any]:
    """Which style path is in use -- archived with every figure-producing run."""
    return {
        "scienceplots": _sciencePlots_available(),
        "styles": ["science", "ieee", "no-latex"] if _sciencePlots_available() else ["intervene3d-fallback"],
        "matplotlib": matplotlib.__version__,
        "usetex": False,
        "note": (
            "SciencePlots' IEEE style with no-latex (no LaTeX installation required)"
            if _sciencePlots_available()
            else "SciencePlots unavailable; using the self-contained IEEE-like fallback rcParams"
        ),
    }


@contextlib.contextmanager
def ieee_style(**overrides: Any) -> Iterator[None]:
    """Context manager applying the IEEE style for the enclosed block."""
    rc = dict(_FALLBACK_RC)
    rc.update(overrides)
    if _sciencePlots_available():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with plt.style.context(["science", "ieee", "no-latex"]):
                # Re-apply the parts we insist on regardless of the base style.
                with plt.rc_context({k: v for k, v in rc.items() if k not in ("font.serif",)}):
                    yield
    else:  # pragma: no cover - exercised only when SciencePlots is missing
        with plt.rc_context(rc):
            yield


@dataclass(frozen=True)
class FigureSpec:
    """Declarative figure size in IEEE column units."""

    column: str = "single"  # "single" | "double"
    height_ratio: float = 1.0 / GOLDEN
    width_scale: float = 1.0

    @property
    def size(self) -> tuple[float, float]:
        base = IEEE_SINGLE_COLUMN_IN if self.column == "single" else IEEE_DOUBLE_COLUMN_IN
        width = base * self.width_scale
        return (width, width * self.height_ratio)


def new_figure(
    spec: FigureSpec | None = None, *, nrows: int = 1, ncols: int = 1, **subplot_kw: Any
):
    """Create a correctly sized IEEE figure and its axes."""
    spec = spec or FigureSpec()
    fig, axes = plt.subplots(nrows, ncols, figsize=spec.size, **subplot_kw)
    return fig, axes


def series_style(index: int) -> dict[str, Any]:
    """Colour + marker + linestyle for series ``index`` -- grayscale-safe by design."""
    return {
        "color": PALETTE[index % len(PALETTE)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
    }


def mechanism_style(mechanism: str, index: int = 0) -> dict[str, Any]:
    return {
        "color": MECHANISM_COLORS.get(mechanism, PALETTE[index % len(PALETTE)]),
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
    }


def mechanism_label(mechanism: str) -> str:
    return MECHANISM_LABELS.get(mechanism, mechanism)


#: Compact display names.  Method keys stay long and explicit in the data files;
#: only the figures abbreviate, so a plot never invents a name the metrics lack.
METHOD_SHORT_NAMES = {
    "single_frame_classifier": "single-frame",
    "passive_multiview_classifier": "passive MV",
    "random_intervention": "random",
    "max_baseline_intervention": "max-baseline",
    "entropy_nbv": "generic NBV",
    "intervene3d_no_hypothesis_conditioning": "no H-cond.",
    "intervene3d_no_abstention": "ours (forced)",
    "intervene3d_noisy_encoder": "ours (noisy enc.)",
    "intervene3d": "ours",
}


def method_label(name: str) -> str:
    """Compact label for a method, falling back to the raw key."""
    return METHOD_SHORT_NAMES.get(name, name.replace("_", " "))
