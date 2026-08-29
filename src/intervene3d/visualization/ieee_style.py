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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

#: IEEE column widths in inches.
IEEE_SINGLE_COLUMN_IN = 3.5
IEEE_DOUBLE_COLUMN_IN = 7.16
GOLDEN = 1.618

#: ColorBrewer qualitative schemes (Brewer, Harrower & Pennsylvania State
#: University, https://colorbrewer2.org).  Two schemes, used for two jobs,
#: because one scheme cannot do both well:
#:
#: ``PALETTE`` -- **Dark2**, for lines, markers, edges and any text drawn in a
#:     series colour.  ColorBrewer flags Dark2 as colour-blind safe and it is
#:     dark enough to hold contrast against the light page.
#: ``FILL_PALETTE`` -- **Set3**, for large filled areas.  Set3 is the pastel
#:     qualitative scheme and reads quietly at bar-sized areas.
#:
#: The split is deliberate and worth stating, because Set3 alone would be a
#: regression: ColorBrewer marks **Set3 as NOT colour-blind safe**, and its
#: pastels lose contrast against a light ground when used for 1-px lines or
#: small markers -- exactly the legibility failures an earlier figure pass was
#: spent fixing.  Pairing a Set3 fill with its Dark2 edge keeps the pastel
#: surface the scheme is good at while the shape stays readable in greyscale
#: and under colour-vision deficiency.
PALETTE = ("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D")
FILL_PALETTE = ("#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69")
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)))

#: Stable colours for the optical mechanisms, used across every figure.
#: Stable per-mechanism colours, drawn from Dark2 so they match PALETTE.  These
#: are identities, not series indices: H_R must be the same colour in every
#: figure, so they are pinned here rather than taken from the cycle.
MECHANISM_COLORS = {
    "direct": "#7570B3",
    "emissive": "#D95F02",
    "reflection": "#1B9E77",
    "transmission": "#E7298A",
    "mixed": "#E6AB02",
    "abstain": "#9AA5AC",
}
#: Matching Set3 fills, indexed identically, for filled areas of each mechanism.
MECHANISM_FILLS = {
    "direct": "#BEBADA",
    "emissive": "#FDB462",
    "reflection": "#8DD3C7",
    "transmission": "#FB8072",
    "mixed": "#FFFFB3",
    "abstain": "#D9D9D9",
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
    "grid.linewidth": 0.35,
    "grid.alpha": 0.25,
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



def tint(colour: str, amount: float = 0.55) -> tuple[float, float, float]:
    """Mix ``colour`` toward white by ``amount``.

    Bar charts previously separated paired series with dense hatching, which
    prints as noise at journal scale.  A light tint against the full-strength
    edge of the same hue separates them by **lightness** instead -- which
    survives greyscale reproduction just as hatching did, without the texture.
    """
    r, g, b = mcolors.to_rgb(colour)
    a = float(np.clip(amount, 0.0, 1.0))
    return (r + (1.0 - r) * a, g + (1.0 - g) * a, b + (1.0 - b) * a)


def bar_style(index: int = 0, *, colour: str | None = None, emphasis: bool = False) -> dict[str, Any]:
    """Fill/edge pair for a bar series: Set3 body, Dark2 outline.

    The two ColorBrewer schemes are index-matched, so bar ``i`` is the Set3
    pastel drawn inside its Dark2 counterpart. Emphasis darkens toward the edge
    colour rather than lightening further -- on a light page the emphasised bar
    has to be the *darker* one, or emphasis reads as fading out.
    """
    base = colour or PALETTE[index % len(PALETTE)]
    if colour is None:
        body = tint(base, 0.42) if emphasis else FILL_PALETTE[index % len(FILL_PALETTE)]
    else:
        body = tint(base, 0.30 if emphasis else 0.62)
    return {"color": body, "edgecolor": base, "linewidth": 0.7}


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
