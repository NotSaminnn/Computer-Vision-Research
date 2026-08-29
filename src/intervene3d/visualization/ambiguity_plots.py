"""Ambiguity and identifiability figures.

All of these read from saved result files; none of them recompute anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import (
    MECHANISM_COLORS,
    FigureSpec,
    bar_style,
    ieee_style,
    mechanism_label,
    new_figure,
    series_style,
)


def _relative_luminance(rgb: Sequence[float]) -> float:
    """WCAG relative luminance, which is gamma-corrected -- raw channel means are not."""
    c = np.asarray(rgb[:3], dtype=np.float64)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


#: The two candidate ink colours. Near-black rather than pure black so the glyph
#: does not vibrate against a saturated cell.
DARK_INK = "#111111"
LIGHT_INK = "#FFFFFF"


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    return tuple(int(v[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _contrast_ratio(fg: Sequence[float], bg: Sequence[float]) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def _prefers_dark_ink(rgb: Sequence[float]) -> bool:
    """True when :data:`DARK_INK` out-contrasts :data:`LIGHT_INK` on this cell.

    Compares the **actual** ink colours rather than idealised black and white:
    ``#111111`` is not pure black, and in the middle of a colormap -- where the
    two options are within a few percent of each other -- that difference decides
    which one wins. Choosing by achieved contrast rather than by a luminance
    threshold is what keeps the label readable across the whole map.
    """
    return _contrast_ratio(_hex_to_rgb(DARK_INK), rgb) >= _contrast_ratio(_hex_to_rgb(LIGHT_INK), rgb)


def plot_initial_view_similarity(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """How similar the causal variants look at the reference view.

    The premise of the whole project: if these bars were not at (numerically)
    zero, a single-frame classifier could solve the task and no intervention
    would be needed.
    """
    pairs = data["pairs"]
    values = np.asarray(data["reference_distance_px"], dtype=np.float64)
    post = np.asarray(data["best_action_distance_px"], dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.78))
        x = np.arange(len(pairs))
        floor = 1e-9

        # The reference-view values are EXACTLY zero -- that is the premise of the
        # whole project. Zero has no position on a log axis, so clamping it to the
        # floor drew nothing at all and the figure showed only the post-action
        # bars. Draw a visible stub at the floor and label it, so "identical at
        # C_0" is something the reader can see rather than infer from an absence.
        ax.bar(x - 0.2, np.full_like(values, floor * 4, dtype=float), width=0.4,
               label=r"reference view $C_0$", **bar_style(colour=MECHANISM_COLORS["direct"]))
        ax.bar(x + 0.2, np.maximum(post, floor), width=0.4, label=r"after best action $a^*$",
               **bar_style(colour=MECHANISM_COLORS["emissive"], emphasis=True))
        for xi, v in zip(x, values, strict=True):
            ax.annotate(r"$0$" if v == 0 else f"{v:.0e}", (xi - 0.2, floor * 4),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5.5,
                        color="#0072B2")

        if "epsilon_px" in data:
            ax.axhline(data["epsilon_px"], color="k", linestyle=":", linewidth=0.8,
                       label=rf"$\epsilon={data['epsilon_px']:g}$ px")
        ax.set_yscale("symlog", linthresh=floor)
        # Headroom so the tallest bar is not clipped by the axis frame.
        ax.set_ylim(0, float(np.nanmax(np.maximum(post, floor))) * 6.0)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=0)
        ax.set_ylabel(r"$\Delta_{ij}$ (px-equivalent)")
        ax.set_xlabel("hypothesis pair")
        ax.set_title("Matched at $C_0$, separated only by intervention")
        # Outside the axes: an in-plot legend overlapped the leftmost bar and the
        # epsilon line, hiding both.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=6,
                  frameon=False, columnspacing=1.2, handlelength=1.6)
        ax.set_title("Matched at $C_0$, separated only by intervention", pad=22)
        return save_figure(fig, stem, **kw)


def plot_separability_matrix(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Pairwise action-set identifiability ``I_A(H_i, H_j)`` for one scene."""
    matrix = np.asarray(data["identifiability_matrix"], dtype=np.float64)
    labels = [mechanism_label(m) for m in data["mechanisms"]]
    eps = float(data.get("epsilon_px", 1.0))
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.92))
        masked = np.ma.masked_where(np.eye(matrix.shape[0], dtype=bool), matrix)
        cmap_obj = plt.get_cmap("viridis").copy()
        cmap_obj.set_bad("#ECEFF1")      # the diagonal, tinted rather than blank
        im = ax.imshow(masked, cmap=cmap_obj, origin="upper")
        # Annotation colour must follow the CELL luminance, not a fixed choice:
        # viridis runs from near-black to bright yellow, so a hardcoded white
        # label is invisible on every high-value cell.
        finite = masked.compressed()
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        span = (hi - lo) or 1.0
        cmap = im.get_cmap()
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    continue
                below = matrix[i, j] < eps
                on_light = _prefers_dark_ink(cmap((matrix[i, j] - lo) / span))
                if below:
                    # Unresolvable pairs stay distinct, but readably so on either ground.
                    colour = "#8B0000" if on_light else "#FF8A80"
                else:
                    colour = DARK_INK if on_light else LIGHT_INK
                # Viridis' mid-range sits near mid-luminance, where the best
                # achievable ink contrast is only ~4.3:1 -- short of WCAG AA. A
                # thin halo in the opposite tone lifts every cell clear of it.
                halo = LIGHT_INK if on_light else "#000000"
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=10,
                        color=colour, fontweight="bold" if below else "normal",
                        path_effects=[pe.withStroke(linewidth=1.6, foreground=halo)])
        # The diagonal compares a hypothesis with itself: undefined, not missing.
        # Bare white read as absent data, so mark it.
        for i in range(matrix.shape[0]):
            ax.text(i, i, "—", ha="center", va="center", fontsize=11, color="#8B979F")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.tick_params(length=0)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\mathcal{I}_{\mathcal{A}}(H_i,H_j)$ (px-eq.)", fontsize=8.5)
        cbar.ax.tick_params(labelsize=7.5)
        # Explain the red convention only when a red cell is actually present:
        # a legend for something absent from the plot is noise.
        off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
        note = "; red = unresolvable" if bool(np.any(matrix[off_diagonal] < eps)) else ""
        ax.set_title(f"Action-set identifiability\n{data.get('scene_id', '')}"
                     rf" ($\epsilon={eps:g}${note}; — = self)", fontsize=9)
        return save_figure(fig, stem, **kw)


def plot_hypothesis_probabilities(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Posterior over optical hypotheses, before and after the intervention."""
    mechanisms = data["mechanisms"]
    trajectory = np.asarray(data["belief_trajectory"], dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.72))
        x = np.arange(len(mechanisms))
        width = 0.8 / max(trajectory.shape[0], 1)
        for step in range(trajectory.shape[0]):
            ax.bar(
                x + (step - (trajectory.shape[0] - 1) / 2) * width,
                trajectory[step],
                width=width,
                label=f"$t={step}$" + (" (prior)" if step == 0 else ""),
                color=series_style(step)["color"],
                edgecolor="k",
                linewidth=0.4,
            )
        if "tau" in data:
            ax.axhline(data["tau"], color="k", linestyle=":", linewidth=0.8, label=rf"$\tau={data['tau']:g}$")
        ax.set_xticks(x)
        ax.set_xticklabels([mechanism_label(m) for m in mechanisms], fontsize=6)
        ax.set_ylabel(r"$p(H_k \mid I_{1:t})$")
        ax.set_ylim(0, 1.05)
        title = data.get("scene_id", "")
        if data.get("abstained"):
            title += "  --  ABSTAINED"
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=6, ncol=2)
        return save_figure(fig, stem, **kw)


def plot_resolvability_distribution(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Distribution of scene-level identifiability, split by ground-truth label."""
    scores = np.asarray(data["identifiability_scores"], dtype=np.float64)
    labels = np.asarray(data["resolvable"], dtype=bool)
    eps = float(data.get("epsilon_px", 1.0))
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.68))
        bins = np.linspace(0, max(float(np.nanpercentile(scores, 97)), eps * 3), 26)
        ax.hist(scores[labels], bins=bins, label="resolvable ($y_{id}=1$)",
                **bar_style(colour=MECHANISM_COLORS["direct"]))
        ax.hist(scores[~labels], bins=bins, label="non-identifiable ($y_{id}=0$)",
                **bar_style(colour=MECHANISM_COLORS["emissive"], emphasis=True))
        ax.axvline(eps, color="k", linestyle=":", linewidth=0.9, label=rf"$\epsilon={eps:g}$ px")
        ax.set_xlabel(r"$\min_{j}\ \mathcal{I}_{\mathcal{A}}(H^*,H_j)$ (px-eq.)")
        ax.set_ylabel("scenes")
        ax.set_title("The benchmark contains genuinely unresolvable cases", fontsize=7)
        ax.legend(fontsize=6)
        return save_figure(fig, stem, **kw)


def plot_uncertainty_decomposition(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """``U_prediction`` versus ``U_identifiability`` -- the paper's key distinction."""
    up = np.asarray(data["prediction_uncertainty"], dtype=np.float64)
    ui = np.asarray(data["identifiability_uncertainty"], dtype=np.float64)
    resolvable = np.asarray(data["resolvable"], dtype=bool)
    # Both quantities are near-discrete here, so points pile up exactly on top of
    # one another.  Sizing by multiplicity keeps the plot honest: without it a
    # cluster of forty scenes is indistinguishable from a single one.
    def _collapse(x, y):
        pts = np.stack([np.round(x, 6), np.round(y, 6)], axis=1)
        uniq, counts = np.unique(pts, axis=0, return_counts=True)
        return uniq[:, 0], uniq[:, 1], counts

    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.80))
        for mask, colour, marker, label in (
            (resolvable, "#0072B2", "o", "resolvable"),
            (~resolvable, "#D55E00", "^", "non-identifiable"),
        ):
            if not np.any(mask):
                continue
            ux, uy, counts = _collapse(up[mask], ui[mask])
            ax.scatter(ux, uy, s=8 + 4.0 * counts, label=f"{label} (n={int(mask.sum())})",
                       color=colour, marker=marker, alpha=0.75, linewidths=0)
            for x0, y0, c in zip(ux, uy, counts, strict=True):
                if c > 1:
                    ax.annotate(str(int(c)), (x0, y0), textcoords="offset points",
                                xytext=(5, 4), fontsize=4.6, color=colour)
        ax.set_xlabel(r"$U_{\mathrm{prediction}}$  (normalised posterior entropy)")
        ax.set_ylabel(r"$U_{\mathrm{identifiability}}$")
        ax.set_title("Two uncertainties that must not be conflated", fontsize=7)
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.05, 1.08)
        ax.legend(fontsize=5.5, loc="center left")
        return save_figure(fig, stem, **kw)


def plot_pairwise_bar(pairs: Sequence[str], values: Sequence[float], stem: Path | str,
                      *, ylabel: str, title: str, epsilon: float | None = None, **kw: Any) -> list[str]:
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.68))
        ax.bar(range(len(pairs)), values, color="#0072B2", edgecolor="k", linewidth=0.4)
        if epsilon is not None:
            ax.axhline(epsilon, color="k", linestyle=":", linewidth=0.8, label=rf"$\epsilon={epsilon:g}$")
            ax.legend(fontsize=6)
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, fontsize=6, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7)
        return save_figure(fig, stem, **kw)
