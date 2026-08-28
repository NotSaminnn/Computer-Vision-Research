"""The conceptual overview figure.

Communicates the scientific loop in one panel::

    same apparent geometry -> multiple hypotheses -> controlled intervention
      -> different predicted consequences -> observation -> belief update
      -> resolved OR abstained

Drawn entirely from primitives so it stays vector, theme-consistent and
regenerable, and so no external diagram file can drift out of sync with the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import FigureSpec, ieee_style, new_figure

_STAGES = [
    ("same apparent\ngeometry", "$I_0$ matched", "#dfe7f2"),
    ("competing\nhypotheses", r"$\{H_D, H_E, H_R\}$", "#cfe0f0"),
    ("controlled\nintervention", r"$a=\Delta C\in SE(3)$", "#bfd9ee"),
    ("predicted\nconsequences", r"$W(F_t,H_k,a)$", "#afd2ec"),
    ("observation", r"$F_{t+1}=E(I_{t+1})$", "#9fcbea"),
    ("belief\nupdate", r"$p\!\propto\!e^{-\beta e_k}p$", "#8fc4e8"),
]

_OUTCOMES = [
    ("resolved", "physical explanation\nand contact geometry", "#a8dfc1"),
    ("unresolved", "explicit abstention:\n" r"$\mathcal{I}_\mathcal{A}<\epsilon$", "#f4c2a1"),
]


def plot_pipeline_overview(stem: Path | str, *, annotation: dict[str, Any] | None = None, **kw: Any) -> list[str]:
    """Draw the Intervene3D scientific loop."""
    annotation = annotation or {}
    with ieee_style():
        fig, ax = new_figure(FigureSpec("double", 0.34))
        ax.set_xlim(0, 108)
        ax.set_ylim(0, 34)
        ax.axis("off")

        n = len(_STAGES)
        box_w, box_h, gap = 12.0, 12.0, 2.6
        x0 = 1.5
        y_mid = 18.0
        centres = []
        for i, (title, sub, colour) in enumerate(_STAGES):
            x = x0 + i * (box_w + gap)
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x, y_mid - box_h / 2), box_w, box_h,
                    boxstyle="round,pad=0.35,rounding_size=1.2",
                    linewidth=0.6, edgecolor="#333333", facecolor=colour,
                )
            )
            ax.text(x + box_w / 2, y_mid + 2.3, title, ha="center", va="center", fontsize=6.2, fontweight="bold")
            ax.text(x + box_w / 2, y_mid - 2.6, sub, ha="center", va="center", fontsize=5.2)
            centres.append(x + box_w / 2)
            if i < n - 1:
                ax.annotate("", xy=(x + box_w + gap - 0.3, y_mid), xytext=(x + box_w + 0.3, y_mid),
                            arrowprops={"arrowstyle": "-|>", "linewidth": 0.7, "color": "#333333"})

        # branch to the two outcomes
        x_branch = x0 + n * (box_w + gap)
        for k, (title, sub, colour) in enumerate(_OUTCOMES):
            y = 27.0 if k == 0 else 9.0
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x_branch, y - 5.0), box_w + 2.0, 10.0,
                    boxstyle="round,pad=0.35,rounding_size=1.2",
                    linewidth=0.6, edgecolor="#333333", facecolor=colour,
                )
            )
            ax.text(x_branch + (box_w + 2.0) / 2, y + 2.1, title, ha="center", va="center",
                    fontsize=6.2, fontweight="bold")
            ax.text(x_branch + (box_w + 2.0) / 2, y - 1.9, sub, ha="center", va="center", fontsize=5.0)
            ax.annotate("", xy=(x_branch - 0.3, y), xytext=(centres[-1] + box_w / 2 + 0.3, y_mid),
                        arrowprops={"arrowstyle": "-|>", "linewidth": 0.7, "color": "#333333",
                                    "connectionstyle": "angle,angleA=0,angleB=90,rad=3"})

        # Feedback arrow: routed *below* the row so it never crosses a box.
        ax.annotate(
            "", xy=(centres[2], y_mid - box_h / 2 - 0.5), xytext=(centres[5], y_mid - box_h / 2 - 0.5),
            arrowprops={"arrowstyle": "-|>", "linewidth": 0.6, "color": "#777777", "linestyle": "--",
                        "connectionstyle": "arc3,rad=-0.35"},
        )
        ax.text((centres[2] + centres[5]) / 2, 3.2, "repeat while unresolved and budget remains",
                ha="center", va="center", fontsize=5.0, color="#666666")

        if annotation:
            ax.text(1.5, 32.4, annotation.get("caption", ""), ha="left", va="center", fontsize=5.4, color="#444444")
        return save_figure(fig, stem, **kw)


def plot_matched_variant_strip(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Reference view (identical) beside each variant's response to the same action.

    Top row: the three causal variants at ``C_0`` -- indistinguishable.
    Bottom row: the same three after the *same* controlled action -- separated.
    """
    variants = data["variants"]
    width = int(data.get("image_width", 320))
    height = int(data.get("image_height", 240))
    with ieee_style():
        fig, axes = new_figure(FigureSpec("double", 0.52), nrows=2, ncols=len(variants))
        axes = np.atleast_2d(axes)
        for col, entry in enumerate(variants):
            for row, key, tag in ((0, "reference", r"$C_0$"), (1, "after", r"$C_0\!\cdot\!a^*$")):
                ax = axes[row, col]
                uv = np.asarray(entry[key]["uv"], dtype=np.float64)
                vis = np.asarray(entry[key]["visible"], dtype=bool)
                ch = np.asarray(entry[key]["channel"], dtype=int)
                for c, colour, marker, size in ((0, "#0072B2", "o", 4), (1, "#7f7f7f", "s", 7), (2, "#D55E00", "*", 24)):
                    m = (ch == c) & vis
                    if np.any(m):
                        ax.scatter(uv[m, 0], uv[m, 1], s=size, c=colour, marker=marker, linewidths=0)
                ax.set_xlim(0, width)
                ax.set_ylim(height, 0)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(entry["label"], fontsize=6.2)
                if col == 0:
                    ax.set_ylabel(tag, fontsize=6.2)
        fig.suptitle("Identical at the reference view; separated by one controlled intervention", fontsize=7)
        return save_figure(fig, stem, **kw)
