"""Ambiguity and identifiability figures.

All of these read from saved result files; none of them recompute anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import (
    FigureSpec,
    ieee_style,
    mechanism_label,
    new_figure,
    series_style,
)


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
        fig, ax = new_figure(FigureSpec("single", 0.72))
        x = np.arange(len(pairs))
        ax.bar(x - 0.2, np.maximum(values, 1e-12), width=0.4, label="reference view $C_0$",
               color="#0072B2", edgecolor="k", linewidth=0.4)
        ax.bar(x + 0.2, np.maximum(post, 1e-12), width=0.4, label="after best action $a^*$",
               color="#D55E00", edgecolor="k", linewidth=0.4, hatch="//")
        if "epsilon_px" in data:
            ax.axhline(data["epsilon_px"], color="k", linestyle=":", linewidth=0.8,
                       label=rf"$\epsilon={data['epsilon_px']:g}$ px")
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=0)
        ax.set_ylabel(r"$\Delta_{ij}$ (px-equivalent)")
        ax.set_xlabel("hypothesis pair")
        ax.set_title("Matched at $C_0$, separated only by intervention")
        ax.legend(loc="upper left", fontsize=6)
        return save_figure(fig, stem, **kw)


def plot_separability_matrix(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Pairwise action-set identifiability ``I_A(H_i, H_j)`` for one scene."""
    matrix = np.asarray(data["identifiability_matrix"], dtype=np.float64)
    labels = [mechanism_label(m) for m in data["mechanisms"]]
    eps = float(data.get("epsilon_px", 1.0))
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.92))
        masked = np.ma.masked_where(np.eye(matrix.shape[0], dtype=bool), matrix)
        im = ax.imshow(masked, cmap="viridis", origin="upper")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    continue
                below = matrix[i, j] < eps
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if not below else "#ff4d4d",
                        fontweight="bold" if below else "normal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\mathcal{I}_{\mathcal{A}}(H_i,H_j)$ (px-eq.)", fontsize=6)
        ax.set_title(f"Action-set identifiability\n{data.get('scene_id', '')}"
                     rf" ($\epsilon={eps:g}$; red = unresolvable)", fontsize=7)
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
        ax.hist(scores[labels], bins=bins, alpha=0.75, label="resolvable ($y_{id}=1$)",
                color="#0072B2", edgecolor="k", linewidth=0.3)
        ax.hist(scores[~labels], bins=bins, alpha=0.75, label="non-identifiable ($y_{id}=0$)",
                color="#D55E00", edgecolor="k", linewidth=0.3, hatch="//")
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
