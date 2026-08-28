"""Intervention figures: baselines, action utility, trajectories and regret."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import (
    FigureSpec,
    ieee_style,
    method_label,
    new_figure,
    series_style,
)


def plot_separability_vs_baseline(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Separability as a function of lateral baseline, with the MCRB marked.

    The single most informative figure about *when* an ambiguity resolves.
    """
    baselines = np.asarray(data["baselines"], dtype=np.float64)
    curves = data["curves"]
    eps = float(data.get("epsilon_px", 1.0))
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.74))
        for i, curve in enumerate(curves):
            style = series_style(i)
            y = np.asarray(curve["separability"], dtype=np.float64)
            ax.plot(baselines, y, label=curve["label"], markevery=max(len(baselines) // 8, 1),
                    markersize=2.6, linewidth=1.0, **style)
            mcrb = curve.get("mcrb")
            if mcrb:
                ax.axvline(mcrb, color=style["color"], linestyle=style["linestyle"], linewidth=0.6, alpha=0.55)
                ax.plot([mcrb], [eps], marker="*", markersize=6, color=style["color"], linestyle="none")
        ax.axhline(eps, color="k", linestyle=":", linewidth=0.9, label=rf"$\epsilon={eps:g}$ px")
        ax.set_xlabel(r"lateral baseline $B$ (m)")
        ax.set_ylabel(r"$\Delta_{ij}(a(B))$ (px-eq.)")
        ax.set_yscale("symlog", linthresh=max(eps / 4, 1e-3))
        # Separability is non-negative, so the symlog axis must not reserve half
        # its height for a region no datum can occupy.
        ax.set_ylim(bottom=0.0)
        ax.set_title(r"Separability vs baseline; $\star$ marks the MCRB", fontsize=7)
        ax.legend(fontsize=5.5, loc="lower right")
        return save_figure(fig, stem, **kw)


def plot_action_utility(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Predicted utility of each candidate action, with the selected one highlighted."""
    names = list(data["action_names"])
    utility = np.asarray(data["utility"], dtype=np.float64)
    chosen = int(data.get("selected_index", int(np.argmax(utility))))
    oracle = data.get("oracle_index")
    order = np.argsort(-utility)[: int(data.get("top_k", 14))]
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.86))
        colors = []
        for idx in order:
            if idx == chosen:
                colors.append("#D55E00")
            elif oracle is not None and idx == oracle:
                colors.append("#009E73")
            else:
                colors.append("#9ecae1")
        y = np.arange(len(order))
        ax.barh(y, utility[order], color=colors, edgecolor="k", linewidth=0.35)
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order], fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel(r"expected separability $\sum_{i<j} p_i p_j \Delta_{ij}(a)$")
        ax.set_title(f"Intervention selection -- {data.get('scene_id','')}\n"
                     "orange = selected, green = oracle optimum", fontsize=6.5)
        return save_figure(fig, stem, **kw)


def plot_camera_trajectory(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Candidate and executed camera poses in the reference-camera frame."""
    candidates = np.asarray(data["candidate_translations"], dtype=np.float64)
    executed = np.asarray(data.get("executed_translations", np.zeros((0, 3))), dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.85))
        ax.scatter(candidates[:, 0], candidates[:, 2], s=9, color="#9ecae1", marker="o",
                   linewidths=0, label="candidate actions $\\mathcal{A}$")
        ax.scatter([0], [0], s=30, color="k", marker="*", label="reference $C_0$", zorder=5)
        if executed.size:
            ax.plot(np.r_[0, executed[:, 0]], np.r_[0, executed[:, 2]], color="#D55E00",
                    marker="D", markersize=3.2, linewidth=1.0, label="executed $a^*$", zorder=4)
        ax.set_xlabel("lateral $x$ (m, reference camera frame)")
        ax.set_ylabel("forward $z$ (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("Bounded action set and the selected intervention", fontsize=7)
        ax.legend(fontsize=5.5, loc="best")
        return save_figure(fig, stem, **kw)


def plot_intervention_regret(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Normalised regret per method: how far from the oracle-optimal experiment."""
    methods = [method_label(m) for m in data["methods"]]
    means = np.asarray(data["mean"], dtype=np.float64)
    errs = np.asarray(data.get("ci95", np.zeros(len(methods))), dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.68))
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=np.nan_to_num(errs), capsize=2, color="#0072B2",
               edgecolor="k", linewidth=0.4, error_kw={"elinewidth": 0.6})
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=5.5)
        ax.set_ylabel(r"normalised regret  $1-\Delta(\hat a)/\Delta(a^*)$")
        ax.set_title("Intervention regret (lower is better)", fontsize=7)
        return save_figure(fig, stem, **kw)


def plot_motion_cost(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """RQ3: does hypothesis-aware selection resolve with less motion?"""
    methods = [method_label(m) for m in data["methods"]]
    means = np.asarray(data["mean"], dtype=np.float64)
    errs = np.asarray(data.get("ci95", np.zeros(len(methods))), dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.68))
        x = np.arange(len(methods))
        ax.bar(x, means, yerr=np.nan_to_num(errs), capsize=2, color="#009E73",
               edgecolor="k", linewidth=0.4, error_kw={"elinewidth": 0.6})
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=5.5)
        ax.set_ylabel("motion cost (m-equivalent)")
        ax.set_title("Camera motion spent per scene", fontsize=7)
        return save_figure(fig, stem, **kw)


def plot_predicted_vs_observed(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Predicted geometric response under each hypothesis versus what was observed."""
    observed = np.asarray(data["observed_uv"], dtype=np.float64)
    predictions = data["predictions"]
    ref = np.asarray(data.get("reference_uv", observed), dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.88))
        ok = np.all(np.isfinite(ref), axis=1) & np.all(np.isfinite(observed), axis=1)
        ax.quiver(ref[ok, 0], ref[ok, 1], (observed - ref)[ok, 0], (observed - ref)[ok, 1],
                  angles="xy", scale_units="xy", scale=1.0, width=0.003, color="0.55",
                  label="observed displacement")
        for i, pred in enumerate(predictions):
            uv = np.asarray(pred["uv"], dtype=np.float64)
            m = ok & np.all(np.isfinite(uv), axis=1)
            style = series_style(i)
            ax.scatter(uv[m, 0], uv[m, 1], s=6, marker=style["marker"], color=style["color"],
                       label=f"predicted {pred['label']}", linewidths=0, alpha=0.85)
        ax.scatter(observed[ok, 0], observed[ok, 1], s=12, marker="x", color="k",
                   label="observed", linewidths=0.6)
        ax.invert_yaxis()
        ax.set_xlabel("$u$ (px)")
        ax.set_ylabel("$v$ (px)")
        ax.set_title(f"Predicted vs observed response to $a^*$\n{data.get('scene_id','')}", fontsize=6.5)
        ax.legend(fontsize=5.0, loc="best", ncol=2)
        return save_figure(fig, stem, **kw)
