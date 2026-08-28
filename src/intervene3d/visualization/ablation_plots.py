"""Ablation and generalisation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import FigureSpec, ieee_style, new_figure, series_style


def plot_ablation_grid(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """One grouped bar chart per metric across the ablation variants."""
    variants = list(data["variants"])
    metrics = list(data["metrics"])
    values = np.asarray(data["values"], dtype=np.float64)  # (metric, variant)
    errors = np.asarray(data["ci95"], dtype=np.float64) if "ci95" in data else None
    with ieee_style():
        fig, axes = new_figure(FigureSpec("double", 0.34), ncols=len(metrics), sharey=False)
        axes = np.atleast_1d(axes)
        x = np.arange(len(variants))
        for k, (ax, metric) in enumerate(zip(axes, metrics, strict=False)):
            err = None if errors is None else np.nan_to_num(errors[k])
            ax.bar(x, values[k], yerr=err, capsize=1.6,
                   color=[series_style(i)["color"] for i in range(len(variants))],
                   edgecolor="k", linewidth=0.35, error_kw={"elinewidth": 0.55})
            ax.set_xticks(x)
            ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=5.5)
            ax.set_title(metric, fontsize=6.5)
        axes[0].set_ylabel(data.get("ylabel", "value"), fontsize=6.5)
        return save_figure(fig, stem, **kw)


def plot_generalisation(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Metric under increasing distribution shift (e.g. action-execution noise)."""
    x = np.asarray(data["x"], dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.70))
        for i, series in enumerate(data["series"]):
            style = series_style(i)
            y = np.asarray(series["y"], dtype=np.float64)
            ax.plot(x, y, label=series["label"], markersize=2.8, linewidth=1.0, **style)
            if "ci95" in series:
                ci = np.nan_to_num(np.asarray(series["ci95"], dtype=np.float64))
                ax.fill_between(x, y - ci, y + ci, color=style["color"], alpha=0.15, linewidth=0)
        ax.set_xlabel(data.get("xlabel", "shift magnitude"))
        ax.set_ylabel(data.get("ylabel", "metric"))
        ax.set_title(data.get("title", "Generalisation"), fontsize=7)
        ax.legend(fontsize=5.5)
        return save_figure(fig, stem, **kw)
