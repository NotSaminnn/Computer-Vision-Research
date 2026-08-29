"""Headline metric figures: CEA, contact depth, identifiability AUROC, MCRB, FPCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import (
    FigureSpec,
    bar_style,
    ieee_style,
    mechanism_label,
    method_label,
    new_figure,
    series_style,
)


def _grouped_bar(ax, groups, series, values, errors=None, ylabel=""):
    """Grouped bars separated by hue and lightness rather than by hatching.

    Dense hatch fills (``//``, ``xx``) print as moire at journal column width;
    a light body against a full-strength edge of the same hue separates the
    series just as well in greyscale and reads far quieter.
    """
    n_s = len(series)
    width = 0.8 / max(n_s, 1)
    x = np.arange(len(groups))
    for i, name in enumerate(series):
        offs = (i - (n_s - 1) / 2) * width
        err = None if errors is None else np.nan_to_num(np.asarray(errors[i], dtype=np.float64))
        row = np.asarray(values[i], dtype=np.float64)
        ax.bar(x + offs, row, width=width, label=name, yerr=err, capsize=1.6,
               **bar_style(i), error_kw={"elinewidth": 0.55, "ecolor": "#55606A"})
        # A zero-height bar is invisible; say so rather than leave a mystery gap.
        for xi, v in zip(x + offs, row, strict=True):
            if np.isfinite(v) and v <= 1e-9:
                ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=4.2, rotation=90, color="0.25")
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    return ax


def plot_explanation_accuracy(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Causal Explanation Accuracy per method, overall and per mechanism."""
    methods = list(data["methods"])
    groups = ["overall"] + list(data["mechanisms"])
    values = [
        [data["cea"][m]["overall"]] + [data["cea"][m]["by_mechanism"].get(g, np.nan) for g in data["mechanisms"]]
        for m in methods
    ]
    errors = None
    if "ci95" in data:
        errors = [
            [data["ci95"][m]["overall"]] + [data["ci95"][m]["by_mechanism"].get(g, 0.0) for g in data["mechanisms"]]
            for m in methods
        ]
    chance = data.get("chance_level")
    labels = [method_label(m) for m in methods]
    with ieee_style():
        fig, ax = new_figure(FigureSpec("double", 0.40))
        _grouped_bar(ax, groups, labels, values, errors, ylabel=r"CEA $=P(\hat H = H^*)$")
        if chance:
            ax.axhline(chance, color="k", linestyle=":", linewidth=0.9, label=f"chance ({chance:.2f})")
        ax.set_xticklabels(["overall"] + [mechanism_label(g) for g in data["mechanisms"]], fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.set_title("Causal explanation accuracy (abstention counted as incorrect)", fontsize=7)
        # The legend goes below the axes: with nine methods it would otherwise
        # cover the bars it is supposed to explain.
        ax.legend(fontsize=5.2, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  frameon=False, columnspacing=1.0, handlelength=1.4)
        return save_figure(fig, stem, **kw)


def plot_identifiability_roc(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """ROC for predicting resolvability under the allowed action set."""
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.90))
        for i, curve in enumerate(data["curves"]):
            style = series_style(i)
            ax.plot(curve["fpr"], curve["tpr"], label=f"{curve['label']} (AUROC {curve['auroc']:.3f})",
                    linewidth=1.1, color=style["color"], linestyle=style["linestyle"])
        ax.plot([0, 1], [0, 1], color="0.6", linestyle=":", linewidth=0.8, label="chance")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title("Identifiability prediction", fontsize=7)
        ax.legend(fontsize=5.5, loc="lower right")
        return save_figure(fig, stem, **kw)


def plot_fpcr(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """False Physical Certainty Rate on non-identifiable cases."""
    methods = [method_label(m) for m in data["methods"]]
    values = np.asarray(data["fpcr"], dtype=np.float64)
    errs = np.asarray(data.get("ci95", np.zeros(len(methods))), dtype=np.float64)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.70))
        x = np.arange(len(methods))
        ax.bar(x, values, yerr=np.nan_to_num(errs), capsize=2, color="#D55E00",
               edgecolor="k", linewidth=0.4, error_kw={"elinewidth": 0.6})
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=5.5)
        ax.set_ylabel(r"FPCR $=P(\max_k p(H_k)>\tau \mid y_{id}=0)$")
        ax.set_ylim(0, 1.05)
        ax.set_title(rf"False physical certainty ($\tau={data.get('tau', 0.8):g}$; lower is better)", fontsize=7)
        return save_figure(fig, stem, **kw)


def plot_contact_depth_error(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """AbsRel and RMSE of the physical contact depth."""
    methods = [method_label(m) for m in data["methods"]]
    with ieee_style():
        fig, axes = new_figure(FigureSpec("double", 0.36), ncols=2)
        for ax, key, label in zip(
            axes,
            ("abs_rel", "rmse"),
            (r"AbsRel$_{\mathrm{contact}}$", r"RMSE$_{\mathrm{contact}}$ (m)"),
            strict=True,
        ):
            values = np.asarray(data[key], dtype=np.float64)
            errs = np.asarray(data.get(f"{key}_ci95", np.zeros(len(methods))), dtype=np.float64)
            x = np.arange(len(methods))
            ax.bar(x, values, yerr=np.nan_to_num(errs), capsize=2, color="#0072B2",
                   edgecolor="k", linewidth=0.4, error_kw={"elinewidth": 0.6})
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=5.5)
            ax.set_ylabel(label)
        axes[0].set_title("Physical contact-depth error (lower is better)", fontsize=7, loc="left")
        return save_figure(fig, stem, **kw)


def plot_mcrb_validation(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Theory validation: does the measured resolving baseline follow the MCRB law?

    Plots ``1/B_measured`` against ``f |1/Z_1 - 1/Z_2|``.  The MCRB derivation
    predicts a straight line through the origin; the fitted slope and ``R^2`` are
    reported in the legend.
    """
    x = np.asarray(data["f_inv_depth_difference"], dtype=np.float64)
    y = np.asarray(data["inverse_measured_baseline"], dtype=np.float64)
    slope = float(data["slope"])
    intercept = float(data.get("intercept", 0.0))
    r2 = float(data["r_squared"])
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.78))
        ax.scatter(x, y, s=9, color="#0072B2", marker="o", linewidths=0, alpha=0.8, label="scenes")
        xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 50)
        ax.plot(xs, slope * xs + intercept, color="#D55E00", linewidth=1.1,
                label=rf"fit: slope $={slope:.3f}$, $R^2={r2:.3f}$")
        ax.set_xlabel(r"$f\,|1/Z_1 - 1/Z_2|$  (px m$^{-1}$)")
        ax.set_ylabel(r"$1 / B_{\min}^{\mathrm{measured}}$  (m$^{-1}$)")
        ax.set_title("MCRB scaling law validated against simulation", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")
        return save_figure(fig, stem, **kw)


def plot_mcrb_error(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Predicted versus ground-truth MCRB."""
    pred = np.asarray(data["predicted"], dtype=np.float64)
    gt = np.asarray(data["ground_truth"], dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(gt)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.88))
        ax.scatter(gt[ok], pred[ok], s=9, color="#0072B2", marker="o", linewidths=0, alpha=0.8)
        if np.any(ok):
            lim = [0.0, float(max(np.nanmax(gt[ok]), np.nanmax(pred[ok]))) * 1.05]
            ax.plot(lim, lim, color="k", linestyle=":", linewidth=0.9, label="ideal")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$B_{\min}^{GT}$ (m)")
        ax.set_ylabel(r"$\hat B_{\min}$ (m)")
        mae = data.get("mae")
        ax.set_title("MCRB prediction" + (f"  (MAE $={mae:.4f}$ m)" if mae is not None else ""), fontsize=7)
        ax.legend(fontsize=6)
        return save_figure(fig, stem, **kw)


def plot_metric_summary_table(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """A compact figure-form summary table of the headline metrics."""
    methods = [method_label(m) for m in data["methods"]]
    metrics = list(data["metric_names"])
    values = np.asarray(data["values"], dtype=np.float64)  # (methods, metrics)
    with ieee_style():
        # Height follows the ROW COUNT. The previous 0.30 + 0.06*n gave a 6-inch
        # canvas for a 2-inch table, and `loc="center"` then stranded it in the
        # middle: 95 % of the image was blank.
        n_rows = len(methods) + 1
        fig, ax = new_figure(FigureSpec("double", 0.055 * n_rows + 0.06))
        ax.axis("off")
        cell = [[("n/a" if not np.isfinite(v) else f"{v:.3f}") for v in row] for row in values]
        table = ax.table(
            cellText=cell, rowLabels=methods, colLabels=metrics, cellLoc="center",
            # Fill the axes exactly rather than floating inside it.
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        ax.set_title(data.get("title", "Headline metrics"), fontsize=7)
        return save_figure(fig, stem, **kw)
