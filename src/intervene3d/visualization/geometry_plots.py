"""Geometry figures: contact versus apparent depth, error maps, landmark views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.visualization.export import save_figure
from intervene3d.visualization.ieee_style import (
    FigureSpec,
    ieee_style,
    mechanism_label,
    new_figure,
)


def plot_contact_vs_apparent(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """The distinction the whole project rests on, for one scene per mechanism.

    A monocular reading reports the *apparent* content depth for every mechanism
    -- the variants are pixel-identical -- while the *contact* depth (the first
    surface an agent would actually touch) is the screen, the mirror or the pane.
    """
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.80))
        for i, entry in enumerate(data["scenes"]):
            apparent = np.asarray(entry["apparent_depth"], dtype=np.float64)
            contact = np.asarray(entry["contact_depth"], dtype=np.float64)
            ok = np.isfinite(apparent) & np.isfinite(contact)
            from intervene3d.visualization.ieee_style import mechanism_style

            style = mechanism_style(entry["mechanism"], i)
            ax.scatter(apparent[ok], contact[ok], s=8, marker=style["marker"], color=style["color"],
                       linewidths=0, alpha=0.8, label=mechanism_label(entry["mechanism"]))
        lim_hi = float(data.get("max_depth", 6.0))
        ax.plot([0, lim_hi], [0, lim_hi], color="k", linestyle=":", linewidth=0.9,
                label="apparent = contact")
        ax.set_xlabel(r"apparent depth $D_{\mathrm{apparent}}$ (m)")
        ax.set_ylabel(r"contact depth $D_{\mathrm{contact}}$ (m)")
        ax.set_title("Same appearance, different physical surface", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left")
        return save_figure(fig, stem, **kw)


def plot_landmark_view(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Reference and post-intervention landmark views side by side."""
    views = data["views"]
    width = int(data.get("image_width", 320))
    height = int(data.get("image_height", 240))
    with ieee_style():
        fig, axes = new_figure(FigureSpec("double", 0.34), ncols=len(views))
        axes = np.atleast_1d(axes)
        for ax, view in zip(axes, views, strict=False):
            uv = np.asarray(view["uv"], dtype=np.float64)
            vis = np.asarray(view["visible"], dtype=bool)
            channel = np.asarray(view["channel"], dtype=int)
            for ch, colour, marker, size, label in (
                (0, "#0072B2", "o", 5, "content"),
                (1, "#7f7f7f", "s", 8, "interface frame"),
                (2, "#D55E00", "*", 26, "virtual observer marker"),
            ):
                m = (channel == ch) & vis
                if np.any(m):
                    ax.scatter(uv[m, 0], uv[m, 1], s=size, c=colour, marker=marker,
                               linewidths=0, label=label)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect("equal")
            ax.set_title(view["title"], fontsize=6.5)
            ax.set_xlabel("$u$ (px)", fontsize=6)
        axes[0].set_ylabel("$v$ (px)", fontsize=6)
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, fontsize=5.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
        return save_figure(fig, stem, **kw)


def plot_rendered_views(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Rendered RGB of the matched variants -- the visual statement of the premise."""
    panels = data["panels"]
    with ieee_style():
        fig, axes = new_figure(FigureSpec("double", 0.30), ncols=len(panels))
        axes = np.atleast_1d(axes)
        for ax, panel in zip(axes, panels, strict=False):
            ax.imshow(np.asarray(panel["image"], dtype=np.float64))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(panel["title"], fontsize=6.5)
        return save_figure(fig, stem, **kw)


def plot_depth_error_map(data: dict[str, Any], stem: Path | str, **kw: Any) -> list[str]:
    """Per-landmark contact-depth error, positioned in the image."""
    uv = np.asarray(data["uv"], dtype=np.float64)
    error = np.asarray(data["error"], dtype=np.float64)
    ok = np.all(np.isfinite(uv), axis=1) & np.isfinite(error)
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single", 0.82))
        sc = ax.scatter(uv[ok, 0], uv[ok, 1], c=error[ok], s=16, cmap="magma", linewidths=0)
        ax.set_xlim(0, int(data.get("image_width", 320)))
        ax.set_ylim(int(data.get("image_height", 240)), 0)
        ax.set_aspect("equal")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("contact-depth error (m)", fontsize=6)
        ax.set_xlabel("$u$ (px)")
        ax.set_ylabel("$v$ (px)")
        ax.set_title(data.get("title", "Contact-depth error map"), fontsize=7)
        return save_figure(fig, stem, **kw)
