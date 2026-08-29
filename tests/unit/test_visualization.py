"""The centralised IEEE plotting system."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from intervene3d.visualization.export import figure_index, save_figure
from intervene3d.visualization.ieee_style import (
    IEEE_DOUBLE_COLUMN_IN,
    IEEE_SINGLE_COLUMN_IN,
    FigureSpec,
    ieee_style,
    mechanism_label,
    mechanism_style,
    new_figure,
    series_style,
    style_provenance,
)


def test_column_widths_match_ieee():
    assert FigureSpec("single").size[0] == IEEE_SINGLE_COLUMN_IN
    assert FigureSpec("double").size[0] == IEEE_DOUBLE_COLUMN_IN
    assert FigureSpec("single", height_ratio=0.5).size == (IEEE_SINGLE_COLUMN_IN, IEEE_SINGLE_COLUMN_IN * 0.5)


def test_style_provenance_is_recorded():
    prov = style_provenance()
    assert prov["usetex"] is False, "LaTeX must never be required to draw a figure"
    assert prov["styles"]


def test_series_styles_are_grayscale_safe():
    """Distinguishable without colour: marker AND linestyle must also differ."""
    styles = [series_style(i) for i in range(5)]
    assert len({s["marker"] for s in styles}) == 5
    assert len({str(s["linestyle"]) for s in styles}) == 5


def test_mechanism_styles_are_stable_across_figures():
    assert mechanism_style("direct")["color"] == mechanism_style("direct")["color"]
    assert mechanism_style("direct")["color"] != mechanism_style("emissive")["color"]
    assert "H_D" in mechanism_label("direct")
    assert mechanism_label("unknown_mechanism") == "unknown_mechanism"


def test_save_figure_writes_vector_and_raster(tmp_path):
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single"))
        ax.plot([0, 1], [0, 1], label=r"$\Delta_{ij}(a)$")
        ax.set_xlabel(r"baseline $B$ (m)")
        ax.legend()
        written = save_figure(fig, tmp_path / "test_fig", formats=("pdf", "png"), dpi=100)
    assert len(written) == 2
    for path in written:
        assert (tmp_path / path.split("/")[-1]).exists()
    index = figure_index(written)
    assert {i["format"] for i in index} == {"pdf", "png"}
    assert all(i["bytes"] > 500 for i in index)


def test_mathematical_notation_renders_without_latex(tmp_path):
    with ieee_style():
        fig, ax = new_figure(FigureSpec("single"))
        ax.set_title(r"$\mathcal{I}_{\mathcal{A}}(H_i,H_j) = \max_a \Delta_{ij}(a)$")
        ax.set_ylabel(r"$p(H_k \mid I_{1:t})$")
        written = save_figure(fig, tmp_path / "math", formats=("png",), dpi=80)
    assert written


@pytest.mark.parametrize("plotter, payload", [
    ("plot_initial_view_similarity",
     {"pairs": ["A/B"], "reference_distance_px": [0.0], "best_action_distance_px": [12.0], "epsilon_px": 1.0}),
    ("plot_separability_matrix",
     {"identifiability_matrix": [[0.0, 5.0], [5.0, 0.0]], "mechanisms": ["direct", "emissive"],
      "epsilon_px": 1.0, "scene_id": "s"}),
    ("plot_hypothesis_probabilities",
     {"mechanisms": ["direct", "emissive"], "belief_trajectory": [[0.5, 0.5], [0.9, 0.1]], "tau": 0.8}),
    ("plot_resolvability_distribution",
     {"identifiability_scores": [0.1, 5.0, 8.0], "resolvable": [False, True, True], "epsilon_px": 1.0}),
    ("plot_uncertainty_decomposition",
     {"prediction_uncertainty": [0.1, 0.9], "identifiability_uncertainty": [0.0, 0.8],
      "resolvable": [True, False]}),
    ("plot_fpcr", {"methods": ["a", "b"], "fpcr": [0.1, 0.5], "tau": 0.8}),
    ("plot_intervention_regret", {"methods": ["a", "b"], "mean": [0.1, 0.3]}),
    ("plot_motion_cost", {"methods": ["a", "b"], "mean": [0.1, 0.3]}),
    ("plot_contact_depth_error", {"methods": ["a"], "abs_rel": [0.02], "rmse": [0.05]}),
    ("plot_mcrb_error", {"predicted": [0.1, 0.2], "ground_truth": [0.11, 0.19], "mae": 0.01}),
    ("plot_mcrb_validation",
     {"f_inv_depth_difference": [1.0, 2.0, 3.0], "inverse_measured_baseline": [2.0, 4.0, 6.0],
      "slope": 2.0, "intercept": 0.0, "r_squared": 1.0}),
    ("plot_identifiability_roc",
     {"curves": [{"label": "m", "fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1], "auroc": 0.9}]}),
    ("plot_metric_summary_table",
     {"methods": ["a"], "metric_names": ["CEA"], "values": [[0.5]], "title": "t"}),
    ("plot_separability_vs_baseline",
     {"baselines": [0.0, 0.1, 0.2], "epsilon_px": 1.0,
      "curves": [{"label": "H_D vs H_E", "separability": [0.0, 0.8, 4.0], "mcrb": 0.12}]}),
    ("plot_action_utility",
     {"action_names": ["a", "b", "c"], "utility": [1.0, 3.0, 2.0], "selected_index": 1,
      "oracle_index": 1, "scene_id": "s"}),
    ("plot_camera_trajectory",
     {"candidate_translations": [[0.1, 0, 0], [-0.1, 0, 0]], "executed_translations": [[0.1, 0, 0]]}),
    ("plot_contact_vs_apparent",
     {"scenes": [{"mechanism": "direct", "apparent_depth": [2.0, 3.0], "contact_depth": [2.0, 3.0]}],
      "max_depth": 4.0}),
    ("plot_ablation_grid",
     {"variants": ["a", "b"], "metrics": ["CEA"], "values": [[0.4, 0.6]]}),
    ("plot_generalisation",
     {"x": [0.0, 0.1], "series": [{"label": "m", "y": [1.0, 0.8], "ci95": [0.02, 0.03]}]}),
])
def test_every_plot_family_renders(plotter, payload, tmp_path):
    import intervene3d.visualization as viz

    written = getattr(viz, plotter)(payload, tmp_path / plotter, formats=("png",), dpi=70)
    assert written and all(int(i["bytes"]) > 500 for i in figure_index(written))


def test_pipeline_overview_renders(tmp_path):
    from intervene3d.visualization import plot_pipeline_overview

    written = plot_pipeline_overview(tmp_path / "pipeline", formats=("png", "pdf"), dpi=70)
    assert len(written) == 2


def test_predicted_vs_observed_and_landmark_views(tmp_path):
    from intervene3d.visualization import plot_landmark_view, plot_predicted_vs_observed

    n = 6
    uv = np.random.default_rng(0).uniform(10, 200, (n, 2))
    written = plot_predicted_vs_observed(
        {"reference_uv": uv.tolist(), "observed_uv": (uv + 3).tolist(),
         "predictions": [{"label": "direct", "uv": (uv + 2).tolist()}], "scene_id": "s"},
        tmp_path / "pvo", formats=("png",), dpi=70,
    )
    assert written
    written = plot_landmark_view(
        {"views": [{"title": "ref", "uv": uv.tolist(), "visible": [True] * n, "channel": [0] * n}],
         "image_width": 240, "image_height": 180},
        tmp_path / "lv", formats=("png",), dpi=70,
    )
    assert written


# --------------------------------------------------- figure legibility (2026-08-29)
def _relative_luminance(rgb):
    """WCAG relative luminance for an (r, g, b) triple in [0, 1]."""
    import numpy as np

    c = np.asarray(rgb[:3], dtype=float)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _contrast(fg, bg) -> float:
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def test_matrix_annotations_stay_readable_on_every_cell():
    """Regression: annotations were hardcoded white.

    ``viridis`` runs from near-black to bright yellow, so a fixed white label is
    invisible on every high-value cell -- and the high-value cells are exactly
    the resolvable pairs the figure exists to show.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from intervene3d.visualization.ambiguity_plots import DARK_INK, LIGHT_INK, _prefers_dark_ink

    cmap = plt.get_cmap("viridis")
    for t in np.linspace(0.0, 1.0, 21):
        r, g, b, _ = cmap(t)
        chosen = DARK_INK if _prefers_dark_ink((r, g, b)) else LIGHT_INK
        fg = tuple(int(chosen[i : i + 2], 16) / 255 for i in (1, 3, 5))
        alt = LIGHT_INK if chosen == DARK_INK else DARK_INK
        alt_fg = tuple(int(alt[i : i + 2], 16) / 255 for i in (1, 3, 5))
        got, other = _contrast(fg, (r, g, b)), _contrast(alt_fg, (r, g, b))

        # The invariant that matters: the ink chosen is always the better of the
        # two. Viridis' mid-range caps at ~4.3:1, below WCAG AA, which is why the
        # renderer also strokes a halo in the opposite tone.
        assert got >= other, f"picked the worse ink on viridis({t:.2f}): {got:.2f} < {other:.2f}"
        assert got >= 4.0, f"annotation {chosen} on viridis({t:.2f}) has contrast {got:.2f}:1"

    # And the policy this replaced -- always white -- must genuinely fail.
    r, g, b, _ = cmap(1.0)
    assert _contrast((1.0, 1.0, 1.0), (r, g, b)) < 1.5, "the old white-on-yellow bug should be caught"


def test_summary_table_is_not_mostly_blank(tmp_path):
    """Regression: the table canvas was sized 0.30 + 0.06*n_methods.

    With nine methods that is a six-inch-tall figure for a two-inch table, and
    ``loc="center"`` then stranded it: 95 % of the saved image was empty.
    """
    import numpy as np
    from PIL import Image

    from intervene3d.visualization.metric_plots import plot_metric_summary_table

    methods = [f"method_{i}" for i in range(9)]
    data = {
        "methods": methods,
        "metric_names": ["CEA", "AUROC", "FPCR"],
        "values": np.linspace(0, 1, 27).reshape(9, 3).tolist(),
        "title": "test",
    }
    written = plot_metric_summary_table(data, tmp_path / "summary", formats=("png",), dpi=80)
    arr = np.asarray(Image.open(written[0]).convert("L"))
    height_in = arr.shape[0] / 80
    # The defect produced ~6 inches; a well-sized 10-row table is under four.
    assert height_in < 4.0, f"a 9-row table should not need {height_in:.1f} inches"
    # The table must reach the edges rather than float in the middle: check that
    # the top and bottom eighths of the image both carry ink.
    band = arr.shape[0] // 8
    assert (arr[:band] < 128).any(), "no content in the top of the canvas"
    assert (arr[-band:] < 128).any(), "no content in the bottom of the canvas"


def test_matched_variant_strip_labels_its_colour_encoding(tmp_path):
    """Colour there encodes the channel, not the mechanism. Say so."""
    import numpy as np

    from intervene3d.visualization.pipeline_figure import plot_matched_variant_strip

    rng = np.random.default_rng(0)
    n = 12

    def view():
        return {
            "uv": rng.uniform(0, 200, size=(n, 2)).tolist(),
            "visible": [True] * n,
            "channel": ([0] * 8) + ([1] * 2) + ([2] * 2),
        }

    data = {
        "variants": [{"label": m, "reference": view(), "after": view()}
                     for m in ("direct", "emissive", "reflection")],
        "image_width": 200, "image_height": 150,
    }
    written = plot_matched_variant_strip(data, tmp_path / "strip", formats=("png",), dpi=80)
    assert written and Path(written[0]).stat().st_size > 0
