"""The centralised IEEE plotting system."""

from __future__ import annotations

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
