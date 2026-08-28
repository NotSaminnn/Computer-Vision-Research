"""Centralised IEEE-style scientific plotting.

Rules enforced by this package:

* no ``matplotlib`` configuration anywhere outside :mod:`ieee_style`;
* every figure is produced from a *saved result file*, so
  ``scripts/generate_all_figures.py`` can rebuild the whole figure set without
  re-running an experiment;
* every figure is written as vector PDF plus a high-resolution PNG;
* no number is ever transcribed into a plot by hand.
"""

from intervene3d.visualization.ablation_plots import plot_ablation_grid, plot_generalisation
from intervene3d.visualization.ambiguity_plots import (
    plot_hypothesis_probabilities,
    plot_initial_view_similarity,
    plot_pairwise_bar,
    plot_resolvability_distribution,
    plot_separability_matrix,
    plot_uncertainty_decomposition,
)
from intervene3d.visualization.export import DEFAULT_FORMATS, figure_index, save_figure
from intervene3d.visualization.geometry_plots import (
    plot_contact_vs_apparent,
    plot_depth_error_map,
    plot_landmark_view,
    plot_rendered_views,
)
from intervene3d.visualization.ieee_style import (
    METHOD_SHORT_NAMES,
    FigureSpec,
    ieee_style,
    mechanism_label,
    mechanism_style,
    method_label,
    new_figure,
    series_style,
    style_provenance,
)
from intervene3d.visualization.intervention_plots import (
    plot_action_utility,
    plot_camera_trajectory,
    plot_intervention_regret,
    plot_motion_cost,
    plot_predicted_vs_observed,
    plot_separability_vs_baseline,
)
from intervene3d.visualization.metric_plots import (
    plot_contact_depth_error,
    plot_explanation_accuracy,
    plot_fpcr,
    plot_identifiability_roc,
    plot_mcrb_error,
    plot_mcrb_validation,
    plot_metric_summary_table,
)
from intervene3d.visualization.pipeline_figure import (
    plot_matched_variant_strip,
    plot_pipeline_overview,
)

__all__ = [
    "plot_ablation_grid", "plot_generalisation",
    "plot_hypothesis_probabilities", "plot_initial_view_similarity", "plot_pairwise_bar",
    "plot_resolvability_distribution", "plot_separability_matrix", "plot_uncertainty_decomposition",
    "DEFAULT_FORMATS", "figure_index", "save_figure",
    "plot_contact_vs_apparent", "plot_depth_error_map", "plot_landmark_view", "plot_rendered_views",
    "METHOD_SHORT_NAMES", "FigureSpec", "ieee_style", "mechanism_label", "mechanism_style",
    "method_label", "new_figure", "series_style", "style_provenance",
    "plot_action_utility", "plot_camera_trajectory", "plot_intervention_regret", "plot_motion_cost",
    "plot_predicted_vs_observed", "plot_separability_vs_baseline",
    "plot_contact_depth_error", "plot_explanation_accuracy", "plot_fpcr", "plot_identifiability_roc",
    "plot_mcrb_error", "plot_mcrb_validation", "plot_metric_summary_table",
    "plot_matched_variant_strip", "plot_pipeline_overview",
]
