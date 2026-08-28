"""Assemble figure inputs from results, and render every figure.

Two strictly separated steps:

1. :func:`build_figure_data` turns an experiment's outcome into a plain,
   JSON-serialisable dictionary, written to ``metrics/figure_data.json``.
2. :func:`generate_figures` reads *only* that dictionary (plus ``metrics.json``)
   and draws.

That separation is what makes ``scripts/generate_all_figures.py`` able to
rebuild the entire figure set from a finished run without recomputing anything,
and it guarantees no number is ever hand-transcribed into a plot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from intervene3d.metrics.identifiability import auroc, roc_curve
from intervene3d.visualization import (
    plot_action_utility,
    plot_camera_trajectory,
    plot_contact_depth_error,
    plot_contact_vs_apparent,
    plot_explanation_accuracy,
    plot_fpcr,
    plot_hypothesis_probabilities,
    plot_identifiability_roc,
    plot_initial_view_similarity,
    plot_intervention_regret,
    plot_landmark_view,
    plot_matched_variant_strip,
    plot_mcrb_error,
    plot_mcrb_validation,
    plot_metric_summary_table,
    plot_motion_cost,
    plot_pipeline_overview,
    plot_predicted_vs_observed,
    plot_resolvability_distribution,
    plot_separability_matrix,
    plot_separability_vs_baseline,
    plot_uncertainty_decomposition,
)
from intervene3d.visualization.ieee_style import method_label


def _mean(values: Sequence[Any]) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def build_figure_data(
    *,
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Mapping[str, Any],
    mechanisms: Sequence[str],
    epsilon: float,
    tau: float,
    examples: Mapping[str, Any],
    initial_view: Mapping[str, Any],
    baseline_curves: Mapping[str, Any],
    theory: Mapping[str, Any],
    matched_strip: Mapping[str, Any],
    contact_vs_apparent: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect everything the figure generator needs into one serialisable dict."""
    methods = list(rows_by_method)
    per_method = metrics["methods"]

    cea = {
        m: {
            "overall": per_method[m]["cea"]["cea_all"],
            "by_mechanism": {k: v["cea_all"] for k, v in per_method[m]["cea"]["by_mechanism"].items()},
        }
        for m in methods
    }
    roc = []
    for m in methods:
        rows = rows_by_method[m]
        scores = [float(r["identifiability_score"]) for r in rows]
        labels = [bool(r["resolvable_gt"]) for r in rows]
        curve = roc_curve(scores, labels)
        roc.append({"label": method_label(m), "auroc": auroc(scores, labels), **curve})

    all_rows = [r for rows in rows_by_method.values() for r in rows]
    primary = methods[-1]
    primary_rows = rows_by_method[primary]

    return {
        "epsilon_px": float(epsilon),
        "tau": float(tau),
        "mechanisms": list(mechanisms),
        "methods": methods,
        "chance_level": 1.0 / max(len(mechanisms), 1),
        "explanation_accuracy": {
            "methods": methods, "mechanisms": list(mechanisms), "cea": cea,
            "chance_level": 1.0 / max(len(mechanisms), 1),
        },
        "identifiability_roc": {"curves": roc},
        "fpcr": {"methods": methods, "fpcr": [per_method[m]["fpcr"]["fpcr"] for m in methods], "tau": float(tau)},
        "contact_depth": {
            "methods": methods,
            "abs_rel": [per_method[m]["contact_depth"]["abs_rel_contact"] for m in methods],
            "rmse": [per_method[m]["contact_depth"]["rmse_contact"] for m in methods],
        },
        "regret": {
            "methods": methods,
            "mean": [per_method[m]["intervention"]["normalised_regret"] for m in methods],
        },
        "motion": {
            "methods": methods,
            "mean": [per_method[m]["intervention"]["motion_cost"] for m in methods],
        },
        "resolvability": {
            "identifiability_scores": [float(r["identifiability_score_gt"]) for r in primary_rows
                                       if r.get("identifiability_score_gt") is not None],
            "resolvable": [bool(r["resolvable_gt"]) for r in primary_rows
                           if r.get("identifiability_score_gt") is not None],
            "epsilon_px": float(epsilon),
        },
        "uncertainty": {
            "prediction_uncertainty": [float(r["prediction_uncertainty"]) for r in primary_rows],
            "identifiability_uncertainty": [float(r["identifiability_uncertainty"]) for r in primary_rows],
            "resolvable": [bool(r["resolvable_gt"]) for r in primary_rows],
        },
        "mcrb_error": {
            "predicted": [r.get("predicted_mcrb") if r.get("predicted_mcrb") is not None else np.nan
                          for r in primary_rows],
            "ground_truth": [r.get("mcrb_gt") if r.get("mcrb_gt") is not None else np.nan
                             for r in primary_rows],
            "mae": per_method[primary]["mcrb"]["mae"],
        },
        "summary_table": {
            "methods": methods,
            "metric_names": ["CEA", "AUROC", "FPCR", "AbsRel", "regret", "motion"],
            "values": [
                [
                    per_method[m]["cea"]["cea_all"],
                    per_method[m]["identifiability"]["identifiability_auroc"],
                    per_method[m]["fpcr"]["fpcr"],
                    per_method[m]["contact_depth"]["abs_rel_contact"],
                    per_method[m]["intervention"]["normalised_regret"],
                    per_method[m]["intervention"]["motion_cost"],
                ]
                for m in methods
            ],
            "title": "Phase 1 headline metrics",
        },
        "initial_view": dict(initial_view),
        "baseline_curves": dict(baseline_curves),
        "theory": dict(theory),
        "examples": dict(examples),
        "matched_strip": dict(matched_strip),
        "contact_vs_apparent": dict(contact_vs_apparent),
        "n_eval_rows": len(all_rows),
    }


def generate_figures(
    figure_data: Mapping[str, Any],
    out_dir: Path | str,
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = 400,
) -> list[str]:
    """Render every Phase 1 figure from ``figure_data``.  Returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kw = {"formats": tuple(formats), "dpi": int(dpi)}
    written: list[str] = []

    written += plot_pipeline_overview(
        out / "fig01_pipeline_overview",
        annotation={"caption": "Intervene3D: identifiability of the physical explanation under a bounded action set"},
        **kw,
    )
    written += plot_initial_view_similarity(figure_data["initial_view"], out / "fig02_initial_view_similarity", **kw)
    if figure_data.get("matched_strip", {}).get("variants"):
        written += plot_matched_variant_strip(figure_data["matched_strip"], out / "fig03_matched_variants", **kw)
    written += plot_separability_vs_baseline(figure_data["baseline_curves"], out / "fig04_separability_vs_baseline", **kw)

    example = figure_data.get("examples", {}).get("primary")
    if example:
        written += plot_separability_matrix(
            {**example, "epsilon_px": figure_data["epsilon_px"]}, out / "fig05_separability_matrix", **kw
        )
        written += plot_action_utility(example, out / "fig06_action_utility", **kw)
        written += plot_camera_trajectory(example, out / "fig07_camera_trajectory", **kw)
        written += plot_hypothesis_probabilities(
            {**example, "tau": figure_data["tau"]}, out / "fig08_hypothesis_probabilities", **kw
        )
        written += plot_predicted_vs_observed(example, out / "fig09_predicted_vs_observed", **kw)
        written += plot_landmark_view(
            {
                "views": [
                    {
                        "title": "reference $C_0$",
                        "uv": example["reference_uv"],
                        "visible": example["reference_visible"],
                        "channel": example["reference_channel"],
                    },
                    {
                        "title": f"after $a^*$ = {example['action_names'][example['selected_index']]}",
                        "uv": example["observed_uv"],
                        "visible": example["observed_visible"],
                        "channel": example["reference_channel"],
                    },
                ],
                **figure_data.get("image_size", {}),
            },
            out / "fig10_landmark_views",
            **kw,
        )

    written += plot_explanation_accuracy(figure_data["explanation_accuracy"], out / "fig11_explanation_accuracy", **kw)
    written += plot_identifiability_roc(figure_data["identifiability_roc"], out / "fig12_identifiability_roc", **kw)
    written += plot_fpcr(figure_data["fpcr"], out / "fig13_fpcr", **kw)
    written += plot_resolvability_distribution(figure_data["resolvability"], out / "fig14_resolvability_distribution", **kw)
    written += plot_uncertainty_decomposition(figure_data["uncertainty"], out / "fig15_uncertainty_decomposition", **kw)
    written += plot_contact_depth_error(figure_data["contact_depth"], out / "fig16_contact_depth_error", **kw)
    written += plot_contact_vs_apparent(figure_data["contact_vs_apparent"], out / "fig17_contact_vs_apparent", **kw)
    if np.isfinite(figure_data["theory"].get("r_squared", np.nan)):
        written += plot_mcrb_validation(figure_data["theory"], out / "fig18_mcrb_theory_validation", **kw)
    written += plot_mcrb_error(figure_data["mcrb_error"], out / "fig19_mcrb_error", **kw)
    written += plot_intervention_regret(figure_data["regret"], out / "fig20_intervention_regret", **kw)
    written += plot_motion_cost(figure_data["motion"], out / "fig21_motion_cost", **kw)
    written += plot_metric_summary_table(figure_data["summary_table"], out / "fig22_metric_summary", **kw)
    return written
