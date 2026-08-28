"""Metrics: perfect prediction, wrong prediction, abstention and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from intervene3d.hypotheses.families import direct_hypothesis, display_hypothesis, mirror_hypothesis
from intervene3d.metrics.aggregate import aggregate_runs, flatten_metrics, format_pm, summarise
from intervene3d.metrics.classification import (
    causal_explanation_accuracy,
    confusion_matrix,
    false_physical_certainty_rate,
)
from intervene3d.metrics.depth import abs_rel, contact_depth_metrics, delta_threshold, rmse
from intervene3d.metrics.identifiability import auroc, binary_decision_metrics, roc_curve
from intervene3d.metrics.mcrb import (
    analytic_pair_is_applicable,
    differential_parallax,
    mcrb_absolute_error,
    mcrb_analytic,
    mcrb_numeric,
)
from intervene3d.metrics.regret import intervention_regret, motion_cost


# ------------------------------------------------------------------- CEA
def test_cea_perfect_and_wrong():
    truth = ["direct", "emissive", "reflection"]
    assert causal_explanation_accuracy(truth, truth)["cea_all"] == 1.0
    wrong = ["emissive", "reflection", "direct"]
    assert causal_explanation_accuracy(wrong, truth)["cea_all"] == 0.0


def test_cea_counts_abstention_as_incorrect_but_reports_committed_separately():
    truth = ["direct", "emissive", "reflection", "direct"]
    pred = ["direct", "emissive", "abstain", "direct"]
    out = causal_explanation_accuracy(pred, truth, abstained=[False, False, True, False])
    assert np.isclose(out["cea_all"], 0.75)
    assert np.isclose(out["cea_committed"], 1.0)
    assert np.isclose(out["abstention_rate"], 0.25)


def test_cea_reports_per_mechanism():
    truth = ["direct", "direct", "emissive", "emissive"]
    pred = ["direct", "emissive", "emissive", "emissive"]
    out = causal_explanation_accuracy(pred, truth)
    assert np.isclose(out["by_mechanism"]["direct"]["cea_all"], 0.5)
    assert np.isclose(out["by_mechanism"]["emissive"]["cea_all"], 1.0)


def test_cea_edge_cases():
    empty = causal_explanation_accuracy([], [])
    assert empty["n"] == 0 and np.isnan(empty["cea_all"])
    with pytest.raises(ValueError):
        causal_explanation_accuracy(["a"], ["a", "b"])


def test_confusion_matrix():
    m = confusion_matrix(["a", "b", "a"], ["a", "a", "b"], ["a", "b"])
    assert m.tolist() == [[1, 1], [1, 0]]


# ------------------------------------------------------------------ FPCR
def test_fpcr_confident_on_non_identifiable_is_penalised():
    out = false_physical_certainty_rate([0.95, 0.95], [False, False], tau=0.8)
    assert out["fpcr"] == 1.0 and out["n_non_identifiable"] == 2


def test_fpcr_abstention_is_not_false_certainty():
    """Abstaining on an unresolvable case is precisely the desired behaviour."""
    out = false_physical_certainty_rate([0.95, 0.95], [False, False], tau=0.8, abstained=[True, True])
    assert out["fpcr"] == 0.0


def test_fpcr_ignores_resolvable_cases_and_reports_them_separately():
    out = false_physical_certainty_rate([0.99, 0.4], [True, False], tau=0.8)
    assert out["fpcr"] == 0.0
    assert out["confident_rate_on_identifiable"] == 1.0


def test_fpcr_is_nan_when_there_are_no_non_identifiable_cases():
    out = false_physical_certainty_rate([0.9], [True], tau=0.8)
    assert np.isnan(out["fpcr"]) and out["n_non_identifiable"] == 0


# ------------------------------------------------------------------ depth
def test_depth_metrics_perfect_and_wrong():
    gt = np.array([1.0, 2.0, 4.0])
    assert abs_rel(gt, gt) == 0.0 and rmse(gt, gt) == 0.0
    assert delta_threshold(gt, gt) == 1.0
    assert np.isclose(abs_rel(gt * 1.1, gt), 0.1)
    assert np.isclose(rmse(np.array([2.0, 2.0, 4.0]), gt), np.sqrt(1 / 3))


def test_depth_metrics_ignore_invalid_entries():
    pred = np.array([1.0, np.nan, 4.0])
    gt = np.array([1.0, 2.0, 0.0])  # zero GT is invalid
    out = contact_depth_metrics(pred, gt)
    assert out["n_valid"] == 1 and out["abs_rel_contact"] == 0.0


def test_depth_metrics_all_invalid_returns_nan():
    assert np.isnan(abs_rel(np.array([np.nan]), np.array([1.0])))
    assert np.isnan(rmse(np.array([1.0]), np.array([0.0])))


# ---------------------------------------------------------------- AUROC / ROC
def test_auroc_perfect_inverted_and_random():
    assert auroc([0.1, 0.2, 0.9, 0.95], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.95, 0.1, 0.2], [False, False, True, True]) == 0.0
    assert np.isclose(auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]), 0.5)


def test_auroc_is_nan_with_a_single_class():
    assert np.isnan(auroc([0.1, 0.9], [True, True]))


def test_roc_curve_endpoints():
    curve = roc_curve([0.1, 0.9], [False, True])
    assert curve["fpr"][0] == 0.0 and curve["tpr"][0] == 0.0
    assert curve["fpr"][-1] == 1.0 and curve["tpr"][-1] == 1.0


def test_binary_decision_metrics():
    out = binary_decision_metrics([True, True, False, False], [True, False, True, False])
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1 and out["tn"] == 1
    assert np.isclose(out["resolvability_accuracy"], 0.5)
    assert np.isclose(out["resolvability_f1"], 0.5)


# ------------------------------------------------------------------- MCRB
def test_mcrb_analytic_matches_the_derivation():
    f, z1, z2, delta = 500.0, 2.0, 4.0, 1.0
    out = mcrb_analytic(f, z1, z2, delta)
    assert out.applicable
    assert np.isclose(out.value, delta / (f * abs(1 / z1 - 1 / z2)))
    # At exactly B_min the differential parallax equals delta.
    assert np.isclose(differential_parallax(f, out.value, z1, z2), delta)


def test_mcrb_analytic_refuses_inapplicable_pairs():
    from intervene3d.geometry.planes import Aperture, Plane

    ap = Aperture.from_plane(Plane(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0])), 0.5, 0.3)
    static = display_hypothesis(ap, display_mode="static")
    direct = direct_hypothesis(ap)
    mirror = mirror_hypothesis(ap)
    tracked = display_hypothesis(ap, display_mode="view_tracked")

    assert analytic_pair_is_applicable(direct, static)
    assert not analytic_pair_is_applicable(direct, mirror)
    assert not analytic_pair_is_applicable(direct, tracked), "no baseline resolves a view-tracked display"
    out = mcrb_analytic(500.0, 2.0, 4.0, 1.0, h_i=direct, h_j=mirror)
    assert out.value is None and not out.applicable and "direct, static display" in out.note


def test_mcrb_analytic_rejects_fronto_planar_content():
    out = mcrb_analytic(500.0, 3.0, 3.0, 1.0)
    assert out.value is None and "no depth spread" in out.note


def test_mcrb_analytic_rejects_invalid_geometry():
    assert mcrb_analytic(500.0, -1.0, 4.0, 1.0).value is None
    assert mcrb_analytic(0.0, 2.0, 4.0, 1.0).value is None
    assert mcrb_analytic(500.0, 2.0, 4.0, 0.0).value is None


def test_mcrb_numeric_interpolates_and_reports_unresolvable():
    baselines = np.array([0.0, 0.1, 0.2, 0.3])
    out = mcrb_numeric(baselines, np.array([0.0, 0.5, 1.5, 3.0]), 1.0)
    assert 0.1 < out.value < 0.2
    unresolved = mcrb_numeric(baselines, np.array([0.0, 0.1, 0.2, 0.3]), 1.0)
    assert unresolved.value is None and "no sampled baseline" in unresolved.note


def test_mcrb_absolute_error():
    assert np.isclose(mcrb_absolute_error(0.12, 0.10), 0.02)
    assert mcrb_absolute_error(None, 0.1) is None
    assert mcrb_absolute_error(0.1, None) is None


# ------------------------------------------------------------------ regret
def test_regret_is_zero_for_the_optimal_action_and_never_negative():
    utility = np.array([1.0, 5.0, 3.0])
    assert intervention_regret(utility, 1)["regret"] == 0.0
    out = intervention_regret(utility, 0)
    assert np.isclose(out["regret"], 4.0)
    assert np.isclose(out["normalised_regret"], 0.8)
    assert out["optimal_action_index"] == 1
    assert intervention_regret(np.zeros(3), 0)["normalised_regret"] == 0.0


def test_motion_cost_combines_translation_and_rotation():
    assert np.isclose(motion_cost(0.2, 0.0), 0.2)
    assert np.isclose(motion_cost(0.2, 0.4, rotation_weight_m_per_rad=0.5), 0.4)


# --------------------------------------------------------------- aggregation
def test_summarise_and_confidence_interval():
    one = summarise([0.5])
    assert one["n"] == 1 and np.isnan(one["ci95"]) and "single seed" in one["note"]
    many = summarise([0.4, 0.5, 0.6])
    assert np.isclose(many["mean"], 0.5) and many["ci95"] > 0
    assert summarise([])["n"] == 0


def test_flatten_and_aggregate_runs():
    runs = [{"a": {"b": 1.0}, "c": 2.0}, {"a": {"b": 3.0}, "c": 4.0}]
    flat = flatten_metrics(runs[0])
    assert flat == {"a.b": 1.0, "c": 2.0}
    agg = aggregate_runs(runs)
    assert np.isclose(agg["a.b"]["mean"], 2.0)
    assert format_pm(agg["c"]).startswith("3.000")
