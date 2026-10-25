import pytest

from benchmark.metrics import (
    THRESHOLDS,
    Metrics,
    best_config,
    cross_validate,
    predict,
    score,
    sweep,
)


def test_score_micro_averages_over_pages():
    gold = {"a": {0, 1}, "b": {2}}
    predicted = {"a": {0, 3}, "b": set()}
    m = score(gold, predicted)
    assert (m.tp, m.fp, m.fn) == (1, 1, 2)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(1 / 3)
    assert m.f1 == pytest.approx(0.4)


def test_score_missing_prediction_is_empty_and_zero_division_is_zero():
    m = score({"a": set()}, {})
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_metrics_dump_includes_derived_fields():
    assert set(Metrics(tp=1, fp=0, fn=0).model_dump()) == {
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    }


def test_thresholds_span_point_one_to_point_nine():
    assert THRESHOLDS[0] == 0.1 and THRESHOLDS[-1] == 0.9 and len(THRESHOLDS) == 17


def test_sweep_applies_each_aggregator_and_threshold():
    gold = {"p": {0}}
    scores = {"p": {0: [0.3, 0.8], 1: [0.4]}}
    curves = sweep(gold, scores, {"max": max, "min": min}, thresholds=[0.35, 0.5])
    assert [row["f1"] for row in curves["max"]] == pytest.approx([2 / 3, 1.0])
    assert [row["f1"] for row in curves["min"]] == pytest.approx([0.0, 0.0])
    assert curves["max"][1]["threshold"] == 0.5
    assert best_config(curves) == ("max", 0.5)


def test_predict_applies_aggregate_and_threshold():
    scores = {"p": {0: [0.3, 0.8], 1: [0.4]}}
    assert predict(scores, max, 0.5) == {"p": {0}}
    assert predict(scores, min, 0.35) == {"p": {1}}


def test_cross_validate_scores_each_fold_with_config_tuned_on_the_other():
    # Alone, page a favours a low threshold and page b a high one, so each fold's config
    # misjudges the other: the honest F1 (0.5) is below the in-sample best (0.8 at 0.2).
    gold = {"a": {0}, "b": {0}}
    scores = {"a": {0: [0.3]}, "b": {0: [0.5], 1: [0.3]}}

    predicted, configs = cross_validate(
        gold, scores, {"max": max}, folds=[["a"], ["b"]], thresholds=[0.2, 0.4]
    )

    assert configs == [
        {"aggregator": "max", "threshold": 0.4},
        {"aggregator": "max", "threshold": 0.2},
    ]
    assert predicted == {"a": set(), "b": {0, 1}}
    assert score(gold, predicted).f1 == pytest.approx(0.5)
    assert best_config(sweep(gold, scores, {"max": max}, thresholds=[0.2, 0.4])) == ("max", 0.2)
