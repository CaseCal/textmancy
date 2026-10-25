import pytest

from textmancy.components.annotators.aggregation import get_aggregator, max_, mean, noisy_or


def test_max_and_mean():
    assert max_([0.1, 0.7, 0.4]) == pytest.approx(0.7)
    assert mean([0.1, 0.7, 0.4]) == pytest.approx(0.4)


def test_empty_input_scores_zero():
    assert max_([]) == mean([]) == noisy_or([]) == 0.0


def test_noisy_or_combines_moderate_evidence():
    assert noisy_or([0.45] * 3) == pytest.approx(0.834, abs=1e-3)


def test_noisy_or_floor_drops_background_noise():
    assert noisy_or([0.10] * 20) == 0.0
    assert noisy_or([0.10] * 20, min_prob=0) == pytest.approx(0.878, abs=1e-3)


def test_get_aggregator_resolves_names_and_callables():
    assert get_aggregator("max") is max_
    assert get_aggregator("noisy_or", min_prob=0)([0.10] * 20) == pytest.approx(0.878, abs=1e-3)
    assert get_aggregator("noisy_or")([0.10] * 20) == 0.0

    def custom(ps):
        return min(ps)

    assert get_aggregator(custom) is custom


def test_get_aggregator_rejects_unknown_name():
    with pytest.raises(ValueError, match="median"):
        get_aggregator("median")
