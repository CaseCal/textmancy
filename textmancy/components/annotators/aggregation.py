"""
Aggregators that combine one target's per-chunk probabilities into a single score.

Each takes the list of probabilities (one per chunk) and returns a value in [0, 1].
"""

import math
import statistics
from functools import partial
from typing import Callable, Sequence, Union

Aggregator = Callable[[Sequence[float]], float]


def max_(ps: Sequence[float]) -> float:
    """The highest chunk probability: present if any chunk is confident."""
    return max(ps, default=0.0)


def mean(ps: Sequence[float]) -> float:
    """The average chunk probability: how pervasive the target is, rather than whether present."""
    return statistics.fmean(ps) if ps else 0.0


def noisy_or(ps: Sequence[float], min_prob: float = 0.2) -> float:
    """
    The probability that at least one chunk is a true detection, `1 - prod(1 - p)`.

    Treats chunks as independent evidence, so several moderate detections add up. Probabilities
    below `min_prob` are dropped first so low background noise does not compound across many
    chunks; returns 0.0 if none survive.
    """
    kept = [p for p in ps if p >= min_prob]
    if not kept:
        return 0.0
    return 1.0 - math.prod(1.0 - p for p in kept)


AGGREGATORS: dict[str, Aggregator] = {"max": max_, "mean": mean, "noisy_or": noisy_or}


def get_aggregator(aggregate: Union[str, Aggregator], min_prob: float = 0.2) -> Aggregator:
    """
    Resolves an aggregator name or callable.

    Args:
        aggregate: "max", "mean", "noisy_or", or a callable taking a list of probabilities.
        min_prob: The floor passed to "noisy_or"; ignored otherwise.
    """
    if callable(aggregate):
        return aggregate
    if aggregate == "noisy_or":
        return partial(noisy_or, min_prob=min_prob)
    if aggregate not in AGGREGATORS:
        raise ValueError(f"Unknown aggregate {aggregate!r}; expected one of {sorted(AGGREGATORS)}")
    return AGGREGATORS[aggregate]
