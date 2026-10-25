"""Micro-averaged precision/recall/F1 and threshold sweeps for score-returning annotators."""

from typing import Callable, Hashable, Mapping, Sequence

from pydantic import BaseModel, computed_field

THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(17)]

Aggregator = Callable[[list[float]], float]
PageScores = Mapping[Hashable, Mapping[int, list[float]]]


class Metrics(BaseModel):
    tp: int
    fp: int
    fn: int

    @computed_field
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @computed_field
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @computed_field
    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def score(gold: Mapping[Hashable, set[int]], predicted: Mapping[Hashable, set[int]]) -> Metrics:
    """Micro-averages over every (page, target) pair; pages are the keys of `gold`."""
    tp = fp = fn = 0
    for page, labels in gold.items():
        guess = predicted.get(page, set())
        tp += len(labels & guess)
        fp += len(guess - labels)
        fn += len(labels - guess)
    return Metrics(tp=tp, fp=fp, fn=fn)


def sweep(
    gold: Mapping[Hashable, set[int]],
    scores: PageScores,
    aggregators: Mapping[str, Aggregator],
    thresholds: Sequence[float] = THRESHOLDS,
) -> dict[str, list[dict]]:
    """Scores every aggregator × threshold pair; returns a PR curve per aggregator."""
    return {
        name: [
            {"threshold": t, **score(gold, predict(scores, aggregate, t)).model_dump()}
            for t in thresholds
        ]
        for name, aggregate in aggregators.items()
    }


def predict(
    scores: PageScores, aggregate: Aggregator, threshold: float
) -> dict[Hashable, set[int]]:
    """The targets on each page whose aggregated chunk scores reach the threshold."""
    return {
        page: {target for target, ps in targets.items() if aggregate(ps) >= threshold}
        for page, targets in scores.items()
    }


def best_config(curves: dict[str, list[dict]]) -> tuple[str, float]:
    """Returns the (aggregator, threshold) with the highest F1, first wins on ties."""
    best = max(
        ((name, row) for name, rows in curves.items() for row in rows),
        key=lambda item: item[1]["f1"],
    )
    return best[0], best[1]["threshold"]


def cross_validate(
    gold: Mapping[Hashable, set[int]],
    scores: PageScores,
    aggregators: Mapping[str, Aggregator],
    folds: Sequence[Sequence[Hashable]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[dict[Hashable, set[int]], list[dict]]:
    """
    Predicts each fold with the best config on the other folds, so the resulting F1 isn't
    inflated by tuning the aggregator and threshold on the pages it is measured on.

    Returns:
        Predictions for every page in the folds, and the config chosen for each fold.
    """
    predicted, configs = {}, []
    for fold in folds:
        held_out = set(fold)
        rest = [page for page in gold if page not in held_out]
        name, threshold = best_config(
            sweep({p: gold[p] for p in rest}, {p: scores[p] for p in rest}, aggregators, thresholds)
        )
        configs.append({"aggregator": name, "threshold": threshold})
        predicted.update(predict({p: scores[p] for p in fold}, aggregators[name], threshold))
    return predicted, configs
