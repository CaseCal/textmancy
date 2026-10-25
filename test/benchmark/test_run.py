from datetime import date

import pytest

from benchmark.dataset import GoldDataset, GoldPage, page_hash
from benchmark.run import (
    UnreviewedPagesError,
    check_reviewed,
    render_results_md,
    result_path,
    run_benchmark_async,
)
from textmancy.components import Annotator, LLMAnnotator
from textmancy.targets import Character
from textmancy.utils import run_sync


def make_character(name):
    return Character(
        name=name, description="d", physical_description="p", known_as=[], summary_of_actions="s"
    )


def make_dataset(tmp_path, name="book", pages=None):
    source = tmp_path / f"{name}.txt"
    source.write_text("Ann walks.\nBob runs.", encoding="utf-8")
    if pages is None:
        pages = [
            GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
            GoldPage(index=1, labels=[1], silver=[1], string_match=[1]),
        ]
    dataset = GoldDataset(
        name=name,
        source=str(source),
        segmentor={"type": "PageSegmentor", "paragraphs_per_page": 1},
        target_class="Character",
        targets=[make_character("Ann").model_dump(), make_character("Bob").model_dump()],
        pages=pages,
    )
    texts = dataset.page_texts()
    for page in dataset.pages:
        page.text_hash = page_hash(texts[page.index])
    return dataset


def test_check_reviewed_raises_on_unreviewed_disagreement(tmp_path):
    pages = [
        GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
        GoldPage(index=1, labels=[1], silver=[], string_match=[1]),
    ]
    with pytest.raises(UnreviewedPagesError, match=r"book.*\[1\]"):
        check_reviewed([make_dataset(tmp_path, pages=pages)])


def test_check_reviewed_raises_on_unlabeled_pages(tmp_path):
    pages = [GoldPage(index=0, labels=[0], silver=[0], string_match=[0])]
    with pytest.raises(UnreviewedPagesError, match="label.py"):
        check_reviewed([make_dataset(tmp_path, pages=pages)])


def test_check_reviewed_raises_when_page_text_changed(tmp_path):
    dataset = make_dataset(tmp_path)
    (tmp_path / "book.txt").write_text("Ann walks.\nBob sprints.", encoding="utf-8")

    with pytest.raises(UnreviewedPagesError, match=r"book.*\[1\].*text_hash"):
        check_reviewed([dataset])


def test_check_reviewed_passes_when_clean(tmp_path):
    check_reviewed([make_dataset(tmp_path)])


def test_run_scores_llm_annotator(tmp_path, recording_model):
    # Correct on page 0, wrong on page 1 (says Ann instead of Bob)
    model = recording_model(lambda system, user, i: [0])
    dataset = make_dataset(tmp_path)
    annotator = LLMAnnotator(dataset.target_objects(), model=model.model)

    result = run_sync(run_benchmark_async([dataset], {"book": annotator}))

    overall = result["metrics"]["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 1)
    assert result["metrics"]["per_dataset"]["book"] == overall
    assert result["predictions"] == {"book": {"0": [0], "1": [0]}}
    assert result["latency"]["pages"] == 2
    assert result["latency"]["total_seconds"] >= result["latency"]["mean_page_seconds"] >= 0
    assert result["best_config"] is None and result["sweep"] is None
    assert result["cross_validation"] is None and result["scores"] is None


def test_run_splits_metrics_per_dataset(tmp_path, recording_model):
    model = recording_model(lambda system, user, i: [0] if "Ann" in user else [1])
    first = make_dataset(tmp_path, name="first")
    second = make_dataset(
        tmp_path,
        name="second",
        pages=[
            GoldPage(index=0, labels=[], silver=[], string_match=[], reviewed=True),
            GoldPage(index=1, labels=[1], silver=[1], string_match=[1]),
        ],
    )
    annotators = {
        d.name: LLMAnnotator(d.target_objects(), model=model.model) for d in (first, second)
    }

    result = run_sync(run_benchmark_async([first, second], annotators))

    per = result["metrics"]["per_dataset"]
    assert per["first"]["f1"] == pytest.approx(1.0)
    assert (per["second"]["tp"], per["second"]["fp"]) == (1, 1)
    assert result["metrics"]["overall"]["tp"] == 3
    assert result["latency"]["pages"] == 4


class FakeScoringAnnotator(Annotator):
    """Scores Ann 0.7 on 'Ann' pages and Bob 0.4 on 'Bob' pages."""

    def __init__(self, targets):
        super().__init__(targets)
        self.calls = []

    async def _annotate_chunk_async(self, chunk):
        raise NotImplementedError

    async def annotate_scores_async(self, text, chunk_size=4000):
        self.calls.append(text)
        return {0: 0.7 if "Ann" in text else 0.0, 1: 0.4 if "Bob" in text else 0.0}


def test_run_sweeps_score_returning_annotator(tmp_path):
    dataset = make_dataset(tmp_path)
    annotator = FakeScoringAnnotator(dataset.target_objects())

    result = run_sync(run_benchmark_async([dataset], {"book": annotator}))

    assert len(annotator.calls) == 2
    assert set(result["sweep"]) == {"max", "mean", "noisy_or"}
    assert len(result["sweep"]["max"]) == 17
    assert result["best_config"] == {"aggregator": "max", "threshold": 0.1}
    assert result["metrics"]["overall"]["f1"] == pytest.approx(1.0)
    assert result["predictions"] == {"book": {"0": [0], "1": [1]}}


class TableScoringAnnotator(FakeScoringAnnotator):
    """Returns fixed scores for each page text."""

    def __init__(self, targets, table):
        super().__init__(targets)
        self.table = table

    async def annotate_scores_async(self, text, chunk_size=4000):
        self.calls.append(text)
        return self.table[text]


def test_run_reports_cross_validated_metrics_for_score_returning_annotator(tmp_path):
    # Alone, page 0 favours threshold 0.1 and page 1 favours 0.35, so each fold's config
    # misjudges the other: cross-validated F1 is 0.5 though the in-sample best is 0.8.
    dataset = make_dataset(tmp_path)
    annotator = TableScoringAnnotator(
        dataset.target_objects(),
        {"Ann walks.": {0: 0.3}, "Bob runs.": {0: 0.3, 1: 0.5}},
    )

    result = run_sync(run_benchmark_async([dataset], {"book": annotator}))

    assert result["best_config"] == {"aggregator": "max", "threshold": 0.1}
    assert result["cross_validation"] == {
        "folds": 2,
        "configs": [
            {"aggregator": "max", "threshold": 0.35},
            {"aggregator": "max", "threshold": 0.1},
        ],
        "in_sample_f1": pytest.approx(0.8),
    }
    assert result["metrics"]["overall"]["f1"] == pytest.approx(0.5)
    assert result["predictions"] == {"book": {"0": [], "1": [0, 1]}}
    assert result["scores"] == {
        "book": {"0": {"0": [0.3], "1": [0.0]}, "1": {"0": [0.3], "1": [0.5]}}
    }


def test_run_scores_each_chunk_separately(tmp_path):
    dataset = make_dataset(tmp_path)
    annotator = FakeScoringAnnotator(dataset.target_objects())

    run_sync(run_benchmark_async([dataset], {"book": annotator}, chunk_size=4))

    assert annotator.calls[:3] == ["Ann ", "walk", "s."]


def test_jev_annotator_is_swept(tmp_path):
    from types import SimpleNamespace

    from textmancy.components import JevAnnotator

    class FakeClient:
        async def system_one(self, state, questions, *, model=None):
            names = {"target_0": "Ann", "target_1": "Bob"}
            nouls = {q: SimpleNamespace(noul=0.8 if names[q] in state else 0.1) for q in questions}
            return SimpleNamespace(nouls=nouls)

    dataset = make_dataset(tmp_path)
    annotator = JevAnnotator(dataset.target_objects(), client=FakeClient())

    result = run_sync(run_benchmark_async([dataset], {"book": annotator}))

    assert result["metrics"]["overall"]["f1"] == pytest.approx(1.0)
    assert result["best_config"]["aggregator"] == "max"


def test_result_path_slugs_model():
    path = result_path("llm", "openai:gpt-4o", date(2026, 10, 24), "0123456789ab")
    assert path.name == "2026-10-24-llm-openai-gpt-4o-0123456789ab.json"


def make_result(annotator, model, f1, day="2026-10-24", config=None, gold="g1"):
    metrics = {"tp": 1, "fp": 0, "fn": 0, "precision": 0.9, "recall": 0.8, "f1": f1}
    return {
        "date": day,
        "annotator": annotator,
        "model": model,
        "gold_digest": gold,
        "best_config": config,
        "metrics": {"overall": metrics, "per_dataset": {"book": metrics}},
        "latency": {"total_seconds": 12.0, "mean_page_seconds": 1.5, "pages": 8},
        "usage": {"requests": 8, "input_tokens": 1000, "output_tokens": 50},
        "cost_usd": 0.0031,
    }


def test_results_md_reports_delta_against_baseline():
    md = render_results_md(
        [
            make_result("jev", "jev-latest", 0.80, config={"aggregator": "max", "threshold": 0.45}),
            make_result("llm", "openai:gpt-4o", 0.83),
        ]
    )
    assert "within **0.05**" in md
    assert "cross-validated" in md
    llm_row = next(line for line in md.splitlines() if "| llm |" in line)
    jev_row = next(line for line in md.splitlines() if "| jev |" in line)
    assert "0.830" in llm_row and "+0.000" in llm_row
    assert "max @ 0.45" in jev_row and "-0.030" in jev_row
    assert "$0.0031" in llm_row and "1000 / 50" in llm_row


def test_results_md_without_baseline_uses_dash():
    md = render_results_md([make_result("jev", "jev-latest", 0.8)])
    row = next(line for line in md.splitlines() if "| jev |" in line)
    assert "| — |" in row


def test_results_md_compares_only_against_baseline_on_same_gold():
    md = render_results_md(
        [
            make_result("llm", "openai:gpt-4o", 0.83, day="2026-10-24", gold="g1"),
            make_result("llm", "openai:gpt-4o", 0.90, day="2026-10-25", gold="g2"),
            make_result("jev", "jev-a", 0.80, day="2026-10-26", gold="g1"),
            make_result("jev", "jev-b", 0.80, day="2026-10-26", gold="g3"),
        ]
    )
    rows = md.splitlines()
    assert "-0.030" in next(line for line in rows if "| jev-a |" in line)
    assert "| — |" in next(line for line in rows if "| jev-b |" in line)
