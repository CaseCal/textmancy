# Annotator Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manually-run benchmark that measures annotator accuracy (micro P/R/F1), latency, and token cost on frozen, human-reviewed gold labels, and commit an `LLMAnnotator` (`openai:gpt-4o`) baseline.

**Architecture:** A top-level `benchmark/` package (not shipped in the wheel) with small focused modules: `dataset.py` (gold file model, page regeneration, alias string match, review-gate checks), `metrics.py` (micro P/R/F1 and threshold sweep), `usage.py` (a Pydantic AI `WrapperModel` that counts tokens, plus a pricing table), and two CLIs, `label.py` (silver labels + review report) and `run.py` (run an annotator, write a results JSON, regenerate `RESULTS.md`). Logic lives in plain functions that take an `Annotator` instance so tests use `FunctionModel` fakes and never reach a provider.

**Tech Stack:** Python 3.10, Pydantic v2, Pydantic AI 2.54 (`WrapperModel`, `FunctionModel`), pytest, argparse, python-dotenv.

**Spec:** `docs/superpowers/specs/2026-10-24-pydantic-ai-jev-design.md`, section "Issue 2 — Annotator benchmark". GitHub issue #7.

## Global Constraints

- No version bump (waits until #6, #7, #8 are all on `main`).
- Benchmark is not part of `pytest`'s run of the benchmark itself (costs money, needs keys). Unit tests of benchmark *logic* are fine and must never call a real provider (`test/conftest.py` already sets `ALLOW_MODEL_REQUESTS = False`).
- Datasets: Snows of Kilimanjaro (all pages) and A Christmas Carol (fixed random sample of 60 pages, seed recorded in the file). `love_and_death.txt` excluded.
- Page text is regenerated from `source` + `segmentor` at run time, never stored.
- Silver labeling model default: `anthropic:claude-opus-5-5`, configurable.
- Alias string match: `name` and each `known_as`, case-insensitive, word-bounded.
- `run.py` refuses to run unless every page where silver and string-match disagree is `reviewed: true`.
- Metrics: micro precision/recall/F1 over all (page, target) pairs, overall + per dataset; wall-clock latency (total and mean per page); token usage and estimated cost (pricing table in `run.py`'s reach, overridable).
- Score-returning annotators: sweep thresholds 0.1–0.9 step 0.05 × every aggregator; record best-F1 config and PR curve.
- Results path: `benchmark/results/YYYY-MM-DD-<annotator>-<model>.json`.
- Acceptance bar (documented in `RESULTS.md`): candidate best F1 within **0.05** of the LLMAnnotator baseline F1.
- Line length 100 (black/isort/flake8 config already in repo).

## File Structure

```
benchmark/
  __init__.py          # empty; makes `python -m benchmark.run` work
  README.md            # how to label, review, run
  dataset.py           # GoldPage, GoldDataset, load/save, page_texts, string_match_labels,
                       # disagreement helpers, DATA_DIR, REPO_ROOT
  metrics.py           # Metrics, score(), THRESHOLDS, sweep()
  usage.py             # UsageTracker, UsageTrackingModel, PRICING, estimate_cost()
  label.py             # label_dataset_async(), render_review_report(), CLI
  run.py               # UnreviewedPagesError, check_reviewed(), run_benchmark_async(),
                       # result_path(), render_results_md(), ANNOTATORS, CLI
  data/
    snows_of_kilimanjaro.json
    christmas_carol.json
  results/             # committed result JSON files
  review/              # generated review reports (gitignored)
  RESULTS.md           # generated summary table (committed)
test/benchmark/
  test_dataset.py  test_metrics.py  test_usage.py  test_label.py  test_run.py
```

`pyproject.toml` gets `pythonpath = ["."]` under `[tool.pytest.ini_options]` so tests can `import benchmark`.

---

### Task 1: Dataset module

**Files:**
- Create: `benchmark/__init__.py` (empty), `benchmark/dataset.py`, `test/benchmark/test_dataset.py`
- Modify: `pyproject.toml` (`pythonpath`), `.gitignore` (add `benchmark/review/`)

**Interfaces:**
- Produces:
  - `REPO_ROOT: Path`, `DATA_DIR: Path`
  - `class GoldPage(BaseModel)`: `index: int`, `labels: list[int]`, `silver: list[int]`, `string_match: list[int]`, `reviewed: bool = False`; property `disagrees -> bool` (`set(silver) != set(string_match)`)
  - `class SampleSpec(BaseModel)`: `size: int`, `seed: int`
  - `class GoldDataset(BaseModel)`: `name: str`, `source: str`, `segmentor: dict`, `sample: Optional[SampleSpec] = None`, `target_class: str`, `targets: list[dict]`, `pages: list[GoldPage] = []`; methods `target_objects() -> list[BaseModel]`, `page_indices() -> list[int]`, `page_texts() -> dict[int, str]`, `unreviewed_disagreements() -> list[GoldPage]`
  - `load_dataset(path: Path) -> GoldDataset`, `save_dataset(dataset, path: Path) -> None`, `dataset_paths() -> list[Path]`
  - `string_match_labels(targets: Sequence[BaseModel], text: str) -> set[int]`

- [ ] **Step 1: Write failing tests** in `test/benchmark/test_dataset.py`

```python
import json

import pytest

from benchmark.dataset import (
    GoldDataset,
    GoldPage,
    load_dataset,
    save_dataset,
    string_match_labels,
)
from textmancy.targets import Character


def make_character(name, known_as=()):
    return Character(
        name=name,
        description="d",
        physical_description="p",
        known_as=list(known_as),
        summary_of_actions="s",
    )


def make_dataset(tmp_path, sample=None, pages=()):
    source = tmp_path / "book.txt"
    source.write_text("\n".join(f"para {i}" for i in range(25)), encoding="utf-8")
    return GoldDataset(
        name="book",
        source=str(source),
        segmentor={"type": "PageSegmentor", "paragraphs_per_page": 10},
        sample=sample,
        target_class="Character",
        targets=[make_character("Ann").model_dump()],
        pages=list(pages),
    )


def test_page_texts_regenerates_all_pages(tmp_path):
    texts = make_dataset(tmp_path).page_texts()
    assert list(texts) == [0, 1, 2]
    assert texts[2] == "\n".join(f"para {i}" for i in range(20, 25))


def test_sample_is_seeded_sorted_and_stable(tmp_path):
    dataset = make_dataset(tmp_path, sample={"size": 2, "seed": 3})
    indices = dataset.page_indices()
    assert len(indices) == 2
    assert indices == sorted(indices)
    assert indices == make_dataset(tmp_path, sample={"size": 2, "seed": 3}).page_indices()
    assert list(dataset.page_texts()) == indices


def test_unknown_segmentor_raises(tmp_path):
    dataset = make_dataset(tmp_path)
    dataset.segmentor = {"type": "Nope"}
    with pytest.raises(ValueError, match="Nope"):
        dataset.page_texts()


def test_target_objects_builds_target_class(tmp_path):
    targets = make_dataset(tmp_path).target_objects()
    assert targets == [make_character("Ann")]


def test_string_match_is_case_insensitive_word_bounded_and_uses_aliases():
    targets = [
        make_character("Ann"),
        make_character("Robert Smith", ["Bob"]),
        make_character("Tim"),
    ]
    text = "ANN met bob. Timothy was absent."
    assert string_match_labels(targets, text) == {0, 1}


def test_unreviewed_disagreements(tmp_path):
    pages = [
        GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
        GoldPage(index=1, labels=[0], silver=[0], string_match=[], reviewed=True),
        GoldPage(index=2, labels=[], silver=[], string_match=[0]),
    ]
    dataset = make_dataset(tmp_path, pages=pages)
    assert [p.index for p in dataset.unreviewed_disagreements()] == [2]


def test_save_and_load_round_trip(tmp_path):
    dataset = make_dataset(tmp_path, pages=[GoldPage(index=0, labels=[0], silver=[0],
                                                     string_match=[])])
    path = tmp_path / "out.json"
    save_dataset(dataset, path)
    assert json.loads(path.read_text(encoding="utf-8"))["pages"][0]["labels"] == [0]
    assert load_dataset(path) == dataset
```

- [ ] **Step 2: Add `pythonpath = ["."]` to `[tool.pytest.ini_options]`, create empty `benchmark/__init__.py`, run** `poetry run pytest test/benchmark/test_dataset.py -q` → FAIL (`ModuleNotFoundError: benchmark.dataset`).

- [ ] **Step 3: Implement `benchmark/dataset.py`**

```python
"""Gold dataset files: frozen targets plus per-page labels, with page text regenerated."""

import json
import random
import re
from pathlib import Path
from typing import Optional, Sequence

from pydantic import BaseModel

from textmancy import targets as target_module
from textmancy.components import segmentor as segmentor_module

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmark" / "data"


class GoldPage(BaseModel):
    index: int
    labels: list[int]
    silver: list[int]
    string_match: list[int]
    reviewed: bool = False

    @property
    def disagrees(self) -> bool:
        return set(self.silver) != set(self.string_match)


class SampleSpec(BaseModel):
    size: int
    seed: int


class GoldDataset(BaseModel):
    name: str
    source: str
    segmentor: dict
    sample: Optional[SampleSpec] = None
    target_class: str
    targets: list[dict]
    pages: list[GoldPage] = []

    def target_objects(self) -> list[BaseModel]:
        target_class = getattr(target_module, self.target_class)
        return [target_class(**t) for t in self.targets]

    def _all_pages(self) -> list[str]:
        config = dict(self.segmentor)
        segmentor_type = config.pop("type")
        segmentor_class = getattr(segmentor_module, segmentor_type, None)
        if segmentor_class is None:
            raise ValueError(f"Unknown segmentor type: {segmentor_type}")
        source = Path(self.source)
        if not source.is_absolute():
            source = REPO_ROOT / source
        return segmentor_class(**config).segment(source.read_text(encoding="utf-8"))

    def page_indices(self) -> list[int]:
        return self._indices(len(self._all_pages()))

    def _indices(self, page_count: int) -> list[int]:
        if self.sample is None:
            return list(range(page_count))
        rng = random.Random(self.sample.seed)
        return sorted(rng.sample(range(page_count), self.sample.size))

    def page_texts(self) -> dict[int, str]:
        pages = self._all_pages()
        return {i: pages[i] for i in self._indices(len(pages))}

    def unreviewed_disagreements(self) -> list[GoldPage]:
        return [p for p in self.pages if p.disagrees and not p.reviewed]


def load_dataset(path: Path) -> GoldDataset:
    return GoldDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_dataset(dataset: GoldDataset, path: Path) -> None:
    text = json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def dataset_paths() -> list[Path]:
    return sorted(DATA_DIR.glob("*.json"))


def string_match_labels(targets: Sequence[BaseModel], text: str) -> set[int]:
    """Indices of targets whose name or any known_as alias appears as a whole word."""
    found = set()
    for i, target in enumerate(targets):
        aliases = [target.name, *getattr(target, "known_as", [])]
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                found.add(i)
                break
    return found
```

- [ ] **Step 4:** `poetry run pytest test/benchmark/test_dataset.py -q` → PASS. Add `benchmark/review/` to `.gitignore`.
- [ ] **Step 5: Commit** `git add benchmark test/benchmark pyproject.toml .gitignore && git commit -m "Add benchmark gold dataset model"`

---

### Task 2: Metrics

**Files:** Create `benchmark/metrics.py`, `test/benchmark/test_metrics.py`

**Interfaces:**
- Produces:
  - `class Metrics(BaseModel)`: `tp: int`, `fp: int`, `fn: int`; computed properties exposed as fields via `model_dump` → use `@computed_field` for `precision`, `recall`, `f1` (0.0 on zero division)
  - `score(gold: Mapping[K, set[int]], predicted: Mapping[K, set[int]]) -> Metrics` (keys of `gold` define the pages; missing predictions count as empty)
  - `THRESHOLDS: list[float]` = 0.10, 0.15, …, 0.90
  - `PageScores = Mapping[K, Mapping[int, list[float]]]` (per page: target index → per-chunk probabilities)
  - `sweep(gold, scores: PageScores, aggregators: Mapping[str, Callable[[list[float]], float]], thresholds=THRESHOLDS) -> dict[str, list[dict]]` — each row `{"threshold": t, **Metrics.model_dump()}`
  - `best_config(curves: dict[str, list[dict]]) -> tuple[str, float]` (highest f1; ties → first aggregator in order, lowest threshold)

- [ ] **Step 1: Write failing tests**

```python
import pytest

from benchmark.metrics import THRESHOLDS, Metrics, best_config, score, sweep


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
        "tp", "fp", "fn", "precision", "recall", "f1"
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
```

- [ ] **Step 2:** `poetry run pytest test/benchmark/test_metrics.py -q` → FAIL (import error).
- [ ] **Step 3: Implement**

```python
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
    curves = {}
    for name, aggregate in aggregators.items():
        page_scores = {
            page: {target: aggregate(ps) for target, ps in targets.items()}
            for page, targets in scores.items()
        }
        curves[name] = [
            {
                "threshold": t,
                **score(
                    gold,
                    {p: {i for i, s in s_map.items() if s >= t} for p, s_map in page_scores.items()},
                ).model_dump(),
            }
            for t in thresholds
        ]
    return curves


def best_config(curves: dict[str, list[dict]]) -> tuple[str, float]:
    """Returns the (aggregator, threshold) with the highest F1, first wins on ties."""
    best = max(
        ((name, row) for name, rows in curves.items() for row in rows),
        key=lambda item: item[1]["f1"],
    )
    return best[0], best[1]["threshold"]
```

(`max` returns the first maximal item, giving the tie rule.)

- [ ] **Step 4:** run tests → PASS.
- [ ] **Step 5: Commit** `git commit -m "Add benchmark metrics and threshold sweep"`

---

### Task 3: Usage tracking and pricing

**Files:** Create `benchmark/usage.py`, `test/benchmark/test_usage.py`

**Interfaces:**
- Produces:
  - `@dataclass class UsageTracker`: `requests: int = 0`, `input_tokens: int = 0`, `output_tokens: int = 0`; `add(usage: RequestUsage) -> None`; `as_dict() -> dict`
  - `class UsageTrackingModel(WrapperModel)`: `__init__(wrapped: Model | str, tracker: UsageTracker)`; overrides async `request(...)` to add `response.usage` to the tracker
  - `PRICING: dict[str, tuple[float, float]]` — USD per million (input, output) tokens
  - `estimate_cost(model: str, tracker: UsageTracker, price_in: float | None = None, price_out: float | None = None) -> float | None` — overrides take precedence; `None` when no price is known

- [ ] **Step 1: Write failing tests**

```python
import pytest
from pydantic_ai import Agent

from benchmark.usage import UsageTracker, UsageTrackingModel, estimate_cost


def test_tracking_model_counts_requests_and_tokens(recording_model):
    model = recording_model(lambda system, user, i: [0])
    tracker = UsageTracker()
    agent = Agent(UsageTrackingModel(model.model, tracker), output_type=list[int])
    agent.run_sync("hello")
    agent.run_sync("again")
    assert tracker.requests == 2
    assert tracker.input_tokens > 0 and tracker.output_tokens > 0


def test_estimate_cost_uses_table_and_overrides():
    tracker = UsageTracker(requests=1, input_tokens=1_000_000, output_tokens=500_000)
    assert estimate_cost("openai:gpt-4o", tracker) == pytest.approx(2.50 + 5.00)
    assert estimate_cost("openai:gpt-4o", tracker, price_in=1, price_out=2) == pytest.approx(2)
    assert estimate_cost("unknown:model", tracker) is None
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement.** First check `WrapperModel`'s constructor: `poetry run python -c "import inspect; from pydantic_ai.models.wrapper import WrapperModel; print(inspect.signature(WrapperModel.__init__))"` and match it (expected `(self, wrapped: Model | KnownModelName)`).

```python
"""Token accounting for benchmark runs."""

from dataclasses import asdict, dataclass
from typing import Optional, Union

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.usage import RequestUsage

# USD per million tokens: (input, output). Override with --price-in/--price-out.
PRICING: dict[str, tuple[float, float]] = {
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class UsageTracker:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: RequestUsage) -> None:
        self.requests += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0

    def as_dict(self) -> dict:
        return asdict(self)


class UsageTrackingModel(WrapperModel):
    """Wraps a model and records every response's token usage."""

    def __init__(self, wrapped: Union[Model, str], tracker: UsageTracker):
        super().__init__(wrapped)
        self.tracker = tracker

    async def request(self, messages, model_settings, model_request_parameters):
        response = await super().request(messages, model_settings, model_request_parameters)
        self.tracker.add(response.usage)
        return response


def estimate_cost(
    model: str,
    tracker: UsageTracker,
    price_in: Optional[float] = None,
    price_out: Optional[float] = None,
) -> Optional[float]:
    table_in, table_out = PRICING.get(model, (None, None))
    price_in = table_in if price_in is None else price_in
    price_out = table_out if price_out is None else price_out
    if price_in is None or price_out is None:
        return None
    return (tracker.input_tokens * price_in + tracker.output_tokens * price_out) / 1_000_000
```

- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** `git commit -m "Add benchmark usage tracking and pricing"`

---

### Task 4: Silver labeling (`label.py`)

**Files:** Create `benchmark/label.py`, `test/benchmark/test_label.py`

**Interfaces:**
- Consumes: `GoldDataset`, `GoldPage`, `string_match_labels`, `save_dataset`, `load_dataset`, `dataset_paths` (Task 1); `UsageTracker`, `UsageTrackingModel` (Task 3); `textmancy.utils.gather_bounded`; `LLMAnnotator`
- Produces:
  - `async label_dataset_async(dataset: GoldDataset, annotator: Annotator, max_concurrency: int = 10) -> GoldDataset` — labels only pages missing from `dataset.pages`; keeps existing pages untouched; new pages get `silver = sorted(annotator result)`, `string_match = sorted(string_match_labels(...))`, `labels = silver`, `reviewed = False`; returns a copy with pages sorted by index
  - `render_review_report(dataset: GoldDataset) -> str` — Markdown, one section per unreviewed disagreement page with the page text, both label sets by `[i] name`, and the diff (`Only silver`, `Only string match`)
  - CLI: `python -m benchmark.label [names...] [--model anthropic:claude-opus-5-5] [--max-concurrency 10]`, writes data files and `benchmark/review/<name>.md`

- [ ] **Step 1: Write failing tests**

```python
from benchmark.dataset import GoldDataset, GoldPage
from benchmark.label import label_dataset_async, render_review_report
from textmancy.components import LLMAnnotator
from textmancy.targets import Character
from textmancy.utils import run_sync


def make_character(name, known_as=()):
    return Character(name=name, description="d", physical_description="p",
                     known_as=list(known_as), summary_of_actions="s")


def make_dataset(tmp_path, pages=()):
    source = tmp_path / "book.txt"
    source.write_text("Ann walks.\nBob runs.\nNobody here.", encoding="utf-8")
    return GoldDataset(
        name="book",
        source=str(source),
        segmentor={"type": "PageSegmentor", "paragraphs_per_page": 1},
        target_class="Character",
        targets=[make_character("Ann").model_dump(), make_character("Bob").model_dump()],
        pages=list(pages),
    )


def test_label_dataset_records_silver_and_string_match(tmp_path, recording_model):
    # Silver says page 0 → [0], page 1 → [] (missed Bob), page 2 → [0] (hallucinated Ann)
    answers = {"Ann walks.": [0], "Bob runs.": [], "Nobody here.": [0]}
    model = recording_model(lambda system, user, i: answers[user.removeprefix("Text:\n")])
    dataset = make_dataset(tmp_path)
    annotator = LLMAnnotator(dataset.target_objects(), model=model.model)

    labeled = run_sync(label_dataset_async(dataset, annotator))

    assert [(p.index, p.silver, p.string_match, p.labels, p.reviewed) for p in labeled.pages] == [
        (0, [0], [0], [0], False),
        (1, [], [1], [], False),
        (2, [0], [], [0], False),
    ]
    assert [p.index for p in labeled.unreviewed_disagreements()] == [1, 2]


def test_label_dataset_skips_existing_pages(tmp_path, recording_model):
    model = recording_model(lambda system, user, i: [])
    existing = GoldPage(index=0, labels=[1], silver=[0], string_match=[0], reviewed=True)
    dataset = make_dataset(tmp_path, pages=[existing])
    annotator = LLMAnnotator(dataset.target_objects(), model=model.model)

    labeled = run_sync(label_dataset_async(dataset, annotator))

    assert len(model.calls) == 2
    assert labeled.pages[0] == existing


def test_review_report_lists_only_unreviewed_disagreements(tmp_path):
    pages = [
        GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
        GoldPage(index=1, labels=[], silver=[], string_match=[1]),
        GoldPage(index=2, labels=[0], silver=[0], string_match=[], reviewed=True),
    ]
    report = render_review_report(make_dataset(tmp_path, pages=pages))
    assert "## Page 1" in report
    assert "Bob runs." in report
    assert "Only string match: [1] Bob" in report
    assert "## Page 0" not in report and "## Page 2" not in report
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement**

```python
"""
Silver-labels gold datasets with a strong LLM and writes a review report of every page where
the LLM disagrees with an alias string match.

Usage: python -m benchmark.label [dataset names...] [--model MODEL] [--max-concurrency N]
"""

import argparse
import asyncio

import dotenv

from textmancy.components import Annotator, LLMAnnotator
from textmancy.utils import gather_bounded

from .dataset import (
    REPO_ROOT,
    GoldDataset,
    GoldPage,
    dataset_paths,
    load_dataset,
    save_dataset,
    string_match_labels,
)
from .usage import UsageTracker, UsageTrackingModel

REVIEW_DIR = REPO_ROOT / "benchmark" / "review"
DEFAULT_MODEL = "anthropic:claude-opus-5-5"


async def label_dataset_async(
    dataset: GoldDataset, annotator: Annotator, max_concurrency: int = 10
) -> GoldDataset:
    """Labels every page not already in the dataset; existing pages are left untouched."""
    targets = dataset.target_objects()
    done = {page.index for page in dataset.pages}
    todo = {i: text for i, text in dataset.page_texts().items() if i not in done}

    async def _label(index: int, text: str) -> GoldPage:
        silver = sorted(await annotator.annotate_async(text))
        return GoldPage(
            index=index,
            labels=silver,
            silver=silver,
            string_match=sorted(string_match_labels(targets, text)),
        )

    new_pages = await gather_bounded((_label(i, t) for i, t in todo.items()), max_concurrency)
    pages = sorted([*dataset.pages, *new_pages], key=lambda page: page.index)
    return dataset.model_copy(update={"pages": pages})


def _names(indices, targets) -> str:
    return ", ".join(f"[{i}] {targets[i].name}" for i in sorted(indices)) or "(none)"


def render_review_report(dataset: GoldDataset) -> str:
    targets = dataset.target_objects()
    pages = dataset.unreviewed_disagreements()
    texts = dataset.page_texts() if pages else {}
    lines = [
        f"# Review: {dataset.name}",
        "",
        f"{len(pages)} page(s) need review. Fix `labels` in the dataset JSON, then set "
        "`reviewed: true` on each page.",
        "",
    ]
    for page in pages:
        silver, matched = set(page.silver), set(page.string_match)
        lines += [
            f"## Page {page.index}",
            "",
            "```text",
            texts[page.index],
            "```",
            "",
            f"- Silver: {_names(silver, targets)}",
            f"- String match: {_names(matched, targets)}",
            f"- Only silver: {_names(silver - matched, targets)}",
            f"- Only string match: {_names(matched - silver, targets)}",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="Dataset names (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrency", type=int, default=10)
    args = parser.parse_args()
    dotenv.load_dotenv()

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for path in dataset_paths():
        if args.datasets and path.stem not in args.datasets:
            continue
        dataset = load_dataset(path)
        tracker = UsageTracker()
        annotator = LLMAnnotator(
            dataset.target_objects(), model=UsageTrackingModel(args.model, tracker)
        )
        dataset = asyncio.run(label_dataset_async(dataset, annotator, args.max_concurrency))
        save_dataset(dataset, path)
        report = REVIEW_DIR / f"{dataset.name}.md"
        report.write_text(render_review_report(dataset), encoding="utf-8")
        print(
            f"{dataset.name}: {len(dataset.pages)} pages, "
            f"{len(dataset.unreviewed_disagreements())} need review -> {report}; "
            f"usage {tracker.as_dict()}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** `git commit -m "Add benchmark silver labeling and review report"`

---

### Task 5: Benchmark runner core (`run.py`)

**Files:** Create `benchmark/run.py`, `test/benchmark/test_run.py`

**Interfaces:**
- Consumes: Tasks 1–3; `textmancy.utils.gather_bounded`, `chunk_input`
- Produces:
  - `class UnreviewedPagesError(RuntimeError)`
  - `check_reviewed(datasets: Sequence[GoldDataset]) -> None` — raises `UnreviewedPagesError` naming dataset and page indices; also raises if a dataset has no pages or its page indices differ from `page_indices()` (unlabeled pages)
  - `DEFAULT_AGGREGATORS: dict[str, Aggregator] = {"max": max, "mean": statistics.fmean}` (Issue 3 replaces with the library's built-ins)
  - `async run_benchmark_async(datasets, annotators: Mapping[str, Annotator], max_concurrency=10, chunk_size=4000, aggregators=DEFAULT_AGGREGATORS) -> dict` with keys `metrics` (`{"overall": Metrics dump, "per_dataset": {name: Metrics dump}}`), `best_config` (`None` or `{"aggregator", "threshold"}`), `sweep` (`None` or curves), `latency` (`{"total_seconds", "mean_page_seconds", "pages"}`), `predictions` (`{name: {str(page_index): sorted labels}}`)
  - `annotators` maps dataset name → annotator, because each dataset has its own target list. One call covers all datasets so overall micro metrics and best-config selection span them.
  - Score-returning annotator = has `annotate_scores_async(text, chunk_size) -> dict[int, float]`. The runner chunks each page with `chunk_input(text, chunk_size)` and calls `annotate_scores_async(chunk, chunk_size)` per chunk, so it can apply every aggregator to the per-chunk scores without extra requests. Its `metrics`/`predictions` are those of the best config.

- [ ] **Step 1: Write failing tests**

```python
import pytest

from benchmark.dataset import GoldDataset, GoldPage
from benchmark.run import UnreviewedPagesError, check_reviewed, run_benchmark_async
from textmancy.components import Annotator, LLMAnnotator
from textmancy.targets import Character
from textmancy.utils import run_sync


def make_character(name):
    return Character(name=name, description="d", physical_description="p", known_as=[],
                     summary_of_actions="s")


def make_dataset(tmp_path, name="book", pages=None):
    source = tmp_path / f"{name}.txt"
    source.write_text("Ann walks.\nBob runs.", encoding="utf-8")
    if pages is None:
        pages = [
            GoldPage(index=0, labels=[0], silver=[0], string_match=[0]),
            GoldPage(index=1, labels=[1], silver=[1], string_match=[1]),
        ]
    return GoldDataset(
        name=name,
        source=str(source),
        segmentor={"type": "PageSegmentor", "paragraphs_per_page": 1},
        target_class="Character",
        targets=[make_character("Ann").model_dump(), make_character("Bob").model_dump()],
        pages=pages,
    )


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
    assert set(result["sweep"]) == {"max", "mean"}
    assert len(result["sweep"]["max"]) == 17
    assert result["best_config"] == {"aggregator": "max", "threshold": 0.1}
    assert result["metrics"]["overall"]["f1"] == pytest.approx(1.0)
    assert result["predictions"] == {"book": {"0": [0], "1": [1]}}
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement** (CLI is added in Task 6)

```python
"""
Runs an annotator over the reviewed gold datasets and records accuracy, latency, and cost.

Usage: python -m benchmark.run --annotator llm --model openai:gpt-4o
"""

import statistics
import time
from typing import Mapping, Sequence

from textmancy.components import Annotator
from textmancy.utils import chunk_input, gather_bounded

from .dataset import GoldDataset
from .metrics import Aggregator, best_config, score, sweep

DEFAULT_AGGREGATORS: dict[str, Aggregator] = {"max": max, "mean": statistics.fmean}


class UnreviewedPagesError(RuntimeError):
    """Raised when gold labels are incomplete or still need human review."""


def check_reviewed(datasets: Sequence[GoldDataset]) -> None:
    problems = []
    for dataset in datasets:
        labeled = sorted(page.index for page in dataset.pages)
        if labeled != dataset.page_indices():
            problems.append(f"{dataset.name}: pages are unlabeled; run benchmark/label.py")
            continue
        unreviewed = [page.index for page in dataset.unreviewed_disagreements()]
        if unreviewed:
            problems.append(f"{dataset.name}: unreviewed disagreement pages {unreviewed}")
    if problems:
        raise UnreviewedPagesError("Gold labels are not ready:\n" + "\n".join(problems))


async def run_benchmark_async(
    datasets: Sequence[GoldDataset],
    annotators: Mapping[str, Annotator],
    max_concurrency: int = 10,
    chunk_size: int = 4000,
    aggregators: dict[str, Aggregator] = DEFAULT_AGGREGATORS,
) -> dict:
    scoring = all(hasattr(a, "annotate_scores_async") for a in annotators.values())

    async def _page(key, text):
        annotator = annotators[key[0]]
        start = time.perf_counter()
        if scoring:
            chunk_scores = [
                await annotator.annotate_scores_async(chunk, chunk_size)
                for chunk in chunk_input(text, chunk_size)
            ]
            output = {
                i: [scores.get(i, 0.0) for scores in chunk_scores]
                for i in range(len(annotator.targets))
            }
        else:
            output = await annotator.annotate_async(text, chunk_size)
        return key, output, time.perf_counter() - start

    gold, pages = {}, []
    for dataset in datasets:
        labels = {page.index: set(page.labels) for page in dataset.pages}
        for index, text in dataset.page_texts().items():
            gold[(dataset.name, index)] = labels[index]
            pages.append(((dataset.name, index), text))

    start = time.perf_counter()
    outputs = await gather_bounded((_page(key, text) for key, text in pages), max_concurrency)
    total = time.perf_counter() - start
    page_seconds = [seconds for _, _, seconds in outputs]

    curves, config = None, None
    if scoring:
        scores = {key: output for key, output, _ in outputs}
        curves = sweep(gold, scores, aggregators)
        name, threshold = best_config(curves)
        config = {"aggregator": name, "threshold": threshold}
        aggregate = aggregators[name]
        predicted = {
            key: {i for i, ps in s.items() if aggregate(ps) >= threshold}
            for key, s in scores.items()
        }
    else:
        predicted = {key: output for key, output, _ in outputs}

    names = [dataset.name for dataset in datasets]
    return {
        "metrics": {
            "overall": score(gold, predicted).model_dump(),
            "per_dataset": {
                name: score({k: v for k, v in gold.items() if k[0] == name}, predicted).model_dump()
                for name in names
            },
        },
        "best_config": config,
        "sweep": curves,
        "latency": {
            "total_seconds": total,
            "mean_page_seconds": statistics.fmean(page_seconds) if page_seconds else 0.0,
            "pages": len(page_seconds),
        },
        "predictions": {
            name: {str(k[1]): sorted(v) for k, v in predicted.items() if k[0] == name}
            for name in names
        },
    }
```

- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: Commit** `git commit -m "Add benchmark runner core"`

---

### Task 6: Results files, `RESULTS.md`, and the `run.py` CLI

**Files:** Modify `benchmark/run.py`, `test/benchmark/test_run.py`

**Interfaces:**
- Consumes: Task 5; `UsageTracker`, `UsageTrackingModel`, `estimate_cost` (Task 3); `dataset_paths`, `load_dataset` (Task 1)
- Produces:
  - `RESULTS_DIR`, `RESULTS_MD` paths; `BASELINE = ("llm", "openai:gpt-4o")`; `ACCEPTANCE_DELTA = 0.05`
  - `result_path(annotator: str, model: str, day: date) -> Path` → `RESULTS_DIR / f"{day.isoformat()}-{annotator}-{slug}.json"`, slug = model with every non-`[A-Za-z0-9._-]` char replaced by `-`
  - `ANNOTATORS: dict[str, Callable[[list[BaseModel], str, UsageTracker], Annotator]]` with `"llm"` → `LLMAnnotator(targets, model=UsageTrackingModel(model, tracker))`
  - `render_results_md(results: list[dict]) -> str` — results sorted by `(date, annotator, model)`; overall table columns `Date | Annotator | Model | Config | P | R | F1 | ΔF1 vs baseline | Total s | s/page | Tokens in/out | Cost`; per-dataset F1 table; acceptance-bar paragraph; baseline = latest result matching `BASELINE`, ΔF1 shows `—` without one
  - `load_results() -> list[dict]`
  - CLI: `--annotator {llm} --model M [--datasets ...] [--max-concurrency N] [--chunk-size N] [--price-in X --price-out Y]`; loads dotenv; `check_reviewed` (exits non-zero with the message); runs; writes result JSON with `date`, `annotator`, `model`, `datasets`, `usage`, `cost_usd` plus run output; rewrites `RESULTS.md`

- [ ] **Step 1: Write failing tests** (append to `test/benchmark/test_run.py`)

```python
from datetime import date

from benchmark.run import render_results_md, result_path


def test_result_path_slugs_model():
    path = result_path("llm", "openai:gpt-4o", date(2026, 10, 24))
    assert path.name == "2026-10-24-llm-openai-gpt-4o.json"


def make_result(annotator, model, f1, day="2026-10-24", config=None):
    metrics = {"tp": 1, "fp": 0, "fn": 0, "precision": 0.9, "recall": 0.8, "f1": f1}
    return {
        "date": day, "annotator": annotator, "model": model, "best_config": config,
        "metrics": {"overall": metrics, "per_dataset": {"book": metrics}},
        "latency": {"total_seconds": 12.0, "mean_page_seconds": 1.5, "pages": 8},
        "usage": {"requests": 8, "input_tokens": 1000, "output_tokens": 50},
        "cost_usd": 0.0031,
    }


def test_results_md_reports_delta_against_baseline():
    md = render_results_md([
        make_result("jev", "jev-latest", 0.80, config={"aggregator": "max", "threshold": 0.45}),
        make_result("llm", "openai:gpt-4o", 0.83),
    ])
    assert "within **0.05**" in md
    llm_row = next(line for line in md.splitlines() if "| llm |" in line)
    jev_row = next(line for line in md.splitlines() if "| jev |" in line)
    assert "0.830" in llm_row and "+0.000" in llm_row
    assert "max @ 0.45" in jev_row and "-0.030" in jev_row
    assert "$0.0031" in llm_row and "1000 / 50" in llm_row


def test_results_md_without_baseline_uses_dash():
    md = render_results_md([make_result("jev", "jev-latest", 0.8)])
    row = next(line for line in md.splitlines() if "| jev |" in line)
    assert "| — |" in row
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: Implement** — add to `benchmark/run.py`:

```python
import argparse
import asyncio
import json
import re
import sys
from datetime import date
from typing import Callable

import dotenv
from pydantic import BaseModel

from textmancy.components import LLMAnnotator

from .dataset import REPO_ROOT, dataset_paths, load_dataset
from .usage import UsageTracker, UsageTrackingModel, estimate_cost

RESULTS_DIR = REPO_ROOT / "benchmark" / "results"
RESULTS_MD = REPO_ROOT / "benchmark" / "RESULTS.md"
BASELINE = ("llm", "openai:gpt-4o")
ACCEPTANCE_DELTA = 0.05

AnnotatorFactory = Callable[[list[BaseModel], str, UsageTracker], Annotator]
ANNOTATORS: dict[str, AnnotatorFactory] = {
    "llm": lambda targets, model, tracker: LLMAnnotator(
        targets, model=UsageTrackingModel(model, tracker)
    ),
}


def result_path(annotator: str, model: str, day: date) -> Path:
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", model)
    return RESULTS_DIR / f"{day.isoformat()}-{annotator}-{slug}.json"


def load_results() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS_DIR.glob("*.json"))]


def _config(result: dict) -> str:
    config = result.get("best_config")
    return f"{config['aggregator']} @ {config['threshold']:.2f}" if config else "—"


def render_results_md(results: list[dict]) -> str:
    results = sorted(results, key=lambda r: (r["date"], r["annotator"], r["model"]))
    baselines = [r for r in results if (r["annotator"], r["model"]) == BASELINE]
    baseline_f1 = baselines[-1]["metrics"]["overall"]["f1"] if baselines else None
    lines = [
        "# Annotator Benchmark Results",
        "",
        "Generated by `python -m benchmark.run`; do not edit by hand.",
        "",
        f"**Acceptance bar:** a candidate annotator's best-configuration F1 must be within "
        f"**{ACCEPTANCE_DELTA}** of the `{BASELINE[0]}` (`{BASELINE[1]}`) baseline F1 on the "
        "same data.",
        "",
        "| Date | Annotator | Model | Config | P | R | F1 | ΔF1 vs baseline | Total s | s/page "
        "| Tokens in/out | Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m, lat, usage = r["metrics"]["overall"], r["latency"], r["usage"]
        delta = "—" if baseline_f1 is None else f"{m['f1'] - baseline_f1:+.3f}"
        cost = "—" if r.get("cost_usd") is None else f"${r['cost_usd']:.4f}"
        lines.append(
            f"| {r['date']} | {r['annotator']} | {r['model']} | {_config(r)} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {delta} "
            f"| {lat['total_seconds']:.1f} | {lat['mean_page_seconds']:.2f} "
            f"| {usage['input_tokens']} / {usage['output_tokens']} | {cost} |"
        )
    names = sorted({name for r in results for name in r["metrics"]["per_dataset"]})
    lines += [
        "",
        "## F1 by dataset",
        "",
        "| Date | Annotator | Model | " + " | ".join(names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for r in results:
        per = r["metrics"]["per_dataset"]
        cells = " | ".join(f"{per[n]['f1']:.3f}" if n in per else "—" for n in names)
        lines.append(f"| {r['date']} | {r['annotator']} | {r['model']} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--annotator", choices=sorted(ANNOTATORS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="*", help="Dataset names (default: all)")
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--price-in", type=float, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens")
    args = parser.parse_args()
    dotenv.load_dotenv()

    datasets = [
        load_dataset(p)
        for p in dataset_paths()
        if not args.datasets or p.stem in args.datasets
    ]
    try:
        check_reviewed(datasets)
    except UnreviewedPagesError as error:
        sys.exit(str(error))

    tracker = UsageTracker()
    annotators = {
        d.name: ANNOTATORS[args.annotator](d.target_objects(), args.model, tracker)
        for d in datasets
    }
    output = asyncio.run(
        run_benchmark_async(datasets, annotators, args.max_concurrency, args.chunk_size)
    )
    today = date.today()
    result = {
        "date": today.isoformat(),
        "annotator": args.annotator,
        "model": args.model,
        "datasets": [d.name for d in datasets],
        "usage": tracker.as_dict(),
        "cost_usd": estimate_cost(args.model, tracker, args.price_in, args.price_out),
        **output,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = result_path(args.annotator, args.model, today)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    RESULTS_MD.write_text(render_results_md(load_results()), encoding="utf-8")
    overall = output["metrics"]["overall"]
    print(f"F1 {overall['f1']:.3f} (P {overall['precision']:.3f}, R {overall['recall']:.3f}) "
          f"-> {path}")


if __name__ == "__main__":
    main()
```

(The executor writes Task 5 with the `annotators` mapping from the start; this note records why.)

- [ ] **Step 4:** `poetry run pytest test/benchmark -q` → PASS; `poetry run python -m benchmark.run --help` prints usage.
- [ ] **Step 5: Commit** `git commit -m "Add benchmark results output and run CLI"`

---

### Task 7: Curated gold datasets and docs

**Files:**
- Create: `benchmark/data/snows_of_kilimanjaro.json`, `benchmark/data/christmas_carol.json`, `benchmark/README.md`, `benchmark/results/.gitkeep`
- Modify: `CLAUDE.md` (Commands + Architecture: a "Benchmark" subsection)

Datasets start with `"pages": []`. Every target is a full `Character` (name, description, physical_description, known_as, summary_of_actions) written from the text. `known_as` holds names and title-like aliases only, never pronouns or generic nouns ("he", "the man"), because the string match uses them.

**Snows of Kilimanjaro** — `source: "sample_data/snows_of_kiliminjaro.txt"`, `segmentor: {"type": "PageSegmentor", "paragraphs_per_page": 10}`, `sample: null` (71 pages). Targets:
Harry (`["Harry", "Bwana"]`), Helen (`["Helen", "Memsahib"]`), Molo (`["Molo"]`), Compton (`["Compton", "Compie"]`), Williamson (`["Williamson"]`), Barker (`["Barker"]`), Herr Lent (`["Herr Lent", "Lent"]`), Tristan Tzara (`["Tristan Tzara", "Tzara"]`), Marie (`["Marie"]`).

**A Christmas Carol** — `source: "sample_data/christmas_carol.txt"`, same segmentor, `sample: {"size": 60, "seed": 1843}` (from 259 pages). Targets:
Ebenezer Scrooge (`["Scrooge", "Ebenezer"]`), Jacob Marley (`["Marley", "Jacob Marley"]`), Bob Cratchit (`["Bob", "Bob Cratchit"]`), Mrs. Cratchit (`["Mrs. Cratchit"]`), Tiny Tim (`["Tiny Tim"]`), Martha Cratchit (`["Martha"]`), Peter Cratchit (`["Peter"]`), Belinda Cratchit (`["Belinda"]`), Fred (`["Fred"]`, Scrooge's nephew), Ghost of Christmas Past (`["Ghost of Christmas Past"]`), Ghost of Christmas Present (`["Ghost of Christmas Present"]`), Ghost of Christmas Yet to Come (`["Ghost of Christmas Yet to Come", "Phantom"]`), Mr. Fezziwig (`["Fezziwig"]`), Mrs. Fezziwig (`["Mrs. Fezziwig"]`), Belle (`["Belle"]`), Fan (`["Fan"]`), Dick Wilkins (`["Dick Wilkins"]`), Old Joe (`["Old Joe"]`), Mrs. Dilber (`["Mrs. Dilber"]`), Topper (`["Topper"]`), Caroline (`["Caroline"]`).

Ambiguous aliases (bare "Ghost"/"Spirit", "Cratchit") are deliberately left out; the review step resolves those pages.

- [ ] **Step 1:** Write both JSON files (use `save_dataset` from a scratch script so formatting matches), verify `load_dataset(p).page_indices()` gives 71 and 60 indices and every target validates as `Character`.
- [ ] **Step 2:** Write `benchmark/README.md`: purpose; prerequisites (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` in `.env`); steps `python -m benchmark.label` → review `benchmark/review/*.md`, edit `labels`, set `reviewed: true` → `python -m benchmark.run --annotator llm --model openai:gpt-4o` → commit `benchmark/results/*.json` and `RESULTS.md`; data format; acceptance bar.
- [ ] **Step 3:** Update `CLAUDE.md`: add benchmark commands and a short "Benchmark" architecture subsection.
- [ ] **Step 4:** `poetry run pytest -q` all pass; `poetry run flake8 textmancy benchmark --count --select=E9,F63,F7,F82 --show-source --statistics` → 0; `poetry run black --check benchmark test/benchmark && poetry run isort --check benchmark test/benchmark`.
- [ ] **Step 5: Commit** `git commit -m "Add curated benchmark datasets and docs"`

---

### Task 8: Label, review, baseline (needs API keys and a human)

- [ ] `poetry run python -m benchmark.label` (needs `ANTHROPIC_API_KEY`). Commit data files with silver labels.
- [ ] Human reviews every page in `benchmark/review/*.md`, fixes `labels`, sets `reviewed: true`. Commit.
- [ ] `poetry run python -m benchmark.run --annotator llm --model openai:gpt-4o` (needs `OPENAI_API_KEY`). Commit `benchmark/results/<date>-llm-openai-gpt-4o.json` and `benchmark/RESULTS.md`.

### Task 9: Pull request

- [ ] Push `benchmark-setup`; open PR with base `pydantic-ai-migration`, body `Closes #7`. If Task 8 is not finished, the PR body lists the remaining manual steps.
