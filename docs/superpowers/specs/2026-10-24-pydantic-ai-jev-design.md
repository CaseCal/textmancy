# Pydantic AI Migration, Annotator Benchmark, and Jev Annotator — Design

**Date:** 2026-10-24
**Status:** Approved in brainstorming, pending spec review

## Goal

Replace LangChain with Pydantic AI so every LLM component is model-agnostic, and add a
`JevAnnotator` that uses TypeSafe's Jev classification model to annotate pages faster and
cheaper than an LLM — with a benchmark proving it is not significantly less accurate.

## Delivery

Three issues, merged to `main` in order. Each issue gets its own implementation plan and PR.

| # | Issue | Depends on |
|---|-------|------------|
| 1 | Pydantic AI migration (incl. Processor bugfix) | — |
| 2 | Annotator benchmark | 1 |
| 3 | JevAnnotator | 1, 2 |

**Versioning:** No version bump in any issue. After all three are merged to `main`, bump once with
`bump2version minor` (0.3.0 → 0.4.0; the `Annotator` API change is breaking).

---

## Issue 1 — Pydantic AI migration

### Dependencies

- Remove `langchain`, `langchain-openai`.
- Add `pydantic-ai-slim[openai]` as a core dependency (brings Pydantic v2).
- Add optional extras that pass through to `pydantic-ai-slim` provider extras:
  `textmancy[anthropic]`, `textmancy[google]`.
- CI matrix: drop Python 3.9 (already below `python = "^3.10"` and unsupported by Pydantic AI);
  test 3.10, 3.11, 3.12.

### Model selection

Every LLM component accepts `model: str | pydantic_ai.models.Model`, default `"openai:gpt-4o"`.
Strings use Pydantic AI's `provider:model` syntax; provider keys come from the usual env vars.
`Processor(model=...)` passes through to Extractor and Consolidator as today.

### Layout

```
textmancy/
  targets.py              # Character, Theme → pydantic v2 BaseModel
  utils.py                # + run_sync(), gather_bounded()
  components/
    segmentor.py          # unchanged
    extractor.py
    consolidator.py
    processor.py
    annotators/
      __init__.py
      base.py             # Annotator ABC
      llm.py              # LLMAnnotator
```

`textmancy.components` re-exports `Annotator`, `LLMAnnotator` (and later `JevAnnotator`) alongside
the existing names.

### Concurrency

- Async core. Every component exposes `<verb>_async` coroutines: `extract_async`,
  `consolidate_async`, `process_async`, `annotate_async`.
- Sync methods (`extract`, `consolidate`, `process`, `annotate`) are thin wrappers:
  `utils.run_sync(self.<verb>_async(...))`.
- `run_sync` uses `asyncio.run`. If called while an event loop is already running, it raises
  `RuntimeError` telling the caller to use the `_async` variant.
- `utils.gather_bounded(coros, limit)` runs coroutines under an `asyncio.Semaphore`; components
  take `max_concurrency: int = 10` (replacing the hard-coded `ThreadPoolExecutor(max_workers=10)`).
  Results are returned in input order.

### Shared component pattern

Each LLM component builds one `pydantic_ai.Agent` in `__init__` with the existing prompt text as
`system_prompt` and an `output_type`. Each component accepts `retries: int = 1`, passed to the
Agent. API and validation errors propagate; an empty or `None` output becomes `[]`.

### Extractor

- `Agent(model, output_type=list[target_class], system_prompt=...)`. System prompt carries role,
  target definition (`__doc__`), `additional_instructions`, and examples (`model_dump()`).
- Per chunk: `agent.run(f"Text:\n{chunk}\n\nLook for around {n} {Target}s.")`.
- Input handling unchanged: `str` is sliced into `chunk_size` pieces; an iterable/generator is
  accumulated until `chunk_size`, with the remainder flushed at the end.
- The `create_model` grouped wrapper type is removed.

### Consolidator

- Same Agent pattern with `output_type=list[target_class]`.
- Each batch of `batch_size` targets is serialized as JSON via `model_dump()` in the user prompt.
- Recursion rule unchanged: re-consolidate while
  `len(results) >= target_num * tolerance and current_iter < max_iter`.

### Annotator (base)

Abstract base owning:
- `targets: list[BaseModel]`, `target_class`, `target_name`.
- Chunking of `text` by `chunk_size` (default 4000).
- Bounded concurrency over chunks.
- Abstract `_annotate_chunk_async(chunk) -> set[int]`. Subclasses may extend `annotate*` with
  extra keyword arguments (JevAnnotator adds `threshold`) but keep the base signature compatible.
- `annotate_async(text, chunk_size=4000) -> set[int]` = union over chunks;
  `annotate(...)` is the sync wrapper.

### LLMAnnotator

- `LLMAnnotator(targets, model="openai:gpt-4o", max_concurrency=10, retries=1)`.
- `output_type = list[Literal[0, 1, ..., n-1]]` built dynamically from `len(targets)`, so
  out-of-range indices fail validation and trigger Pydantic AI's retry instead of being silently
  filtered.
- Prompt lists targets with explicit indices: `[0] {json}`, `[1] {json}`, ….

### Targets

`Character` and `Theme` import from `pydantic` directly. All `.dict()` calls become
`.model_dump()`.

### Processor bugfix

`Processor.process()` is annotated as returning `TextmancyResult` but returns the consolidated list,
and `TextmancyResult` is unused. Fix: return type becomes `list[BaseModel]`, and `TextmancyResult` is
deleted. Annotation is **not** wired into Processor.

### Docs and example

- `example.py` uses `LLMAnnotator`.
- `CLAUDE.md` Architecture and Key Dependencies sections updated (Pydantic AI, model strings,
  `_async` methods, annotators package).
- `README.md` is currently empty; add a short install + usage section.

### Tests

- `test/conftest.py` sets `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`.
- Extractor: str chunking, generator chunking incl. remainder flush, examples in prompt, empty
  output → `[]` — via `TestModel` / `FunctionModel`.
- Consolidator: batching, recursion stops at `max_iter`, stops when under `target_num * tolerance`.
- LLMAnnotator: indices unioned across chunks; out-of-range index triggers retry then succeeds;
  empty output → `set()`.
- Processor: passes model and args through; returns consolidated list.
- `run_sync`: returns result; raises inside a running loop. `gather_bounded`: never exceeds limit,
  preserves input order.
- Existing segmentor and utils tests unchanged and passing.

---

## Issue 2 — Annotator benchmark

Measures annotator accuracy, latency, and cost on fixed targets and pages, so extraction
variance is excluded. Not part of `pytest`; run manually (costs money, needs keys).

### Layout

```
benchmark/
  data/
    snows_of_kilimanjaro.json
    christmas_carol.json
  label.py        # silver labeling + disagreement report
  run.py          # run an annotator, write results
  results/        # YYYY-MM-DD-<annotator>-<model>.json
  RESULTS.md      # committed summary table
```

### Gold data format

```json
{
  "source": "sample_data/snows_of_kiliminjaro.txt",
  "segmentor": {"type": "PageSegmentor", "paragraphs_per_page": 10},
  "target_class": "Character",
  "targets": [ { "name": "...", "description": "...", "known_as": ["..."], ... } ],
  "pages": [ { "index": 0, "labels": [0, 2], "reviewed": true } ]
}
```

Page text is regenerated from `source` + `segmentor` at run time rather than duplicated.
`targets` are hand-curated and frozen.

### Datasets

- **Snows of Kilimanjaro:** all pages (~70).
- **A Christmas Carol:** a fixed random sample of 60 pages (seed recorded in the file), to keep
  labeling and review cost bounded.
- `love_and_death.txt` is excluded (one page).

### Labeling — silver + human review

1. `label.py` labels every page with a strong model (default `anthropic:claude-opus-5-5`,
   configurable) using `LLMAnnotator`.
2. It also computes an alias string match (`name` and each `known_as`, case-insensitive,
   word-bounded) per page.
3. Pages where the two disagree are written to a review report with the page text, both label
   sets, and the diff. All other pages are marked `reviewed: false`.
4. The human edits labels in the JSON and sets `reviewed: true` on reviewed pages.
5. `run.py` refuses to run unless every page where silver and string-match disagreed is
   `reviewed: true`.

Unreviewed silver labels would bias the benchmark toward the LLM annotator; the review step is
mandatory.

### Metrics

Per run, over all (page, target) pairs:
- Precision, recall, F1 (micro-averaged), plus per-dataset breakdown.
- Wall-clock latency (total and mean per page).
- Token usage and estimated cost (pricing table in `run.py`, overridable).

Annotators that expose scores (JevAnnotator) additionally get a sweep over thresholds 0.1–0.9 in
steps of 0.05 and every built-in aggregator, recording the best-F1 configuration and the PR curve.

### Baseline

Issue 2 ends by committing an `LLMAnnotator` (`openai:gpt-4o`) baseline result and `RESULTS.md`.

### Acceptance bar (used in Issue 3)

JevAnnotator's best-configuration F1 must be within **0.05** of the LLMAnnotator baseline F1 on
the same data.

---

## Issue 3 — JevAnnotator

### Dependency

Optional extra `textmancy[jev]` → `typesafe-sdk`. `jev.py` imports `typesafe_sdk` lazily; if
missing, raises `ImportError("JevAnnotator requires the 'jev' extra: pip install textmancy[jev]")`.
Auth via `TYPESAFE_API_KEY` (read by the SDK).

### API

```python
JevAnnotator(
    targets,
    model="jev-latest",
    threshold=0.5,
    aggregate="max",            # "max" | "noisy_or" | "mean" | Callable[[list[float]], float]
    min_prob=0.2,               # floor, used by "noisy_or" only
    max_questions_per_request=100,
    max_concurrency=10,
    client=None,                # injectable TypeSafeClient (tests)
)

annotate(text, chunk_size=4000, threshold=None) -> set[int]
annotate_async(text, chunk_size=4000, threshold=None) -> set[int]
annotate_scores(text, chunk_size=4000) -> dict[int, float]
annotate_scores_async(text, chunk_size=4000) -> dict[int, float]
```

- `threshold=None` on a call falls back to the constructor threshold.
- `annotate*` = `{i for i, p in annotate_scores*(...).items() if p >= threshold}`.
- `annotate_scores*` returns a score for every target index.

### Request construction

One `system_one` call per chunk (split further if targets exceed `max_questions_per_request`),
one Noul per target:

```python
questions = {
    f"target_{i}": Noul(
        instructions={
            "question": f"Does the text include or refer to this {Target}?",
            Target: t.model_dump(),
        },
        criteria=NoulCriteria(
            true=f"This {Target} appears or is referenced, including by alias",
            false=f"This {Target} does not appear",
        ),
    )
    for i, t in enumerate(targets)
}
```

The chunk text is the `state`. Answers are read from `response.answers[f"target_{i}"].noul`.

### Async

If `typesafe-sdk` provides an async client, use it. Otherwise wrap sync calls with
`asyncio.to_thread` under the same semaphore. (To be confirmed when the SDK is installed during
planning.)

### Aggregation across chunks

`textmancy/components/annotators/aggregation.py`, pure functions over one target's per-chunk
probabilities:

- `max_(ps)` — default; most conservative.
- `noisy_or(ps, min_prob=0.2)` — `1 - ∏(1 - p)` over `p >= min_prob`; `0.0` if none survive.
  Treats chunks as independent evidence; the floor stops low background noise compounding
  across many chunks.
- `mean(ps)` — pervasiveness rather than presence.
- A user callable `Callable[[list[float]], float]`.

`annotate*` and `annotate_scores*` use the same aggregator.

### Tests

- Aggregation: max, mean; noisy-OR `[0.45]*3 ≈ 0.834`; noisy-OR `[0.10]*20` with floor → `0.0`,
  without floor (`min_prob=0`) ≈ `0.878`; custom callable.
- Question construction: one Noul per target with expected keys/instructions.
- `max_questions_per_request` splits targets across calls and merges answers.
- Threshold: constructor default; per-call override; `annotate` agrees with `annotate_scores`.
- Missing SDK → `ImportError` with install hint.
- All via an injected fake client returning canned `noul` values.
- Live smoke test marked `@pytest.mark.live`, skipped without `TYPESAFE_API_KEY`.

### Benchmark acceptance

Issue 3 is done when a JevAnnotator benchmark run is committed to `benchmark/results/`,
`RESULTS.md` is updated with accuracy, latency, and cost against the baseline, and the best
configuration meets the 0.05 F1 bar. If it misses, record the result and stop for a decision
before merging.

### Docs

- `example.py`: commented `JevAnnotator` block demonstrating `annotate_scores`.
- `CLAUDE.md` and `README.md`: document `JevAnnotator`, the `jev` extra, and aggregators.

---

## Out of scope

- Jev for Extractor/Consolidator (generative tasks, not classification).
- Response caching, streaming, token/cost tracking in the library itself (benchmark only).
- Wiring annotation into `Processor`.
