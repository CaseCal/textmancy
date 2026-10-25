# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install --with dev --all-extras

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest test/test_segmentor.py

# Run a single test by name
poetry run pytest test/test_segmentor.py::test_segmentor_init

# Lint (strict — errors only)
poetry run flake8 textmancy --count --select=E9,F63,F7,F82 --show-source --statistics

# Lint (full — warnings allowed)
poetry run flake8 textmancy --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Build package
poetry build

# Annotator benchmark (manual; calls paid APIs, see benchmark/README.md)
poetry run python -m benchmark.label
poetry run python -m benchmark.run --annotator llm --model openai:gpt-4o
poetry run python -m benchmark.run --annotator jev --model jev-latest
```

## Architecture

Textmancy is a Python library for LLM-driven text analysis. It extracts, consolidates, and annotates features (e.g. characters, themes) from literary text using Pydantic AI. It is model-agnostic: every LLM component takes a Pydantic AI model string such as `"openai-chat:gpt-4o"` or `"anthropic:claude-sonnet-5-5"`, or a `Model` instance.

Every LLM component is async-first: `extract_async`, `consolidate_async`, `process_async`, `annotate_async`, each with a sync wrapper (`extract`, …) built on `utils.run_sync`. Concurrency is bounded by `max_concurrency` (default 10) via `utils.gather_bounded`.

### Pipeline

```
Text → Segmentor → Extractor → Consolidator → (Processor returns results)
                                                ↓
                                            Annotator → annotated pages
```

- **Segmentor** (`components/segmentor.py`): Splits raw text into segments. `ParagraphSegmentor` splits by newline; `PageSegmentor` groups paragraphs into pages (default 10/page). Both handle `max_length` via split/truncate/raise strategies.
- **Extractor** (`components/extractor.py`): Chunks text (or a stream of text) and runs a Pydantic AI `Agent` with `output_type=list[Target]` on each chunk concurrently.
- **Consolidator** (`components/consolidator.py`): Merges duplicate targets by sending batches to an LLM, re-running until the list is within `target_num * tolerance` or `max_iter` is reached.
- **Processor** (`components/processor.py`): Orchestrates Extractor → Consolidator into a single `process(texts)` call; returns the consolidated list.
- **Annotators** (`components/annotators/`): `Annotator` is an abstract base that chunks text and unions per-chunk results. `LLMAnnotator` asks an LLM for the indices of targets present, with the output type restricted to valid indices. `JevAnnotator` (`jev.py`, `textmancy[jev]` extra, `TYPESAFE_API_KEY`) asks TypeSafe's Jev classifier one Noul (yes/no) question per target via `AsyncTypeSafeClient.system_one`; `annotate_scores*` returns a probability per target, `annotate*` thresholds it (constructor default 0.5, per-call override). Per-chunk probabilities are combined by `aggregation.py` (`max_` default, `noisy_or` with a `min_prob` floor, `mean`, or a callable). The SDK is imported lazily; tests inject a fake `client`.

### Target Models

Defined in `targets.py`. Current built-in targets:
- `Character`: name, description, physical_description, known_as (aliases), summary_of_actions
- `Theme`: name, reasoning

Targets drive the structured output schema passed to the LLM.

### Key Dependencies

- `pydantic-ai-slim[openai]` — LLM orchestration and structured output. Extras: `textmancy[anthropic]`, `textmancy[google]`.
- `pydantic` v2 — target models
- `thefuzz` — declared but currently unused
- `typesafe-sdk` (optional `jev` extra) � Jev classification for `JevAnnotator`
- `python-dotenv` (dev) — loading API keys from `.env` in local dev

### Testing

Tests never call real providers: `test/conftest.py` sets `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`. Use the `recording_model` fixture (a `FunctionModel` wrapper) to return canned output and inspect prompts. Tests that call real paid APIs are marked `@pytest.mark.live`; they are deselected by default (`-m "not live"` in `addopts`) and run with `poetry run pytest -m live`.

### Benchmark

`benchmark/` (not shipped in the package) scores annotators against frozen targets and
human-reviewed page labels in `benchmark/data/*.json`. `label.py` writes silver labels plus a
review report of pages where the LLM and an alias string match disagree; `run.py` refuses to run
until those pages are `reviewed: true` and every page's `text_hash` matches its regenerated text,
then writes `benchmark/results/*.json` (named with a digest of the gold data) and regenerates
`RESULTS.md`. A candidate annotator's F1 (2-fold cross-validated when it tunes an aggregator and
threshold) must be within 0.05 of the `openai:gpt-4o` `LLMAnnotator` baseline. Its logic is unit-tested in `test/benchmark/`.

### Linting Config

Max line length is **100** (from `.flake8`). Ignored rules: E203, W503.

### Versioning

Uses `bump2version`. Current version tracked in `pyproject.toml` and `.bumpversion.cfg`.
