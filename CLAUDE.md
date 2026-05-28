# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
poetry install --with dev

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
```

## Architecture

Textmancy is a Python library for LLM-driven text analysis. It extracts, consolidates, and annotates features (e.g. characters, themes) from literary text using LangChain + OpenAI.

### Pipeline

```
Text → Segmentor → Extractor → Consolidator → (Processor returns results)
                                                ↓
                                            Annotator → annotated pages
```

- **Segmentor** (`components/segmentor.py`): Splits raw text into segments. `ParagraphSegmentor` splits by newline; `PageSegmentor` groups paragraphs into pages (default 10/page). Both handle `max_length` via split/truncate/raise strategies.
- **Extractor** (`components/extractor.py`): Streams segments through LangChain + OpenAI to extract structured target objects. Uses `ThreadPoolExecutor` for concurrent extraction.
- **Consolidator** (`components/consolidator.py`): Deduplicates extracted features using fuzzy string matching (`thefuzz`).
- **Processor** (`components/processor.py`): Orchestrates Extractor → Consolidator into a single `process(text)` call.
- **Annotator** (`components/annotator.py`): Takes consolidated targets and uses an LLM to map them back to each page of text.

### Target Models

Defined in `targets.py`. Current built-in targets:
- `Character`: name, description, physical_description, known_as (aliases), summary_of_actions
- `Theme`: name, reasoning

Targets drive the structured output schema passed to the LLM.

### Key Dependencies

- `langchain` / `langchain-openai` — LLM orchestration and structured output
- `thefuzz` — fuzzy matching for consolidation
- `python-dotenv` — loading `OPENAI_API_KEY` from `.env` in local dev

### Linting Config

Max line length is **100** (from `.flake8`). Ignored rules: E203, W503.

### Versioning

Uses `bump2version`. Current version tracked in `pyproject.toml` and `.bumpversion.cfg`.
