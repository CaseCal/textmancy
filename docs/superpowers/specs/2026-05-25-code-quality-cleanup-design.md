# Code Quality Cleanup Design

**Date:** 2026-05-25
**Scope:** Full repository quality pass — tooling, source, tests, example

## Overview

Three-phase cleanup to enforce consistent type hints, docstrings, naming, and formatting across the textmancy repo, plus add pre-commit infrastructure to prevent future drift.

---

## Phase 1: Tooling

### Goal
Establish enforcement infrastructure so all future code is formatted and linted consistently.

### Changes

**Dependencies:**
- Add `pre-commit` to dev dependencies via `poetry add --group dev pre-commit`

**`.pre-commit-config.yaml`** — new file with three hooks:
- `black` — formatter
- `flake8` — linter
- `isort` — import ordering

**Line length alignment:**
- Standardize to **100 chars** in both `pyproject.toml` (Black config) and `.flake8`
- Currently misaligned: Black uses 88, Flake8 uses 100

**`pyproject.toml` isort config:**
- Add `[tool.isort]` section with `profile = "black"` and `line_length = 100`

### Out of Scope
- `mypy` — deferred to a future pass

---

## Phase 2: Source Code Cleanup (`textmancy/`)

### Goal
Make the source package internally consistent, correct, and well-documented.

### `textmancy/__init__.py` — Top-Level Exports
Add exports for the four primary user-facing classes:
- `Processor`
- `Extractor`
- `Consolidator`
- `Annotator`

Targets (`Character`, `Theme`) and concrete segmentors (`PageSegmentor`, `ParagraphSegmentor`) are not exported at the top level — users import from submodules.

### Type Hints
Use `typing` module style throughout (no modernization to `X | None` syntax). Specific fixes:

| File | Fix |
|------|-----|
| All `__init__` methods | Add `-> None` return type |
| `segmentor.py` | Bare `list` return types → `List[str]` |
| `extractor.py` | `target_examples` param → `Optional[List[BaseModel]]`; `number` param → `Optional[int]`; `chunk_size` → `int` |
| `annotator.py` | `chunk_size` param → `int`; `_create_annotation_prompt` return → `ChatPromptTemplate` |
| `consolidator.py` | `_create_consolidation_prompt` return → `ChatPromptTemplate` |
| `processor.py` | `extractor_args`/`consolidator_args` → `Optional[Dict[str, Any]]` |

### Bug Fixes

**Mutable default arguments (`processor.py` lines 25–26):**
```python
# Before (bug: shared dict across instances)
def __init__(self, ..., extractor_args: dict = {}, consolidator_args: dict = {}):

# After
def __init__(self, ..., extractor_args: Optional[Dict[str, Any]] = None, consolidator_args: Optional[Dict[str, Any]] = None):
    extractor_args = extractor_args or {}
    consolidator_args = consolidator_args or {}
```

**Deprecated Pydantic API (`extractor.py`, `annotator.py`):**
- `.dict()` → `.model_dump()` everywhere

### Docstrings — Add Missing
- `Processor.__init__`
- `Extractor._create_extraction_prompt`
- `Extractor.extract`
- `Consolidator._create_consolidation_prompt`
- `Annotator._create_annotation_prompt`

### Typo Fixes
| File | Typo | Correction |
|------|------|-----------|
| `utils.py` | "a n iterable" | "an iterable" |
| `targets.py` | "peice of text" | "piece of text" |
| `processor.py` | "Extracts featres" | "Extracts features" |
| `annotator.py` | comment "valdi" | "valid" |

---

## Phase 3: Tests & Example Cleanup

### Goal
Apply the same quality bar to non-source files.

### `test/test_segmentor.py`
- Add `-> None` return type to all test functions
- Fix `SampleSegmentor._segment()` bare `list` return → `List[str]`
- Add `from typing import List` if not already present

### `test/test_utils.py`
- Add `-> None` return type to all test functions

### `example.py`
- Add module-level docstring describing the script's purpose
- Fix variable name typo: `annotater` → `annotator`
- No individual function docstrings needed (script is a top-level script with no function definitions)

### No Test Function Docstrings
Test function names are self-documenting; docstrings are not required.

---

## Success Criteria

1. `pre-commit run --all-files` passes with no errors
2. All public methods and classes in `textmancy/` have type hints and docstrings
3. All `__init__` methods have `-> None` return types
4. No bare `list` return types — all use generic form (`List[str]`, etc.)
5. No mutable default arguments
6. No deprecated Pydantic `.dict()` calls
7. All test functions have `-> None` return types
8. `poetry run pytest` still passes after all changes
