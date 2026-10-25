# Pydantic AI Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LangChain with Pydantic AI so Extractor, Consolidator, Processor, and the new `LLMAnnotator` are model-agnostic, async-first, and covered by offline tests.

**Architecture:** Each LLM component builds one `pydantic_ai.Agent` with a typed `output_type`. Work is done in `<verb>_async` coroutines that fan out over chunks via a bounded `asyncio.gather`; sync methods wrap them with `run_sync`. `Annotator` becomes an abstract base in `components/annotators/`, with `LLMAnnotator` as the first implementation.

**Tech Stack:** Python 3.10+, poetry 2.x, pydantic-ai-slim 2.54, pydantic 2, pytest

**Spec:** `docs/superpowers/specs/2026-10-24-pydantic-ai-jev-design.md` (section "Issue 1 — Pydantic AI migration"). GitHub issue #6.

## Global Constraints

- Python floor `^3.10`; CI matrix `["3.10", "3.11", "3.12"]`.
- Core LLM dependency: `pydantic-ai-slim = {version = "^2.54.0", extras = ["openai"]}`. No `langchain*` anywhere when done.
- Pydantic v2 only: import `BaseModel`/`Field` from `pydantic`; use `.model_dump()`, never `.dict()`.
- Default model everywhere: `"openai:gpt-4o"`. `model` params are typed `Union[str, Model]` (`from pydantic_ai.models import Model`).
- Every `Agent(...)` is constructed with `defer_model_check=True` so components can be built without API keys.
- Async method naming: `<verb>_async` (`extract_async`, `consolidate_async`, `process_async`, `annotate_async`).
- Components take `max_concurrency: int = 10` and `retries: int = 1`.
- Tests never hit the network: `test/conftest.py` sets `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`.
- **No version bump** in this issue.
- Line length 100; black + isort + flake8 run via pre-commit on every commit. If a hook reformats files and the commit aborts, re-stage the same files and commit again; never use `--no-verify`.
- Commit messages end with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Verified Pydantic AI behavior (probed against 2.54.0)

- `Agent(model, output_type=list[X], system_prompt=str, retries=n, defer_model_check=True)`; `await agent.run(prompt)` returns a result whose `.output` is the typed value.
- `FunctionModel(fn)` where `fn(messages, info) -> ModelResponse`. For a `list[...]` output type, the output tool is `info.output_tools[0].name` and its args are `{"response": [...]}`.
- `output_type=list[Literal[0, 1, 2]]` produces JSON schema `enum: [0, 1, 2]`; an out-of-range value triggers a retry. With `retries=0` it raises `pydantic_ai.exceptions.UnexpectedModelBehavior`.
- `models.ALLOW_MODEL_REQUESTS = False` blocks real providers (`RuntimeError`) but allows `TestModel`/`FunctionModel`.
- Without `defer_model_check=True`, `Agent("openai:gpt-4o")` raises `UserError` when `OPENAI_API_KEY` is unset.

## File Structure

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Swap LangChain for pydantic-ai-slim; `anthropic` / `google` extras |
| `.github/workflows/ci.yml` | Python 3.10–3.12 matrix |
| `textmancy/utils.py` | `text_generator` (existing), `chunk_input`, `run_sync`, `gather_bounded` |
| `textmancy/targets.py` | `Theme`, `Character` on Pydantic v2 |
| `textmancy/components/extractor.py` | `Extractor` |
| `textmancy/components/consolidator.py` | `Consolidator` |
| `textmancy/components/annotators/__init__.py` | Re-exports `Annotator`, `LLMAnnotator` |
| `textmancy/components/annotators/base.py` | `Annotator` ABC: chunking + bounded concurrency |
| `textmancy/components/annotators/llm.py` | `LLMAnnotator` |
| `textmancy/components/annotator.py` | **Deleted** |
| `textmancy/components/processor.py` | `Processor` (+ return type fix, `TextmancyResult` removed) |
| `textmancy/components/__init__.py` | Public exports (rebuilt incrementally) |
| `test/conftest.py` | Blocks real requests; `recording_model` fixture |
| `test/test_utils.py` | + tests for new utils |
| `test/test_extractor.py`, `test/test_consolidator.py`, `test/test_annotators.py`, `test/test_processor.py` | New |
| `example.py`, `CLAUDE.md`, `README.md` | Docs |

---

### Task 1: Dependencies, test harness, and async utilities

Swapping dependencies breaks every LangChain module, so this task also trims `components/__init__.py` to only the segmentors. Later tasks add each component back as it's migrated. The old modules stay on disk, unimported, until their task replaces them.

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `textmancy/components/__init__.py`
- Modify: `textmancy/utils.py`
- Create: `test/conftest.py`
- Test: `test/test_utils.py`

**Interfaces:**
- Produces:
  - `textmancy.utils.chunk_input(input_data: Union[str, Iterable[str]], chunk_size: int) -> list[str]`
  - `textmancy.utils.run_sync(coro: Coroutine[Any, Any, T]) -> T`
  - `async textmancy.utils.gather_bounded(coros: Iterable[Awaitable[T]], limit: int) -> list[T]` (results in input order)
  - pytest fixture `recording_model` → the `RecordingModel` class. `RecordingModel(respond)` where `respond(system: str, user: str, call_index: int) -> Any` returns the value placed in the output tool's `"response"` arg. Attributes: `.model` (a `FunctionModel`), `.calls` (`list[dict]` with keys `"system"`, `"user"`).

- [ ] **Step 1: Swap dependencies**

  Run:
  ```bash
  poetry remove langchain langchain-openai
  poetry add "pydantic-ai-slim[openai]@^2.54.0"
  ```

  Then edit `pyproject.toml` so `[tool.poetry.dependencies]` reads exactly as below, and add the `[tool.poetry.extras]` table directly after it:
  ```toml
  [tool.poetry.dependencies]
  python = "^3.10"
  thefuzz = "^0.20.0"
  pydantic-ai-slim = {version = "^2.54.0", extras = ["openai"]}
  anthropic = {version = ">=1.9.0", optional = true}
  google-genai = {version = ">=2.25.0", optional = true}

  [tool.poetry.extras]
  anthropic = ["anthropic"]
  google = ["google-genai"]
  ```
  (The version floors match pydantic-ai-slim's own `anthropic` / `google` extras.)

  Run: `poetry lock && poetry install --with dev`
  Expected: completes without resolver errors.

- [ ] **Step 2: Update CI matrix**

  In `.github/workflows/ci.yml`, replace:
  ```yaml
        python-version: ["3.9", "3.10", "3.11"]
  ```
  with:
  ```yaml
        python-version: ["3.10", "3.11", "3.12"]
  ```

- [ ] **Step 3: Trim component exports**

  Replace the contents of `textmancy/components/__init__.py` with:
  ```python
  from .segmentor import PageSegmentor, ParagraphSegmentor

  __all__ = [
      "ParagraphSegmentor",
      "PageSegmentor",
  ]
  ```

  Run: `poetry run pytest -q`
  Expected: 14 passed (the segmentor and utils tests import cleanly again).

- [ ] **Step 4: Create the test harness**

  Create `test/conftest.py`:
  ```python
  from typing import Any, Callable

  import pytest
  from pydantic_ai import models
  from pydantic_ai.messages import (
      ModelMessage,
      ModelResponse,
      SystemPromptPart,
      ToolCallPart,
      UserPromptPart,
  )
  from pydantic_ai.models.function import AgentInfo, FunctionModel

  # Never let a test reach a real provider.
  models.ALLOW_MODEL_REQUESTS = False

  Responder = Callable[[str, str, int], Any]


  class RecordingModel:
      """
      Wraps a FunctionModel that records each request's prompts and replies with canned
      structured output.

      Args:
          respond: Called as respond(system_prompt, user_prompt, call_index); its return value is
              sent as the output tool's "response" argument.
      """

      def __init__(self, respond: Responder):
          self.calls: list[dict[str, str]] = []
          self._respond = respond
          self.model = FunctionModel(self._handle)

      def _handle(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
          parts = [part for message in messages for part in getattr(message, "parts", [])]
          system = "".join(p.content for p in parts if isinstance(p, SystemPromptPart))
          user = next(p.content for p in parts if isinstance(p, UserPromptPart))
          self.calls.append({"system": system, "user": user})
          output = self._respond(system, user, len(self.calls) - 1)
          return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": output})])


  @pytest.fixture
  def recording_model() -> type[RecordingModel]:
      return RecordingModel
  ```

- [ ] **Step 5: Write failing tests for the new utils**

  Append to `test/test_utils.py`. Change the existing import line at the top to:
  ```python
  import asyncio

  import pytest

  from textmancy.utils import chunk_input, gather_bounded, run_sync, text_generator
  ```
  and add at the end of the file:
  ```python
  def test_chunk_input_slices_strings():
      assert chunk_input("aaaabbbbcc", 4) == ["aaaa", "bbbb", "cc"]


  def test_chunk_input_empty_string():
      assert chunk_input("", 4) == []


  def test_chunk_input_accumulates_iterables_and_flushes_remainder():
      assert chunk_input(iter(["ab", "cd", "ef", "g"]), 4) == ["abcd", "efg"]


  async def _value(value):
      return value


  def test_run_sync_returns_result():
      assert run_sync(_value(42)) == 42


  def test_run_sync_inside_running_loop_raises():
      async def outer():
          with pytest.raises(RuntimeError, match="_async"):
              run_sync(_value(1))

      asyncio.run(outer())


  def test_gather_bounded_respects_limit_and_preserves_order():
      state = {"active": 0, "peak": 0}

      async def work(i):
          state["active"] += 1
          state["peak"] = max(state["peak"], state["active"])
          await asyncio.sleep(0.01 * (5 - i))
          state["active"] -= 1
          return i

      result = asyncio.run(gather_bounded([work(i) for i in range(5)], limit=2))

      assert result == [0, 1, 2, 3, 4]
      assert state["peak"] == 2


  def test_gather_bounded_rejects_non_positive_limit():
      with pytest.raises(ValueError, match="limit"):
          asyncio.run(gather_bounded([], limit=0))
  ```

- [ ] **Step 6: Run tests to verify they fail**

  Run: `poetry run pytest test/test_utils.py -q`
  Expected: collection error, `ImportError: cannot import name 'chunk_input'`.

- [ ] **Step 7: Implement the utils**

  Replace `textmancy/utils.py` with:
  ```python
  import asyncio
  from typing import Any, Awaitable, Coroutine, Generator, Iterable, TypeVar, Union

  T = TypeVar("T")


  def text_generator(texts: Iterable[str], max_chunk_size: int) -> Generator[str, None, None]:
      """
      Given a n iterable of strings, generates a stream that concatenates the strings
      into chunks of a given size.

      """
      next_chunk = ""
      for text in texts:
          # Split large text
          if len(text) > max_chunk_size:
              for i in range(0, len(text), max_chunk_size):
                  yield text[i : i + max_chunk_size]

          # Return when chunk is full
          elif len(next_chunk) + len(text) > max_chunk_size:
              if next_chunk:
                  yield next_chunk
              next_chunk = text
          else:
              next_chunk += text
      if next_chunk:
          yield next_chunk


  def chunk_input(input_data: Union[str, Iterable[str]], chunk_size: int) -> list[str]:
      """
      Splits input into chunks for LLM calls.

      A string is sliced into pieces of exactly chunk_size (the last may be shorter). Any other
      iterable of strings is accumulated until a chunk reaches at least chunk_size, and the
      remainder is flushed at the end.
      """
      if isinstance(input_data, str):
          return [input_data[i : i + chunk_size] for i in range(0, len(input_data), chunk_size)]

      chunks = []
      current = ""
      for piece in input_data:
          current += piece
          if len(current) >= chunk_size:
              chunks.append(current)
              current = ""
      if current:
          chunks.append(current)
      return chunks


  def run_sync(coro: Coroutine[Any, Any, T]) -> T:
      """
      Runs a coroutine to completion from synchronous code.

      Raises:
          RuntimeError: If called while an event loop is already running (e.g. Jupyter). Await the
              matching *_async method instead.
      """
      try:
          asyncio.get_running_loop()
      except RuntimeError:
          return asyncio.run(coro)
      coro.close()
      raise RuntimeError(
          "Cannot run synchronously inside a running event loop; await the *_async method instead."
      )


  async def gather_bounded(coros: Iterable[Awaitable[T]], limit: int) -> list[T]:
      """
      Awaits coroutines concurrently with at most `limit` in flight, returning results in input
      order.
      """
      if limit < 1:
          raise ValueError(f"limit must be at least 1, got {limit}")
      semaphore = asyncio.Semaphore(limit)

      async def _bounded(coro: Awaitable[T]) -> T:
          async with semaphore:
              return await coro

      return list(await asyncio.gather(*(_bounded(c) for c in coros)))
  ```

- [ ] **Step 8: Run tests to verify they pass**

  Run: `poetry run pytest -q`
  Expected: 21 passed.

- [ ] **Step 9: Commit**

  ```bash
  git add pyproject.toml poetry.lock .github/workflows/ci.yml textmancy/components/__init__.py textmancy/utils.py test/conftest.py test/test_utils.py
  git commit -m "Swap LangChain for pydantic-ai-slim and add async utilities"
  ```
  (If `poetry.lock` is gitignored, `git add` will warn about it; drop it from the command.)

---

### Task 2: Pydantic v2 targets and Extractor

**Files:**
- Modify: `textmancy/targets.py`
- Modify: `textmancy/components/extractor.py` (full rewrite)
- Modify: `textmancy/components/__init__.py`
- Test: `test/test_extractor.py`

**Interfaces:**
- Consumes: `utils.chunk_input`, `utils.gather_bounded`, `utils.run_sync`; `recording_model` fixture.
- Produces:
  - `Extractor(target_class: type[BaseModel], target_num: int = 3, target_examples: Optional[list[BaseModel]] = None, model: Union[str, Model] = "openai:gpt-4o", additional_instructions: str = "", max_concurrency: int = 10, retries: int = 1)`
  - `async Extractor.extract_async(input_data: Union[str, Iterable[str]], chunk_size: int = 4000, number: Optional[int] = None) -> list[BaseModel]`
  - `Extractor.extract(...)` — same signature, sync.
  - Attributes: `target_class`, `target_num`, `target_name`, `agent`.
  - System prompt contains the phrase `"first stage"` (Task 5's test relies on this). The user prompt is `f"Text:\n{chunk}\n\nLook for around {n} {Target}s."`.

- [ ] **Step 1: Migrate targets**

  Replace `textmancy/targets.py` with:
  ```python
  from typing import List

  from pydantic import BaseModel, Field


  class Theme(BaseModel):
      """A recurring idea or concept in a work of literature"""

      name: str = Field(
          ..., description="The name of the theme. Ex: Death and Mortality, Regret, etc."
      )
      reasoning: str = Field(
          ...,
          description="The reasoning behind the theme occurring in a specific piece of text",
      )


  class Character(BaseModel):
      """A person or being in a work of literature"""

      name: str = Field(..., description="The name of the character")
      description: str = Field(
          ..., description="A description of the character and their role in the story"
      )
      physical_description: str = Field(
          ..., description="A description of the character's physical appearance"
      )
      known_as: List[str] = Field(
          ..., description="A list of names the character is known as in the story"
      )
      summary_of_actions: str = Field(
          ..., description="A summary of the actions the character takes in the story"
      )
  ```

- [ ] **Step 2: Write failing tests**

  Create `test/test_extractor.py`:
  ```python
  import asyncio

  from textmancy.components.extractor import Extractor
  from textmancy.targets import Theme


  def _theme(name):
      return {"name": name, "reasoning": "r"}


  def _chunk_of(user_prompt):
      return user_prompt.split("\n\n")[0].removeprefix("Text:\n")


  def test_extract_splits_string_into_chunks_in_order(recording_model):
      rec = recording_model(lambda system, user, i: [_theme(_chunk_of(user))])
      extractor = Extractor(Theme, model=rec.model)

      result = extractor.extract("aaaabbbbcc", chunk_size=4)

      assert [t.name for t in result] == ["aaaa", "bbbb", "cc"]
      assert all(isinstance(t, Theme) for t in result)
      assert len(rec.calls) == 3


  def test_extract_accumulates_generator_input(recording_model):
      rec = recording_model(lambda system, user, i: [_theme(_chunk_of(user))])
      extractor = Extractor(Theme, model=rec.model)

      result = extractor.extract(iter(["ab", "cd", "ef", "g"]), chunk_size=4)

      assert [t.name for t in result] == ["abcd", "efg"]


  def test_extract_prompts_include_definition_examples_and_instructions(recording_model):
      rec = recording_model(lambda system, user, i: [])
      extractor = Extractor(
          Theme,
          target_num=7,
          target_examples=[Theme(name="Loss", reasoning="x")],
          additional_instructions="Be terse.",
          model=rec.model,
      )

      extractor.extract("some text")

      system, user = rec.calls[0]["system"], rec.calls[0]["user"]
      assert "first stage" in system
      assert "Theme is defined as" in system
      assert "A recurring idea or concept" in system
      assert "Be terse." in system
      assert '"name": "Loss"' in system
      assert user.startswith("Text:\nsome text")
      assert "around 7 Themes" in user


  def test_extract_number_overrides_target_num(recording_model):
      rec = recording_model(lambda system, user, i: [])
      extractor = Extractor(Theme, target_num=7, model=rec.model)

      extractor.extract("some text", number=2)

      assert "around 2 Themes" in rec.calls[0]["user"]


  def test_extract_empty_output_returns_empty_list(recording_model):
      rec = recording_model(lambda system, user, i: [])
      extractor = Extractor(Theme, model=rec.model)

      assert extractor.extract("some text") == []


  def test_extract_async(recording_model):
      rec = recording_model(lambda system, user, i: [_theme("x")])
      extractor = Extractor(Theme, model=rec.model)

      result = asyncio.run(extractor.extract_async("some text"))

      assert [t.name for t in result] == ["x"]
  ```

- [ ] **Step 3: Run tests to verify they fail**

  Run: `poetry run pytest test/test_extractor.py -q`
  Expected: collection error, `ModuleNotFoundError: No module named 'langchain'` (the old extractor still imports LangChain).

- [ ] **Step 4: Rewrite the Extractor**

  Replace `textmancy/components/extractor.py` with:
  ```python
  import json
  import logging
  from typing import Iterable, Optional, Union

  from pydantic import BaseModel
  from pydantic_ai import Agent
  from pydantic_ai.models import Model

  from .. import utils


  class Extractor:
      """
      Extracts target objects from text using an LLM.

      Attributes:
          target_class (type[BaseModel]): The class of the target to extract from the text.
          target_num (int): The expected number of targets to extract per chunk.
          agent (Agent): The Pydantic AI agent that performs extraction.
      """

      def __init__(
          self,
          target_class: type[BaseModel],
          target_num: int = 3,
          target_examples: Optional[list[BaseModel]] = None,
          model: Union[str, Model] = "openai:gpt-4o",
          additional_instructions: str = "",
          max_concurrency: int = 10,
          retries: int = 1,
      ):
          self._logger = logging.getLogger(__name__)

          self.target_class = target_class
          self.target_num = target_num
          self.target_name = target_class.__name__
          self.max_concurrency = max_concurrency

          self.agent = Agent(
              model,
              output_type=list[target_class],
              system_prompt=self._create_system_prompt(
                  target_class, additional_instructions, target_examples
              ),
              retries=retries,
              defer_model_check=True,
          )

      @staticmethod
      def _create_system_prompt(
          target_class: type[BaseModel],
          additional_instructions: str = "",
          examples: Optional[list[BaseModel]] = None,
      ) -> str:
          name = target_class.__name__
          prompt = (
              "You are an ethnographer. This is the first stage of classification. "
              f"Your job is to determine any {name}s present in a given text. "
              f"{name} is defined as: {target_class.__doc__}"
          )
          if additional_instructions:
              prompt += f"\n{additional_instructions}"
          if examples:
              prompt += f"\nExamples: {json.dumps([e.model_dump() for e in examples])}"
          return prompt

      async def _extract_from_block_async(
          self, text: str, number: Optional[int] = None
      ) -> list[BaseModel]:
          n = number or self.target_num
          result = await self.agent.run(
              f"Text:\n{text}\n\nLook for around {n} {self.target_name}s."
          )
          return result.output or []

      async def extract_async(
          self,
          input_data: Union[str, Iterable[str]],
          chunk_size: int = 4000,
          number: Optional[int] = None,
      ) -> list[BaseModel]:
          """
          Extracts targets from a string or a stream of strings.

          Args:
              input_data: A string (sliced into chunk_size pieces) or an iterable of strings
                  (accumulated into chunks of at least chunk_size).
              chunk_size: The chunk size in characters.
              number: Overrides target_num for this call.

          Returns:
              All extracted targets, in chunk order.
          """
          chunks = utils.chunk_input(input_data, chunk_size)
          results = await utils.gather_bounded(
              (self._extract_from_block_async(chunk, number) for chunk in chunks),
              self.max_concurrency,
          )
          self._logger.debug(f"Extracted from {len(chunks)} chunks")
          return [target for chunk_result in results for target in chunk_result]

      def extract(
          self,
          input_data: Union[str, Iterable[str]],
          chunk_size: int = 4000,
          number: Optional[int] = None,
      ) -> list[BaseModel]:
          """Synchronous wrapper for extract_async."""
          return utils.run_sync(self.extract_async(input_data, chunk_size, number))
  ```

- [ ] **Step 5: Export the Extractor**

  Replace `textmancy/components/__init__.py` with:
  ```python
  from .extractor import Extractor
  from .segmentor import PageSegmentor, ParagraphSegmentor

  __all__ = [
      "Extractor",
      "ParagraphSegmentor",
      "PageSegmentor",
  ]
  ```

- [ ] **Step 6: Run tests to verify they pass**

  Run: `poetry run pytest -q`
  Expected: 27 passed.

- [ ] **Step 7: Commit**

  ```bash
  git add textmancy/targets.py textmancy/components/extractor.py textmancy/components/__init__.py test/test_extractor.py
  git commit -m "Migrate targets to Pydantic v2 and Extractor to Pydantic AI"
  ```

---

### Task 3: Consolidator

**Files:**
- Modify: `textmancy/components/consolidator.py` (full rewrite)
- Modify: `textmancy/components/__init__.py`
- Test: `test/test_consolidator.py`

**Interfaces:**
- Consumes: `utils.gather_bounded`, `utils.run_sync`; `recording_model` fixture.
- Produces:
  - `Consolidator(target_class: type[BaseModel], target_num: int = 3, model: Union[str, Model] = "openai:gpt-4o", batch_size: int = 10, max_iter: int = 3, tolerance: float = 1.5, additional_instructions: str = "", max_concurrency: int = 10, retries: int = 1)`
  - `async Consolidator.consolidate_async(items: list[BaseModel], current_iter: int = 0) -> list[BaseModel]`
  - `Consolidator.consolidate(...)` — same signature, sync.
  - Attributes: `target_num`, `agent`.
  - System prompt contains `"consolidating"` (Task 5 relies on this). The user prompt is `f"{Target}s to consolidate:\n{json}"`, with the JSON produced by `json.dumps([...], indent=2)`.

- [ ] **Step 1: Write failing tests**

  Create `test/test_consolidator.py`:
  ```python
  import asyncio
  import json

  from textmancy.components.consolidator import Consolidator
  from textmancy.targets import Theme


  def _themes(n):
      return [Theme(name=f"t{i}", reasoning="r") for i in range(n)]


  def _batch_of(user_prompt):
      return json.loads(user_prompt.split("\n", 1)[1])


  def _echo(system, user, i):
      return _batch_of(user)


  def test_consolidate_batches_and_preserves_order(recording_model):
      rec = recording_model(lambda system, user, i: _batch_of(user)[:1])
      consolidator = Consolidator(Theme, target_num=3, batch_size=10, model=rec.model)

      result = consolidator.consolidate(_themes(25))

      assert [t.name for t in result] == ["t0", "t10", "t20"]
      assert len(rec.calls) == 3


  def test_consolidate_recurses_until_max_iter(recording_model):
      rec = recording_model(_echo)
      consolidator = Consolidator(Theme, target_num=3, batch_size=10, max_iter=2, model=rec.model)

      result = consolidator.consolidate(_themes(25))

      # Echo never shrinks the list: iterations 0, 1, 2 × 3 batches each.
      assert len(rec.calls) == 9
      assert len(result) == 25


  def test_consolidate_stops_when_under_tolerance(recording_model):
      rec = recording_model(_echo)
      consolidator = Consolidator(Theme, target_num=3, tolerance=1.5, model=rec.model)

      result = consolidator.consolidate(_themes(4))

      assert len(rec.calls) == 1
      assert len(result) == 4


  def test_consolidate_prompt_includes_instructions_and_targets(recording_model):
      rec = recording_model(lambda system, user, i: [])
      consolidator = Consolidator(
          Theme, additional_instructions="Merge aggressively.", model=rec.model
      )

      consolidator.consolidate(_themes(2))

      system, user = rec.calls[0]["system"], rec.calls[0]["user"]
      assert "consolidating" in system
      assert "Merge aggressively." in system
      assert user.startswith("Themes to consolidate:\n")
      assert [t["name"] for t in _batch_of(user)] == ["t0", "t1"]


  def test_consolidate_async(recording_model):
      rec = recording_model(lambda system, user, i: _batch_of(user)[:1])
      consolidator = Consolidator(Theme, model=rec.model)

      result = asyncio.run(consolidator.consolidate_async(_themes(2)))

      assert [t.name for t in result] == ["t0"]
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `poetry run pytest test/test_consolidator.py -q`
  Expected: collection error, `ModuleNotFoundError: No module named 'langchain'`.

- [ ] **Step 3: Rewrite the Consolidator**

  Replace `textmancy/components/consolidator.py` with:
  ```python
  import json
  import logging
  from typing import Union

  from pydantic import BaseModel
  from pydantic_ai import Agent
  from pydantic_ai.models import Model

  from .. import utils


  class Consolidator:
      """
      Consolidates a list of target objects by merging duplicates with an LLM.

      Attributes:
          target_class (type[BaseModel]): The class of the target objects.
          target_num (int): The desired number of target objects in the final list.
          batch_size (int): The number of targets sent per LLM call.
          max_iter (int): The maximum number of re-consolidation passes.
          tolerance (float): Re-consolidate while len(results) >= target_num * tolerance.
          agent (Agent): The Pydantic AI agent that performs consolidation.
      """

      def __init__(
          self,
          target_class: type[BaseModel],
          target_num: int = 3,
          model: Union[str, Model] = "openai:gpt-4o",
          batch_size: int = 10,
          max_iter: int = 3,
          tolerance: float = 1.5,
          additional_instructions: str = "",
          max_concurrency: int = 10,
          retries: int = 1,
      ):
          self._logger = logging.getLogger(__name__)

          self.target_class = target_class
          self.target_num = target_num
          self.batch_size = batch_size
          self.max_iter = max_iter
          self.tolerance = tolerance
          self.target_name = target_class.__name__
          self.max_concurrency = max_concurrency

          self.agent = Agent(
              model,
              output_type=list[target_class],
              system_prompt=self._create_system_prompt(target_class, additional_instructions),
              retries=retries,
              defer_model_check=True,
          )

      @staticmethod
      def _create_system_prompt(
          target_class: type[BaseModel], additional_instructions: str = ""
      ) -> str:
          name = target_class.__name__
          prompt = (
              "You are an ethnographer. In this stage, text has been freely classified. "
              "You are now consolidating various annotations to create a final list. "
              f"Consolidate the given {name}s into a list of {name}s. "
              f"Where possible, combine similar {name}s into one. "
              f"Look for cases where the same {name} is referred to by different names. "
              "Avoid repetition and redundancy."
          )
          if additional_instructions:
              prompt += f"\n{additional_instructions}"
          return prompt

      async def _consolidate_batch_async(self, targets: list[BaseModel]) -> list[BaseModel]:
          payload = json.dumps([t.model_dump() for t in targets], indent=2)
          result = await self.agent.run(f"{self.target_name}s to consolidate:\n{payload}")
          return result.output or []

      async def consolidate_async(
          self, items: list[BaseModel], current_iter: int = 0
      ) -> list[BaseModel]:
          """
          Consolidates targets in batches, re-running while the result is still too large.

          Args:
              items: The targets to consolidate.
              current_iter: The current pass; used internally for recursion.

          Returns:
              The consolidated targets.
          """
          batches = [items[i : i + self.batch_size] for i in range(0, len(items), self.batch_size)]
          results = await utils.gather_bounded(
              (self._consolidate_batch_async(batch) for batch in batches),
              self.max_concurrency,
          )
          consolidated = [target for batch_result in results for target in batch_result]

          if (
              len(consolidated) >= self.target_num * self.tolerance
              and current_iter < self.max_iter
          ):
              return await self.consolidate_async(consolidated, current_iter + 1)

          return consolidated

      def consolidate(self, items: list[BaseModel], current_iter: int = 0) -> list[BaseModel]:
          """Synchronous wrapper for consolidate_async."""
          return utils.run_sync(self.consolidate_async(items, current_iter))
  ```

- [ ] **Step 4: Export the Consolidator**

  Replace `textmancy/components/__init__.py` with:
  ```python
  from .consolidator import Consolidator
  from .extractor import Extractor
  from .segmentor import PageSegmentor, ParagraphSegmentor

  __all__ = [
      "Consolidator",
      "Extractor",
      "ParagraphSegmentor",
      "PageSegmentor",
  ]
  ```

- [ ] **Step 5: Run tests to verify they pass**

  Run: `poetry run pytest -q`
  Expected: 32 passed.

- [ ] **Step 6: Commit**

  ```bash
  git add textmancy/components/consolidator.py textmancy/components/__init__.py test/test_consolidator.py
  git commit -m "Migrate Consolidator to Pydantic AI"
  ```

---

### Task 4: Annotator base and LLMAnnotator

**Files:**
- Create: `textmancy/components/annotators/__init__.py`
- Create: `textmancy/components/annotators/base.py`
- Create: `textmancy/components/annotators/llm.py`
- Delete: `textmancy/components/annotator.py`
- Modify: `textmancy/components/__init__.py`
- Test: `test/test_annotators.py`

**Interfaces:**
- Consumes: `utils.chunk_input`, `utils.gather_bounded`, `utils.run_sync`; `recording_model` fixture.
- Produces (Issue #8's `JevAnnotator` subclasses this base):
  - `Annotator(targets: Sequence[BaseModel], max_concurrency: int = 10)` is an ABC. It raises `ValueError` on empty targets. Attributes: `targets: list[BaseModel]`, `target_class`, `target_name`, `max_concurrency`.
  - Abstract method: `async _annotate_chunk_async(self, chunk: str) -> set[int]`.
  - `async Annotator.annotate_async(text: str, chunk_size: int = 4000) -> set[int]` returns the union of indices over all chunks.
  - `Annotator.annotate(text: str, chunk_size: int = 4000) -> set[int]` is the sync wrapper.
  - `LLMAnnotator(targets: Sequence[BaseModel], model: Union[str, Model] = "openai:gpt-4o", max_concurrency: int = 10, retries: int = 1)`, with attribute `agent`.

- [ ] **Step 1: Write failing tests**

  Create `test/test_annotators.py`:
  ```python
  import asyncio

  import pytest
  from pydantic_ai.exceptions import UnexpectedModelBehavior

  from textmancy.components.annotators import Annotator, LLMAnnotator
  from textmancy.targets import Theme

  THEMES = [
      Theme(name="alpha", reasoning="r"),
      Theme(name="beta", reasoning="r"),
      Theme(name="gamma", reasoning="r"),
  ]


  def test_annotator_is_abstract():
      with pytest.raises(TypeError):
          Annotator(THEMES)


  def test_annotator_rejects_empty_targets(recording_model):
      rec = recording_model(lambda system, user, i: [])
      with pytest.raises(ValueError, match="targets"):
          LLMAnnotator([], model=rec.model)


  def test_llm_annotator_unions_indices_across_chunks(recording_model):
      rec = recording_model(lambda system, user, i: [0] if "alpha" in user else [2])
      annotator = LLMAnnotator(THEMES, model=rec.model)

      result = annotator.annotate("alpha text" + "gamma text", chunk_size=10)

      assert result == {0, 2}
      assert len(rec.calls) == 2


  def test_llm_annotator_prompt_lists_indexed_targets(recording_model):
      rec = recording_model(lambda system, user, i: [])
      annotator = LLMAnnotator(THEMES, model=rec.model)

      annotator.annotate("some text")

      system, user = rec.calls[0]["system"], rec.calls[0]["user"]
      assert '[0] {"name": "alpha"' in system
      assert '[2] {"name": "gamma"' in system
      assert user == "Text:\nsome text"


  def test_llm_annotator_retries_out_of_range_index(recording_model):
      rec = recording_model(lambda system, user, i: [7] if i == 0 else [1])
      annotator = LLMAnnotator(THEMES, model=rec.model, retries=1)

      assert annotator.annotate("some text") == {1}
      assert len(rec.calls) == 2


  def test_llm_annotator_raises_when_retries_exhausted(recording_model):
      rec = recording_model(lambda system, user, i: [7])
      annotator = LLMAnnotator(THEMES, model=rec.model, retries=0)

      with pytest.raises(UnexpectedModelBehavior):
          annotator.annotate("some text")


  def test_llm_annotator_empty_output_returns_empty_set(recording_model):
      rec = recording_model(lambda system, user, i: [])
      annotator = LLMAnnotator(THEMES, model=rec.model)

      assert annotator.annotate("some text") == set()


  def test_llm_annotator_async(recording_model):
      rec = recording_model(lambda system, user, i: [0])
      annotator = LLMAnnotator(THEMES, model=rec.model)

      assert asyncio.run(annotator.annotate_async("alpha")) == {0}
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `poetry run pytest test/test_annotators.py -q`
  Expected: collection error, `ModuleNotFoundError: No module named 'textmancy.components.annotators'`.

- [ ] **Step 3: Create the base class**

  Create `textmancy/components/annotators/base.py`:
  ```python
  import logging
  from abc import ABC, abstractmethod
  from typing import Sequence

  from pydantic import BaseModel

  from ... import utils


  class Annotator(ABC):
      """
      Base class for mapping a fixed list of targets back onto text.

      Subclasses implement _annotate_chunk_async. They may extend annotate/annotate_async with
      extra keyword arguments, but must keep the base signature compatible.

      Attributes:
          targets (list[BaseModel]): The targets to identify; results are indices into this list.
          max_concurrency (int): The maximum number of chunks annotated concurrently.
      """

      def __init__(self, targets: Sequence[BaseModel], max_concurrency: int = 10):
          if not targets:
              raise ValueError("targets must not be empty")

          self._logger = logging.getLogger(__name__)

          self.targets = list(targets)
          self.target_class = type(self.targets[0])
          self.target_name = self.target_class.__name__
          self.max_concurrency = max_concurrency

      @abstractmethod
      async def _annotate_chunk_async(self, chunk: str) -> set[int]:
          """Returns the indices of targets present in a single chunk."""

      async def annotate_async(self, text: str, chunk_size: int = 4000) -> set[int]:
          """
          Identifies which targets appear in the text.

          Args:
              text: The text to annotate.
              chunk_size: The chunk size in characters.

          Returns:
              The indices of targets found in any chunk.
          """
          chunks = utils.chunk_input(text, chunk_size)
          results = await utils.gather_bounded(
              (self._annotate_chunk_async(chunk) for chunk in chunks),
              self.max_concurrency,
          )
          self._logger.debug(f"Annotated {len(chunks)} chunks")
          return set().union(*results)

      def annotate(self, text: str, chunk_size: int = 4000) -> set[int]:
          """Synchronous wrapper for annotate_async."""
          return utils.run_sync(self.annotate_async(text, chunk_size))
  ```

- [ ] **Step 4: Create the LLMAnnotator**

  Create `textmancy/components/annotators/llm.py`:
  ```python
  import json
  from typing import Literal, Sequence, Union

  from pydantic import BaseModel
  from pydantic_ai import Agent
  from pydantic_ai.models import Model

  from .base import Annotator


  class LLMAnnotator(Annotator):
      """
      Annotates text by asking an LLM which target indices appear in each chunk.

      The output type only allows valid indices, so an out-of-range answer fails validation and
      is retried rather than silently dropped.

      Attributes:
          agent (Agent): The Pydantic AI agent that performs annotation.
      """

      def __init__(
          self,
          targets: Sequence[BaseModel],
          model: Union[str, Model] = "openai:gpt-4o",
          max_concurrency: int = 10,
          retries: int = 1,
      ):
          super().__init__(targets, max_concurrency)

          index_type = Literal[tuple(range(len(self.targets)))]
          self.agent = Agent(
              model,
              output_type=list[index_type],
              system_prompt=self._create_system_prompt(),
              retries=retries,
              defer_model_check=True,
          )

      def _create_system_prompt(self) -> str:
          listing = "\n".join(
              f"[{i}] {json.dumps(target.model_dump())}" for i, target in enumerate(self.targets)
          )
          return (
              f"You are annotating a text sample to identify {self.target_name}s. "
              f"Here is the indexed list of valid {self.target_name}s:\n{listing}\n"
              f"Return the index of every {self.target_name} that appears in, or is referred to "
              "by, the text. Return an empty list if none appear."
          )

      async def _annotate_chunk_async(self, chunk: str) -> set[int]:
          result = await self.agent.run(f"Text:\n{chunk}")
          return set(result.output or [])
  ```

- [ ] **Step 5: Create the package init and remove the old module**

  Create `textmancy/components/annotators/__init__.py`:
  ```python
  from .base import Annotator
  from .llm import LLMAnnotator

  __all__ = [
      "Annotator",
      "LLMAnnotator",
  ]
  ```

  Run: `git rm textmancy/components/annotator.py`

  Replace `textmancy/components/__init__.py` with:
  ```python
  from .annotators import Annotator, LLMAnnotator
  from .consolidator import Consolidator
  from .extractor import Extractor
  from .segmentor import PageSegmentor, ParagraphSegmentor

  __all__ = [
      "Annotator",
      "Consolidator",
      "Extractor",
      "LLMAnnotator",
      "ParagraphSegmentor",
      "PageSegmentor",
  ]
  ```

- [ ] **Step 6: Run tests to verify they pass**

  Run: `poetry run pytest -q`
  Expected: 40 passed.

- [ ] **Step 7: Commit**

  ```bash
  git add textmancy/components/annotators textmancy/components/__init__.py test/test_annotators.py
  git commit -m "Replace Annotator with abstract base and Pydantic AI LLMAnnotator"
  ```

---

### Task 5: Processor migration and return-type fix

**Files:**
- Modify: `textmancy/components/processor.py` (full rewrite)
- Modify: `textmancy/components/__init__.py`
- Test: `test/test_processor.py`

**Interfaces:**
- Consumes:
  - From Task 2: `Extractor(...)`, `extract_async`, and a system prompt containing `"first stage"`
  - From Task 3: `Consolidator(...)`, `consolidate_async`, and a system prompt containing `"consolidating"`
  - `utils.text_generator`, `utils.run_sync`
- Produces:
  - `Processor(target_class: type[BaseModel], target_num: int = 3, model: Union[str, Model] = "openai:gpt-4o", extractor_args: Optional[dict] = None, consolidator_args: Optional[dict] = None)`
  - `async Processor.process_async(texts: list[str]) -> list[BaseModel]`
  - `Processor.process(texts: list[str]) -> list[BaseModel]`
  - Attributes: `extractor`, `consolidator`
  - `TextmancyResult` no longer exists.

- [ ] **Step 1: Write failing tests**

  Create `test/test_processor.py`:
  ```python
  import asyncio

  from textmancy.components.processor import Processor
  from textmancy.targets import Theme


  def _respond(system, user, i):
      if "first stage" in system:
          return [{"name": "raw", "reasoning": "r"}]
      return [{"name": "merged", "reasoning": "r"}]


  def test_processor_extracts_then_consolidates(recording_model):
      rec = recording_model(_respond)
      processor = Processor(Theme, target_num=5, model=rec.model)

      result = processor.process(["some text"])

      assert result == [Theme(name="merged", reasoning="r")]
      assert any("first stage" in c["system"] for c in rec.calls)
      assert any("consolidating" in c["system"] for c in rec.calls)


  def test_processor_passes_target_num_and_args(recording_model):
      rec = recording_model(_respond)
      processor = Processor(
          Theme,
          target_num=5,
          model=rec.model,
          extractor_args={"additional_instructions": "EXTRACT-EXTRA"},
          consolidator_args={"additional_instructions": "CONSOLIDATE-EXTRA"},
      )

      processor.process(["some text"])

      assert processor.extractor.target_num == 3
      assert processor.consolidator.target_num == 5
      assert any("EXTRACT-EXTRA" in c["system"] for c in rec.calls)
      assert any("CONSOLIDATE-EXTRA" in c["system"] for c in rec.calls)


  def test_processor_constructs_without_api_key(monkeypatch):
      monkeypatch.delenv("OPENAI_API_KEY", raising=False)

      processor = Processor(Theme)

      assert processor.extractor.agent is not None


  def test_processor_async(recording_model):
      rec = recording_model(_respond)
      processor = Processor(Theme, model=rec.model)

      result = asyncio.run(processor.process_async(["some text"]))

      assert [t.name for t in result] == ["merged"]
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `poetry run pytest test/test_processor.py -q`
  Expected: collection error, `ModuleNotFoundError: No module named 'langchain'`.

- [ ] **Step 3: Rewrite the Processor**

  Replace `textmancy/components/processor.py` with:
  ```python
  import logging
  from typing import Optional, Union

  from pydantic import BaseModel
  from pydantic_ai.models import Model

  from .. import utils
  from .consolidator import Consolidator
  from .extractor import Extractor


  class Processor:
      """
      Extracts targets from a corpus of texts and consolidates them into a final list.

      Attributes:
          extractor (Extractor): Extracts raw targets; asked for 60% of target_num per chunk.
          consolidator (Consolidator): Merges raw targets down toward target_num.
      """

      def __init__(
          self,
          target_class: type[BaseModel],
          target_num: int = 3,
          model: Union[str, Model] = "openai:gpt-4o",
          extractor_args: Optional[dict] = None,
          consolidator_args: Optional[dict] = None,
      ):
          self._logger = logging.getLogger(__name__)

          self.extractor = Extractor(
              target_class=target_class,
              model=model,
              target_num=int(target_num * 0.6),  # Extractor extracts 60% of the targets
              **(extractor_args or {}),
          )
          self.consolidator = Consolidator(
              target_class=target_class,
              model=model,
              target_num=target_num,
              **(consolidator_args or {}),
          )

      async def process_async(self, texts: list[str]) -> list[BaseModel]:
          """
          Extracts features from the texts, then consolidates them.

          Args:
              texts: The text fragments (e.g. pages) to process.

          Returns:
              The consolidated targets.
          """
          self._logger.info("Creating text stream for extraction")
          text_stream = utils.text_generator(texts, max_chunk_size=10000)

          self._logger.info("Extracting features")
          results = await self.extractor.extract_async(text_stream)
          self._logger.debug(f"Extracted {len(results)} features")

          self._logger.info("Consolidating features")
          consolidated = await self.consolidator.consolidate_async(results)
          self._logger.debug(f"Consolidated into {len(consolidated)} features")

          return consolidated

      def process(self, texts: list[str]) -> list[BaseModel]:
          """Synchronous wrapper for process_async."""
          return utils.run_sync(self.process_async(texts))
  ```

- [ ] **Step 4: Export the Processor**

  Replace `textmancy/components/__init__.py` with:
  ```python
  from .annotators import Annotator, LLMAnnotator
  from .consolidator import Consolidator
  from .extractor import Extractor
  from .processor import Processor
  from .segmentor import PageSegmentor, ParagraphSegmentor

  __all__ = [
      "Annotator",
      "Consolidator",
      "Extractor",
      "LLMAnnotator",
      "Processor",
      "ParagraphSegmentor",
      "PageSegmentor",
  ]
  ```

- [ ] **Step 5: Run tests to verify they pass**

  Run: `poetry run pytest -q`
  Expected: 44 passed.

- [ ] **Step 6: Commit**

  ```bash
  git add textmancy/components/processor.py textmancy/components/__init__.py test/test_processor.py
  git commit -m "Migrate Processor to Pydantic AI and fix its return type"
  ```

---

### Task 6: Example, docs, and final verification

**Files:**
- Modify: `example.py`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the public exports from `textmancy.components` (Task 5's final `__init__.py`).

- [ ] **Step 1: Update example.py**

  In `example.py`, change the import line:
  ```python
  from textmancy.components import Annotator, PageSegmentor, Processor
  ```
  to:
  ```python
  from textmancy.components import LLMAnnotator, PageSegmentor, Processor
  ```
  and replace the annotator construction:
  ```python
  annotater = Annotator(
      targets=results,
      model="gpt-4o",
  )
  ```
  with:
  ```python
  annotater = LLMAnnotator(
      targets=results,
      model="openai:gpt-4o",
  )
  ```

  Run: `poetry run python -c "import ast; ast.parse(open('example.py').read())"`
  Expected: no output (it parses). Don't run `example.py` itself, since it makes paid API calls.

- [ ] **Step 2: Update CLAUDE.md**

  In `CLAUDE.md`, replace the paragraph under `## Architecture`:
  ```markdown
  Textmancy is a Python library for LLM-driven text analysis. It extracts, consolidates, and annotates features (e.g. characters, themes) from literary text using LangChain + OpenAI.
  ```
  with:
  ```markdown
  Textmancy is a Python library for LLM-driven text analysis. It extracts, consolidates, and annotates features (e.g. characters, themes) from literary text using Pydantic AI. It is model-agnostic: every LLM component takes a Pydantic AI model string such as `"openai:gpt-4o"` or `"anthropic:claude-sonnet-5-5"`, or a `Model` instance.

  Every LLM component is async-first: `extract_async`, `consolidate_async`, `process_async`, `annotate_async`, each with a sync wrapper (`extract`, …) built on `utils.run_sync`. Concurrency is bounded by `max_concurrency` (default 10) via `utils.gather_bounded`.
  ```

  Replace the Extractor, Consolidator, and Annotator bullets under `### Pipeline` with:
  ```markdown
  - **Extractor** (`components/extractor.py`): Chunks text (or a stream of text) and runs a Pydantic AI `Agent` with `output_type=list[Target]` on each chunk concurrently.
  - **Consolidator** (`components/consolidator.py`): Merges duplicate targets by sending batches to an LLM, re-running until the list is within `target_num * tolerance` or `max_iter` is reached.
  - **Processor** (`components/processor.py`): Orchestrates Extractor → Consolidator into a single `process(texts)` call; returns the consolidated list.
  - **Annotators** (`components/annotators/`): `Annotator` is an abstract base that chunks text and unions per-chunk results. `LLMAnnotator` asks an LLM for the indices of targets present, with the output type restricted to valid indices.
  ```
  (This replaces the old Processor bullet too, so Processor isn't listed twice.)

  Replace the `### Key Dependencies` list with:
  ```markdown
  - `pydantic-ai-slim[openai]` — LLM orchestration and structured output. Extras: `textmancy[anthropic]`, `textmancy[google]`.
  - `pydantic` v2 — target models
  - `thefuzz` — declared but currently unused
  - `python-dotenv` (dev) — loading API keys from `.env` in local dev
  ```

  Add this section after `### Key Dependencies`:
  ```markdown
  ### Testing

  Tests never call real providers: `test/conftest.py` sets `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False`. Use the `recording_model` fixture (a `FunctionModel` wrapper) to return canned output and inspect prompts.
  ```

- [ ] **Step 3: Write the README**

  `README.md` is currently empty. Replace it with:
  ````markdown
  # textmancy

  LLM-driven package for mapping, analyzing, and expanding text works. Extract features such as characters or themes from a text, consolidate duplicates, and annotate which features appear on each page.

  ## Install

  ```bash
  pip install textmancy              # OpenAI support
  pip install "textmancy[anthropic]" # + Anthropic
  pip install "textmancy[google]"    # + Google
  ```

  Set the API key for your provider (e.g. `OPENAI_API_KEY`).

  ## Usage

  ```python
  from textmancy.components import LLMAnnotator, PageSegmentor, Processor
  from textmancy.targets import Character

  pages = PageSegmentor(paragraphs_per_page=10).segment(text)

  characters = Processor(target_class=Character, model="openai:gpt-4o").process(pages)

  annotator = LLMAnnotator(targets=characters, model="anthropic:claude-sonnet-5-5")
  for page in pages:
      print([characters[i].name for i in annotator.annotate(page)])
  ```

  Every component also has an async variant (`process_async`, `annotate_async`, …) for use inside an event loop.
  ````

- [ ] **Step 4: Verify no LangChain remains**

  Run: `git grep -n -i langchain -- . ':!docs/'`
  Expected: no output (exit code 1).

- [ ] **Step 5: Full verification**

  Run each command and confirm the expected result:
  - `poetry run pytest -q` → 44 passed
  - `poetry run pre-commit run --all-files` → black, isort, and flake8 all report "Passed"
  - `poetry run flake8 textmancy --count --select=E9,F63,F7,F82 --show-source --statistics` → `0`
  - `poetry run python -c "from textmancy.components import Annotator, LLMAnnotator, Processor, Extractor, Consolidator, PageSegmentor, ParagraphSegmentor"` → no output

- [ ] **Step 6: Commit**

  ```bash
  git add example.py CLAUDE.md README.md
  git commit -m "Update example and docs for Pydantic AI migration"
  ```
