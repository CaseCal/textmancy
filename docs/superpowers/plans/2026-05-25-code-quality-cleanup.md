# Code Quality Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply consistent type hints, docstrings, bug fixes, and pre-commit enforcement across the entire textmancy repo in three sequential phases.

**Architecture:** Phase 1 installs and configures pre-commit (Black + isort + Flake8 at line-length 100). Phase 2 fixes all source files in `textmancy/`. Phase 3 fixes test files and `example.py`. Each phase ends with a green test suite and a clean pre-commit run.

**Tech Stack:** Python 3.10+, poetry, pre-commit, black, isort, flake8, pytest

**Important note on `.dict()` calls:** `extractor.py` and `annotator.py` call `.dict()` on models imported from `langchain.pydantic_v1`. This is Pydantic v1's method and is correct for these models — do NOT change them to `.model_dump()`.

---

## Phase 1 — Tooling

### Task 1: Configure pyproject.toml and create .pre-commit-config.yaml

**Files:**
- Modify: `pyproject.toml`
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Update pyproject.toml**

  Replace the `[tool.black]` section and add `[tool.isort]` and dev dep. The `[tool.flake8]` section in pyproject.toml is redundant with `.flake8`; remove it. Make these exact changes:

  In `[tool.poetry.group.dev.dependencies]`, add after `pytest-cov`:
  ```toml
  pre-commit = "^3.7.0"
  black = "^24.4.2"
  isort = "^5.13.2"
  ```

  Replace:
  ```toml
  [tool.black]
  line-length = 88

  [tool.flake8]
  max-line-length = 88
  ```

  With:
  ```toml
  [tool.black]
  line-length = 100

  [tool.isort]
  profile = "black"
  line_length = 100
  ```

  The `[tool.pytest.ini_options]` and `[tool.coverage.run]` sections stay unchanged.

- [ ] **Step 2: Create .pre-commit-config.yaml**

  Create `.pre-commit-config.yaml` at the repo root:
  ```yaml
  repos:
    - repo: https://github.com/psf/black
      rev: 24.4.2
      hooks:
        - id: black

    - repo: https://github.com/pycqa/isort
      rev: 5.13.2
      hooks:
        - id: isort
          name: isort (python)

    - repo: https://github.com/pycqa/flake8
      rev: 7.0.0
      hooks:
        - id: flake8
  ```

- [ ] **Step 3: Install dependencies**

  ```
  poetry install --with dev
  ```

  Expected: installs pre-commit, black, isort without errors.

- [ ] **Step 4: Install pre-commit hooks**

  ```
  poetry run pre-commit install
  ```

  Expected: `pre-commit installed at .git/hooks/pre-commit`

- [ ] **Step 5: Run pre-commit across all files**

  ```
  poetry run pre-commit run --all-files
  ```

  Black and isort will auto-fix formatting. Flake8 will report remaining issues (none expected since the existing code already conforms to 100-char lines).

  If files were modified by Black/isort, that's expected — stage them.

- [ ] **Step 6: Run tests to confirm nothing broke**

  ```
  poetry run pytest
  ```

  Expected: all existing tests pass.

- [ ] **Step 7: Commit**

  ```bash
  git add pyproject.toml .pre-commit-config.yaml
  git commit -m "chore: add pre-commit with black, isort, flake8 at line-length 100"
  ```

---

## Phase 2 — Source Code Cleanup

### Task 2: textmancy/__init__.py — add top-level exports

**Files:**
- Modify: `textmancy/__init__.py`

- [ ] **Step 1: Replace the file content**

  Current file is empty (one blank line). Replace entirely with:
  ```python
  from textmancy.components.annotator import Annotator
  from textmancy.components.consolidator import Consolidator
  from textmancy.components.extractor import Extractor
  from textmancy.components.processor import Processor

  __all__ = [
      "Annotator",
      "Consolidator",
      "Extractor",
      "Processor",
  ]
  ```

- [ ] **Step 2: Verify imports work**

  ```
  poetry run python -c "from textmancy import Processor, Extractor, Consolidator, Annotator; print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 3: Commit**

  ```bash
  git add textmancy/__init__.py
  git commit -m "feat: expose Processor, Extractor, Consolidator, Annotator at package level"
  ```

---

### Task 3: textmancy/utils.py — fix docstring typo

**Files:**
- Modify: `textmancy/utils.py`

- [ ] **Step 1: Fix the typo**

  In `utils.py` line 8, change:
  ```python
      Given a n iterable of strings, generates a stream that concatenates the strings
  ```
  To:
  ```python
      Given an iterable of strings, generates a stream that concatenates the strings
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add textmancy/utils.py
  git commit -m "fix: typo in text_generator docstring"
  ```

---

### Task 4: textmancy/targets.py — fix docstring typo

**Files:**
- Modify: `textmancy/targets.py`

- [ ] **Step 1: Fix the typo**

  In `targets.py` line 13, change:
  ```python
          description="The reasoning behind the theme occuring in a specific peice of text",
  ```
  To:
  ```python
          description="The reasoning behind the theme occurring in a specific piece of text",
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add textmancy/targets.py
  git commit -m "fix: typo in Theme field description"
  ```

---

### Task 5: textmancy/components/processor.py — full cleanup

**Files:**
- Modify: `textmancy/components/processor.py`

Current state of the file:
```python
import logging

from langchain.pydantic_v1 import BaseModel

from .consolidator import Consolidator
from .extractor import Extractor
from .. import utils


class TextmancyResult(BaseModel):
    """
    A class that represents the result of processing a text corpus.
    """

    targets: list
    annotations: list[int]


class Processor:
    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        model: str = "gpt-4o",
        extractor_args: dict = {},
        consolidator_args: dict = {},
    ):

        self.extractor = Extractor(
            target_class=target_class,
            model=model,
            target_num=int(target_num * 0.6),  # Extractor extracts 60% of the targets
            **extractor_args,
        )
        self.consolidator = Consolidator(
            target_class=target_class,
            model=model,
            target_num=target_num,
            **consolidator_args,
        )
        self._logger = logging.getLogger(__name__)

    def process(self, texts: list[str]) -> TextmancyResult:
        """
        Extracts featres from text, consolidates and then annotates the given text fragments.
        """
        ...
```

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  import logging
  from typing import Any, Dict, Optional

  from langchain.pydantic_v1 import BaseModel

  from .consolidator import Consolidator
  from .extractor import Extractor
  from .. import utils


  class TextmancyResult(BaseModel):
      """
      A class that represents the result of processing a text corpus.
      """

      targets: list
      annotations: list[int]


  class Processor:
      def __init__(
          self,
          target_class: type[BaseModel],
          target_num: int = 3,
          model: str = "gpt-4o",
          extractor_args: Optional[Dict[str, Any]] = None,
          consolidator_args: Optional[Dict[str, Any]] = None,
      ) -> None:
          """
          Initializes the Processor with an extractor and consolidator.

          Args:
              target_class (type[BaseModel]): The Pydantic model class to extract and consolidate.
              target_num (int): The desired number of consolidated targets.
              model (str): The OpenAI model name to use for extraction and consolidation.
              extractor_args (Optional[Dict[str, Any]]): Extra kwargs forwarded to Extractor.
              consolidator_args (Optional[Dict[str, Any]]): Extra kwargs forwarded to Consolidator.
          """
          extractor_args = extractor_args or {}
          consolidator_args = consolidator_args or {}

          self.extractor = Extractor(
              target_class=target_class,
              model=model,
              target_num=int(target_num * 0.6),  # Extractor extracts 60% of the targets
              **extractor_args,
          )
          self.consolidator = Consolidator(
              target_class=target_class,
              model=model,
              target_num=target_num,
              **consolidator_args,
          )
          self._logger = logging.getLogger(__name__)

      def process(self, texts: list[str]) -> TextmancyResult:
          """
          Extracts features from text, consolidates and then annotates the given text fragments.
          """

          # Create text stream for extractor
          self._logger.info("Creating text stream for extraction")
          text_stream = utils.text_generator(texts, max_chunk_size=10000)

          # Extract and consolidate
          self._logger.info("Extracting features")
          results = self.extractor.extract(text_stream)
          self._logger.debug(f"Extracted {len(results)} features")

          self._logger.info("Consolidating features")
          consolidated = self.consolidator.consolidate(results)
          self._logger.debug(f"Consolidated into {len(consolidated)} features")

          return consolidated
  ```

- [ ] **Step 2: Run tests**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add textmancy/components/processor.py
  git commit -m "fix: mutable default args in Processor; add type hints and docstring"
  ```

---

### Task 6: textmancy/components/segmentor.py — fix bare list return types

**Files:**
- Modify: `textmancy/components/segmentor.py`

Only two methods have bare `list` return types. The base class already uses `list[str]` throughout, so match that style.

- [ ] **Step 1: Fix ParagraphSegmentor._segment return type**

  Change line 65:
  ```python
      def _segment(self, text: str) -> list:
  ```
  To:
  ```python
      def _segment(self, text: str) -> list[str]:
  ```

- [ ] **Step 2: Fix PageSegmentor._segment return type**

  Change line 81:
  ```python
      def _segment(self, text: str) -> list:
  ```
  To:
  ```python
      def _segment(self, text: str) -> list[str]:
  ```

- [ ] **Step 3: Run tests**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 4: Commit**

  ```bash
  git add textmancy/components/segmentor.py
  git commit -m "fix: use list[str] return types in ParagraphSegmentor and PageSegmentor"
  ```

---

### Task 7: textmancy/components/extractor.py — type hints and docstrings

**Files:**
- Modify: `textmancy/components/extractor.py`

Current imports: `from typing import Generator, Iterable, List, Optional, Sequence, Union`

Changes needed:
- `__init__`: add `-> None`, fix `target_examples: list = None` → `Optional[List[BaseModel]]`
- `_extract_from_block`: fix `number: int = None` → `Optional[int]`, fix `-> list` → `-> List[BaseModel]`, add `-> None` on `__init__`
- `extract`: add `chunk_size: int` type, fix `-> list` → `-> List[BaseModel]`, add docstring
- `_create_extraction_prompt`: add docstring
- `.dict()` calls: **leave as-is** — these models come from `langchain.pydantic_v1` which uses the Pydantic v1 API

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  import logging
  from typing import Generator, Iterable, List, Optional, Sequence, Union

  from langchain.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain.pydantic_v1 import BaseModel, Field, create_model


  class Extractor:
      """
      A class for extracting information from a given text using a language chain.

      Attributes:
          target_class (type[BaseModel]): The class of the target to extract from the text.
          target_num (int): The expected number of targets to extract from the text.
          target_examples (List[BaseModel]): A list of examples of the target to extract.
      """

      def __init__(
          self,
          target_class: type[BaseModel],
          target_num: int = 3,
          target_examples: Optional[List[BaseModel]] = None,
          model: str = "gpt-4o",
          additional_instructions: str = "",
      ) -> None:
          # Logger
          self._logger = logging.getLogger(__name__)

          # Vars
          self.target_class = target_class
          self.target_num = target_num
          self.target_examples = target_examples or []
          self.target_name = target_class.__name__

          # Grouped class
          fields = {
              f"{self.target_name}s": (
                  Sequence[target_class],
                  Field(..., description=f"A list of {self.target_name}s"),
              ),
          }
          self.grouped_target_type = create_model(f"{self.target_name}s", **fields)
          self.grouped_target_type.__doc__ = f"A list of {self.target_name}s"

          # Create runnable
          prompt = self._create_extraction_prompt(
              target_class, additional_instructions, target_examples
          )
          self.extraction_runnable = prompt | ChatOpenAI(
              model=model
          ).with_structured_output(self.grouped_target_type)

      @classmethod
      def _create_extraction_prompt(
          cls,
          target_class: type[BaseModel],
          additional_instructions: Optional[str] = None,
          examples: Optional[List[BaseModel]] = None,
      ) -> ChatPromptTemplate:
          """
          Builds the extraction prompt template for the given target class.

          Args:
              target_class (type[BaseModel]): The model class to extract.
              additional_instructions (Optional[str]): Extra instructions appended to the prompt.
              examples (Optional[List[BaseModel]]): Example instances shown to the model.

          Returns:
              ChatPromptTemplate: The assembled prompt template.
          """
          prompt = (
              "You are an ethnographer. This is the first stage of classification."
              f"Your job is to determine any {target_class.__name__}s "
              "present in a given text."
              f"{target_class.__name__} is defined as {target_class.__doc__}"
              "Here is the text: \n {text}"
              f"\n Please determine any {target_class.__name__}s present in the text. "
              f"Look for around {{target_num}} {target_class.__name__}s."
              f"\n {additional_instructions}"
          )
          if examples:
              prompt += f"\n ex: {[e.dict() for e in examples]}"

          return ChatPromptTemplate.from_template(prompt)

      def _extract_from_block(self, text: str, number: Optional[int] = None) -> List[BaseModel]:
          """
          Extracts information from the given text using the language chain.

          Args:
              text (str): The text to extract information from.

          Returns:
              List[BaseModel]: A list of extracted target instances.
          """
          result = self.extraction_runnable.invoke(
              {"text": text, "target_num": number or self.target_num}
          )
          return getattr(result, self.target_name + "s")

      def extract(
          self,
          input_data: Union[str, Iterable, Generator],
          chunk_size: int = 4000,
          **kwargs,
      ) -> List[BaseModel]:
          """
          Extracts targets from a string or stream of text.

          Args:
              input_data (Union[str, Iterable, Generator]): Text to extract from. If a string,
                  it is split into chunks. If a stream/generator, chunks are accumulated.
              chunk_size (int): Maximum character size of each chunk sent to the LLM.

          Returns:
              List[BaseModel]: All extracted target instances across all chunks.
          """
          pool = ThreadPoolExecutor(max_workers=10)
          futures = []

          if isinstance(input_data, str):
              # Input is a string, process it in chunks
              for i in range(0, len(input_data), chunk_size):
                  text_chunk = input_data[i : i + chunk_size]
                  futures.append(
                      pool.submit(self._extract_from_block, text_chunk, **kwargs)
                  )
          else:
              # Input is a stream or generator, process it iteratively
              text_chunk = ""
              for piece in input_data:
                  text_chunk += piece
                  if len(text_chunk) >= chunk_size:
                      # Submit the current chunk for processing
                      futures.append(
                          pool.submit(self._extract_from_block, text_chunk, **kwargs)
                      )
                      text_chunk = ""
              # Make sure to process the last chunk if it's not empty
              if text_chunk:
                  futures.append(
                      pool.submit(self._extract_from_block, text_chunk, **kwargs)
                  )

          # Retrieve results as they complete
          results = []
          completed = 0
          for future in as_completed(futures):
              result = future.result()
              results.extend(result)
              completed += 1
              self._logger.debug(f"Finished {completed} of {len(futures)}")

          return results
  ```

- [ ] **Step 2: Run tests**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add textmancy/components/extractor.py
  git commit -m "fix: type hints and docstrings in Extractor"
  ```

---

### Task 8: textmancy/components/annotator.py — type hints, docstring, comment typo

**Files:**
- Modify: `textmancy/components/annotator.py`

Changes needed:
- `__init__`: add `-> None`
- `_create_annotation_prompt`: add docstring (already has return type `ChatPromptTemplate`)
- `_annotate_chunk`: fix `-> list` → `-> List[int]`
- `annotate`: fix `chunk_size=4000` → `chunk_size: int = 4000`
- Line 124 comment: "valdi" → "valid"
- `.dict()` call on line 39: **leave as-is** (Pydantic v1 model from `langchain.pydantic_v1`)
- Add `List` to the existing `from typing import List` import (already imported)

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  import logging
  from typing import List

  from langchain.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain.pydantic_v1 import BaseModel


  class Annotator:
      """
      A class for annotating text samples to identify target indices.

      Attributes:
          targets (List[BaseModel]): The list of target objects to identify in the text.
          model (str): The model to use for annotation.
      """

      def __init__(
          self,
          targets: List[BaseModel],
          model: str = "gpt-4o",
      ) -> None:
          # Logger
          self._logger = logging.getLogger(__name__)

          # Vars
          self.targets = targets
          self.target_class = targets[0].__class__
          self.target_name = self.target_class.__name__
          self.target_desc = self.target_class.__doc__
          self.llm = ChatOpenAI(model=model)

          # Prepare condensed example
          _max_example_size: int = 500
          example_target = targets[0]
          example = {}

          for attr, value in example_target.dict().items():
              if len(str(example)) < _max_example_size:
                  example[attr] = value
              else:
                  break

          # Schema
          self.json_schema = {
              "title": f"{self.target_name}s",
              "description": (
                  f"List of {self.target_name}s indices that are present in the text, 0-based. For example, "  # noqa: E501
                  f"if [{example}] is present in the text, then include 0 in the response."
              ),
              "type": "object",
              "properties": {
                  "indices": {
                      "title": f"{self.target_name}s",
                      "description": f"List of 0-based indices of {self.target_name}s that are present in the text.",  # noqa: E501
                      "type": "array",
                      "items": {
                          "type": "number",
                          "enum": [i for i in range(len(self.targets))],
                      },
                  },
              },
          }

          # Create runnable
          prompt = self._create_annotation_prompt()
          self.annotation_runnable = prompt | ChatOpenAI(
              model=model
          ).with_structured_output(self.json_schema)

      def _create_annotation_prompt(self) -> ChatPromptTemplate:
          """
          Builds the annotation prompt template using the current target list.

          Returns:
              ChatPromptTemplate: The assembled prompt template.
          """
          return ChatPromptTemplate.from_template(
              f"You are tasked with annotating a text sample to identify {self.target_name}s. "
              f"Here is an ordered list of valid targets: \n {self.targets}\n"
              "Here is the text sample: \n {text} \n"
              f"Please return all of the given {self.target_name} indices that are in the text."
          )

      def _annotate_chunk(self, text: str) -> List[int]:
          """
          Annotates a chunk of text to identify target indices.

          Args:
              text (str): The text chunk to annotate.

          Returns:
              List[int]: The list of target indices found in the text chunk.
          """
          result = self.annotation_runnable.invoke({"text": text})
          if not result:
              return []
          return result["indices"]

      def annotate(self, text: str, chunk_size: int = 4000, **kwargs) -> set:
          """
          Annotates the given text to identify target indices.

          Args:
              text (str): The text to annotate.
              chunk_size (int, optional): The size of text chunks for annotation. Defaults to 4000.

          Returns:
              set: A set of unique target indices found in the text.
          """
          pool = ThreadPoolExecutor(max_workers=10)
          futures = []

          # Split text into chunks and extract asynchronously
          for i in range(0, len(text), chunk_size):
              text_chunk = text[i : i + chunk_size]
              futures.append(pool.submit(self._annotate_chunk, text_chunk, **kwargs))

          # Retrieve results as they complete
          results = []
          completed = 0
          for future in as_completed(futures):
              result = future.result()
              results.extend(result)
              completed += 1
              self._logger.debug(f"Finished {completed} of {len(futures)}")

          # Clean results for only valid indices
          return {
              int(i)
              for i in results
              if i is not None and i < len(self.targets) and i >= 0
          }
  ```

- [ ] **Step 2: Run tests**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add textmancy/components/annotator.py
  git commit -m "fix: type hints, docstring, and comment typo in Annotator"
  ```

---

### Task 9: textmancy/components/consolidator.py — type hints and docstring

**Files:**
- Modify: `textmancy/components/consolidator.py`

Current imports: `from typing import List, Optional, Sequence, Type`

Changes needed:
- `__init__`: add `-> None`
- `_create_consolidation_prompt`: add docstring
- `_consolidate_batch`: fix `-> list` → `-> List[BaseModel]`
- `consolidate`: fix `-> list` → `-> List[BaseModel]`

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  from concurrent.futures import ThreadPoolExecutor, as_completed
  import logging
  from typing import List, Optional, Sequence, Type

  from langchain.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  from langchain.pydantic_v1 import BaseModel, Field, create_model


  class Consolidator:
      """
      A class that consolidates a list of target objects into a grouped list.

      Attributes:
          target_class (Type[BaseModel]): The class of the target objects.
          target_num (int): The desired number of target objects in the grouped list.
          batch_size (int): The batch size for processing the target objects.
          max_iter (int): The maximum number of iterations for consolidation.
          tolerance (float): The tolerance level for the number of consolidated targets.
      """

      def __init__(
          self,
          target_class: Type[BaseModel],
          target_num: int = 3,
          model: str = "gpt-4o",
          batch_size: int = 10,
          max_iter: int = 3,
          tolerance: float = 1.5,
          additional_instructions: str = "",
      ) -> None:
          # Logger
          self._logger = logging.getLogger(__name__)

          # Vars
          self.target_class = target_class
          self.target_num = target_num
          self.batch_size = batch_size
          self.max_iter = max_iter
          self.tolerance = tolerance
          self.target_name = target_class.__name__
          self.target_desc = target_class.__doc__
          self.additional_instructions = additional_instructions

          # Grouped class
          fields = {
              f"{self.target_name}s": (
                  Sequence[target_class],
                  Field(..., description=f"A list of {self.target_name}s"),
              ),
          }
          self.grouped_target_type = create_model(f"{self.target_name}s", **fields)
          self.grouped_target_type.__doc__ = f"A list of {self.target_name}s"

          # Create runnable
          prompt = self._create_consolidation_prompt(
              target_class, additional_instructions
          )
          self.consolidation_runnable = prompt | ChatOpenAI(
              model=model
          ).with_structured_output(self.grouped_target_type)

      @classmethod
      def _create_consolidation_prompt(
          cls,
          target_class: Type[BaseModel],
          additional_instructions: Optional[str] = None,
      ) -> ChatPromptTemplate:
          """
          Builds the consolidation prompt template for the given target class.

          Args:
              target_class (Type[BaseModel]): The model class being consolidated.
              additional_instructions (Optional[str]): Extra instructions appended to the prompt.

          Returns:
              ChatPromptTemplate: The assembled prompt template.
          """
          prompt = (
              "You are an ethnographer. In this stage, text has been freely classified."
              "You are now consolidating various annotations to create a final list."
              f"Please consolidate the following {target_class.__name__}s into a list: \n"
              + "{targets}"
              + f"\n Consolidate into {target_class.__name__}s."
              + f"Where possible, combine similar {target_class.__name__}s into one."
              + f"Look for cases where the same {target_class.__name__} is referred to by "
              + "different names. Avoid repetition and redundancy."
              + f"\n {additional_instructions}"
          )
          return ChatPromptTemplate.from_template(prompt)

      def _consolidate_batch(self, targets: List[BaseModel]) -> List[BaseModel]:
          """
          Consolidates a batch of target objects using the consolidation chain.

          Args:
              targets (List[BaseModel]): The batch of target objects to be consolidated.

          Returns:
              List[BaseModel]: The consolidated list of target objects.
          """
          result = self.consolidation_runnable.invoke({"targets": targets})
          return getattr(result, self.target_name + "s")

      def consolidate(self, items: List[BaseModel], current_iter: int = 0) -> List[BaseModel]:
          """
          Consolidates a list of target objects into a grouped list.

          Args:
              items (List[BaseModel]): The list of target objects to be consolidated.
              current_iter (int): The current iteration count for consolidation.

          Returns:
              List[BaseModel]: The consolidated grouped list of target objects.
          """
          batches = [
              items[i : i + self.batch_size]
              for i in range(0, len(items), self.batch_size)
          ]

          with ThreadPoolExecutor(max_workers=10) as executor:
              futures = [
                  executor.submit(self._consolidate_batch, batch) for batch in batches
              ]
              results = []
              for future in as_completed(futures):
                  result = future.result()
                  results.extend(result)

          if (
              len(results) >= self.target_num * self.tolerance
              and current_iter < self.max_iter
          ):
              return self.consolidate(results, current_iter + 1)

          return results
  ```

- [ ] **Step 2: Run tests**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add textmancy/components/consolidator.py
  git commit -m "fix: type hints and docstring in Consolidator"
  ```

---

### Task 10: Phase 2 — run pre-commit and final verification

**Files:** none new

- [ ] **Step 1: Run pre-commit across all files**

  ```
  poetry run pre-commit run --all-files
  ```

  Expected: all hooks pass. If Black or isort reformats anything, stage the changes and re-run until clean.

- [ ] **Step 2: Run tests one final time**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit any pre-commit auto-fixes if present**

  ```bash
  git add -u
  git commit -m "style: apply black and isort formatting to source files"
  ```

  (Skip this commit if pre-commit made no changes.)

---

## Phase 3 — Tests & Example Cleanup

### Task 11: test/test_segmentor.py — add return type hints

**Files:**
- Modify: `test/test_segmentor.py`

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  from typing import List

  from textmancy.components.segmentor import Segmentor, ParagraphSegmentor, PageSegmentor

  import pytest


  class SampleSegmentor(Segmentor):
      def _segment(self, text: str) -> List[str]:
          return [p for p in text.split(" ") if p]


  @pytest.mark.parametrize("max_length, handle_length", [
      (10, "split"),
      (20, "truncate"),
      (30, "raise")
  ])
  def test_segmentor_init(max_length, handle_length) -> None:
      segmentor = SampleSegmentor(max_length=max_length, handle_length=handle_length)
      assert segmentor._max_length == max_length
      assert segmentor._handle_length == handle_length


  def test_segmentor_init_validation() -> None:
      with pytest.raises(ValueError):
          SampleSegmentor(max_length=10, handle_length="invalid")


  def test_segmentor_handle_max_length() -> None:
      text = "This is a test sentence"

      segmentor = SampleSegmentor(max_length=5, handle_length="split")
      expected_output = ["This", "is", "a", "test", "sente", "nce"]
      assert segmentor.segment(text) == expected_output

      segmentor = SampleSegmentor(max_length=5, handle_length="truncate")
      expected_output = ["This", "is", "a", "test", "sente"]
      assert segmentor.segment(text) == expected_output

      segmentor = SampleSegmentor(max_length=5, handle_length="raise")
      with pytest.raises(ValueError):
          segmentor.segment(text)


  def test_paragraph_segmentor_empty_text() -> None:
      segmentor = ParagraphSegmentor()
      text = ""
      expected_output = []
      assert segmentor.segment(text) == expected_output


  def test_paragraph_segmentor_single_paragraph() -> None:
      segmentor = ParagraphSegmentor()
      text = "This is a single paragraph."
      expected_output = ["This is a single paragraph."]
      assert segmentor.segment(text) == expected_output


  def test_paragraph_segmentor_multiple_paragraphs() -> None:
      segmentor = ParagraphSegmentor()
      text = "This is the first paragraph.\n\nThis is the second paragraph."
      expected_output = ["This is the first paragraph.", "This is the second paragraph."]
      assert segmentor.segment(text) == expected_output


  def test_page_segmentor_empty_text() -> None:
      segmentor = PageSegmentor()
      text = ""
      expected_output = []
      assert segmentor.segment(text) == expected_output


  def test_page_segmentor_single_page() -> None:
      segmentor = PageSegmentor()
      text = "This is a single page."
      expected_output = ["This is a single page."]
      assert segmentor.segment(text) == expected_output


  def test_page_segmentor_multiple_pages() -> None:
      segmentor = PageSegmentor(paragraphs_per_page=2)
      text = (
              "This is the first para.\nThis is the second para.\n"
              "This is the third para.\nThis is the fourth para.\n"
      )
      expected_output = [
          "This is the first para.\nThis is the second para.",
          "This is the third para.\nThis is the fourth para."
      ]
      assert segmentor.segment(text) == expected_output
  ```

- [ ] **Step 2: Run tests**

  ```
  poetry run pytest test/test_segmentor.py -v
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add test/test_segmentor.py
  git commit -m "fix: add return type hints to test_segmentor.py"
  ```

---

### Task 12: test/test_utils.py — add return type hints

**Files:**
- Modify: `test/test_utils.py`

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  from textmancy.utils import text_generator


  def test_text_generator() -> None:
      texts = ["Lorem", "ipsum", "dolor", "sit", "amet"]
      max_chunk_size = 5
      expected_output = ["Lorem", "ipsum", "dolor", "sit", "amet"]

      generator = text_generator(texts, max_chunk_size)
      output = list(generator)

      assert output == expected_output


  def test_text_generator_with_large_text() -> None:
      texts = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit."]
      max_chunk_size = 10
      expected_output = [
          "Lorem ipsu",
          "m dolor si",
          "t amet, co",
          "nsectetur ",
          "adipiscing",
          " elit.",
      ]

      generator = text_generator(texts, max_chunk_size)
      output = list(generator)

      assert output == expected_output


  def test_text_generator_with_empty_input() -> None:
      texts = []
      max_chunk_size = 5
      expected_output = []

      generator = text_generator(texts, max_chunk_size)
      output = list(generator)

      assert output == expected_output
  ```

- [ ] **Step 2: Run tests**

  ```
  poetry run pytest test/test_utils.py -v
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add test/test_utils.py
  git commit -m "fix: add return type hints to test_utils.py"
  ```

---

### Task 13: example.py — module docstring and variable name typo

**Files:**
- Modify: `example.py`

Changes:
- Add a module-level docstring at the top
- Rename `annotater` → `annotator` on lines 31 and 35

- [ ] **Step 1: Replace the file with the cleaned version**

  ```python
  """
  Example script demonstrating the full textmancy pipeline:
  extracts characters from a text file, consolidates them,
  and annotates pages with the characters found on each page.
  """
  import dotenv
  import logging

  from textmancy.components import Processor, Annotator, PageSegmentor
  from textmancy.targets import Character

  dotenv.load_dotenv()

  # Print logs to stream
  logging.getLogger("textmancy").setLevel(logging.INFO)
  logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


  # Get text and split into pages
  segmentor = PageSegmentor(paragraphs_per_page=10)
  with open("sample_data/snows_of_kiliminjaro.txt", encoding="utf-8") as f:
      text = f.read()
      pages = segmentor.segment(text)
      # pages = pages[:5]  # Only process first 5 pages for speed

  # Extract and consolidate characters
  processor = Processor(target_class=Character)
  results = processor.process(pages)

  for r in results:
      print(r)
      print()


  # Annotate pages with characters
  annotator = Annotator(
      targets=results,
      model="gpt-4o",
  )
  annotations = [annotator.annotate(page) for page in pages[:5]]

  # Show annotations
  for page, annotation in zip(pages, annotations):
      print(page)
      print(annotation)
      print([results[i].name for i in annotation])
      print()
  ```

- [ ] **Step 2: Commit**

  ```bash
  git add example.py
  git commit -m "fix: module docstring and rename annotater -> annotator in example.py"
  ```

---

### Task 14: Phase 3 — run pre-commit and final verification

- [ ] **Step 1: Run pre-commit across all files**

  ```
  poetry run pre-commit run --all-files
  ```

  Expected: all hooks pass. Stage and commit any auto-fixes.

- [ ] **Step 2: Run full test suite**

  ```
  poetry run pytest
  ```

  Expected: all tests pass.

- [ ] **Step 3: Commit any pre-commit auto-fixes if present**

  ```bash
  git add -u
  git commit -m "style: apply black and isort formatting to test and example files"
  ```

  (Skip if pre-commit made no changes.)
