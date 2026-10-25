# textmancy

LLM-driven package for mapping, analyzing, and expanding text works. Extract features such as characters or themes from a text, consolidate duplicates, and annotate which features appear on each page.

## Install

```bash
pip install textmancy              # OpenAI support
pip install "textmancy[anthropic]" # + Anthropic
pip install "textmancy[google]"    # + Google
pip install "textmancy[jev]"       # + JevAnnotator (TypeSafe)
```

Set the API key for your provider (e.g. `OPENAI_API_KEY`, or `TYPESAFE_API_KEY` for Jev).

## Usage

```python
from textmancy.components import LLMAnnotator, PageSegmentor, Processor
from textmancy.targets import Character

pages = PageSegmentor(paragraphs_per_page=10).segment(text)

characters = Processor(target_class=Character, model="openai-chat:gpt-4o").process(pages)

annotator = LLMAnnotator(targets=characters, model="anthropic:claude-sonnet-5-5")
for page in pages:
    print([characters[i].name for i in annotator.annotate(page)])
```

### Jev annotation

`JevAnnotator` uses TypeSafe's [Jev](https://typesafe.ai/) classifier instead of an LLM. It asks one
yes/no question per target and returns a probability for each, which is faster and cheaper than an
LLM call:

```python
from textmancy.components import JevAnnotator

annotator = JevAnnotator(targets=characters, threshold=0.5, aggregate="max")
scores = annotator.annotate_scores(page)          # {target index: probability}
present = annotator.annotate(page, threshold=0.4)  # {target index, ...}
```

Long text is chunked (`chunk_size`, default 4000 characters) and each target's per-chunk
probabilities are combined by `aggregate`:

- `"max"` (default): present if any chunk is confident.
- `"noisy_or"`: `1 - prod(1 - p)`, so several moderate detections add up. Probabilities below
  `min_prob` (default 0.2) are ignored so background noise does not compound.
- `"mean"`: how pervasive the target is rather than whether it appears.
- Any callable taking a list of probabilities.

Every component also has an async variant (`process_async`, `annotate_async`, …) for use inside an event loop.

## Migrating from 0.3

- Model names now use Pydantic AI's `provider:model` form: `model="gpt-4o"` becomes `model="openai-chat:gpt-4o"` (the default). `openai:gpt-4o` selects OpenAI's Responses API instead, which stores responses on OpenAI's side unless `openai_store=False` is set.
- `Annotator` is now an abstract base class; use `LLMAnnotator` (same `targets`/`model` arguments).
- Every component has an `*_async` variant; the sync methods cannot be called from inside a running event loop.
- A component instance must not be called from several threads at once; create one per thread, or use the `*_async` methods with `max_concurrency` for parallelism.
- `LLMAnnotator` only accepts valid target indices from the model and retries otherwise (previously invalid indices were silently dropped).
