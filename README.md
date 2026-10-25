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
