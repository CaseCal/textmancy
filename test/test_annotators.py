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
