import pytest
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIChatModel

from textmancy.components import Consolidator, Extractor, LLMAnnotator, Processor
from textmancy.targets import Theme


@pytest.mark.parametrize(
    "make_agent",
    [
        lambda: Extractor(Theme).agent,
        lambda: Consolidator(Theme).agent,
        lambda: LLMAnnotator([Theme(name="a", reasoning="r")]).agent,
        lambda: Processor(Theme).extractor.agent,
        lambda: Processor(Theme).consolidator.agent,
    ],
    ids=["extractor", "consolidator", "annotator", "processor-extractor", "processor-consolidator"],
)
def test_default_model_uses_chat_completions(monkeypatch, make_agent):
    # The Responses API stores requests on OpenAI's side by default; Chat Completions does not.
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    assert isinstance(infer_model(make_agent().model), OpenAIChatModel)
