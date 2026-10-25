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


def test_processor_asks_extractor_for_at_least_one_target(recording_model):
    rec = recording_model(_respond)
    processor = Processor(Theme, target_num=1, model=rec.model)

    processor.process(["some text"])

    assert "around 1 Themes" in rec.calls[0]["user"]


def test_processor_constructs_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    processor = Processor(Theme)

    assert processor.extractor.agent is not None


def test_processor_async(recording_model):
    rec = recording_model(_respond)
    processor = Processor(Theme, model=rec.model)

    result = asyncio.run(processor.process_async(["some text"]))

    assert [t.name for t in result] == ["merged"]
