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
