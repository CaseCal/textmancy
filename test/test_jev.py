import builtins
import os
from types import SimpleNamespace

import pytest

from textmancy.components import JevAnnotator
from textmancy.targets import Character, Theme

THEMES = [
    Theme(name="alpha", reasoning="r"),
    Theme(name="beta", reasoning="r"),
    Theme(name="gamma", reasoning="r"),
]


class FakeClient:
    """
    Records system_one calls and answers each question with respond(state, question_name).
    """

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    async def system_one(self, state, questions, *, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        nouls = {name: SimpleNamespace(noul=self.respond(state, name)) for name in questions}
        return SimpleNamespace(nouls=nouls, usage=SimpleNamespace(input_tokens=10, output_tokens=1))


def by_name(table):
    """Responds from {question_name: probability}, defaulting to 0."""
    return lambda state, name: table.get(name, 0.0)


def test_one_noul_question_per_target():
    annotator = JevAnnotator(THEMES, client=FakeClient(by_name({})))

    assert list(annotator.questions) == ["target_0", "target_1", "target_2"]
    question = annotator.questions["target_1"]
    assert question.type == "noul"
    assert question.instructions == {
        "question": "Does the text include or refer to this Theme?",
        "Theme": {"name": "beta", "reasoning": "r"},
    }
    assert question.criteria == {
        "true": "This Theme appears or is referenced, including by alias",
        "false": "This Theme does not appear",
    }


def test_questions_carry_full_target_dump():
    character = Character(
        name="Ann",
        description="d",
        physical_description="p",
        known_as=["A"],
        summary_of_actions="s",
    )
    annotator = JevAnnotator([character], client=FakeClient(by_name({})))

    assert annotator.questions["target_0"].instructions["Character"] == character.model_dump()


def test_scores_every_target_with_chunk_as_state():
    client = FakeClient(by_name({"target_0": 0.9, "target_2": 0.3}))
    annotator = JevAnnotator(THEMES, model="jev-test", client=client)

    scores = annotator.annotate_scores("some text")

    assert scores == {0: pytest.approx(0.9), 1: 0.0, 2: pytest.approx(0.3)}
    assert client.calls[0]["state"] == "some text"
    assert client.calls[0]["model"] == "jev-test"


def test_max_questions_per_request_splits_and_merges():
    client = FakeClient(by_name({"target_0": 0.9, "target_1": 0.8, "target_2": 0.7}))
    annotator = JevAnnotator(THEMES, max_questions_per_request=2, client=client)

    scores = annotator.annotate_scores("text")

    assert [list(call["questions"]) for call in client.calls] == [
        ["target_0", "target_1"],
        ["target_2"],
    ]
    assert scores == pytest.approx({0: 0.9, 1: 0.8, 2: 0.7})


def test_rejects_non_positive_max_questions():
    with pytest.raises(ValueError, match="max_questions_per_request"):
        JevAnnotator(THEMES, max_questions_per_request=0, client=FakeClient(by_name({})))


def chunk_client():
    # Chunk "aaaa" is sure about alpha; chunks "bbbb" and "cccc" are each moderately sure.
    table = {
        "aaaa": {"target_0": 0.9},
        "bbbb": {"target_1": 0.45},
        "cccc": {"target_1": 0.45},
    }
    return FakeClient(lambda state, name: table[state].get(name, 0.0))


def test_aggregates_across_chunks_with_default_max():
    annotator = JevAnnotator(THEMES, client=chunk_client())

    scores = annotator.annotate_scores("aaaabbbbcccc", chunk_size=4)

    assert scores == pytest.approx({0: 0.9, 1: 0.45, 2: 0.0})


@pytest.mark.parametrize(
    "aggregate, expected",
    [
        ("mean", {0: 0.3, 1: 0.3, 2: 0.0}),
        ("noisy_or", {0: 0.9, 1: 0.6975, 2: 0.0}),
        (lambda ps: min(ps), {0: 0.0, 1: 0.0, 2: 0.0}),
    ],
)
def test_aggregate_option(aggregate, expected):
    annotator = JevAnnotator(THEMES, aggregate=aggregate, client=chunk_client())

    scores = annotator.annotate_scores("aaaabbbbcccc", chunk_size=4)

    assert scores == pytest.approx(expected)


def test_threshold_default_and_per_call_override():
    client = FakeClient(by_name({"target_0": 0.9, "target_1": 0.45, "target_2": 0.2}))
    default = JevAnnotator(THEMES, client=client)
    strict = JevAnnotator(THEMES, threshold=0.95, client=client)

    assert default.annotate("text") == {0}
    assert strict.annotate("text") == set()
    assert default.annotate("text", threshold=0.4) == {0, 1}


def test_annotate_agrees_with_scores():
    annotator = JevAnnotator(THEMES, aggregate="noisy_or", threshold=0.6, client=chunk_client())

    scores = annotator.annotate_scores("aaaabbbbcccc", chunk_size=4)
    found = annotator.annotate("aaaabbbbcccc", chunk_size=4)

    assert found == {i for i, p in scores.items() if p >= 0.6} == {0, 1}


def test_missing_sdk_raises_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "typesafe_sdk":
            raise ImportError("No module named 'typesafe_sdk'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install textmancy\[jev\]"):
        JevAnnotator(THEMES)


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
def test_live_smoke():
    characters = [
        Character(
            name="Ebenezer Scrooge",
            description="A miserly old businessman.",
            physical_description="Old and thin.",
            known_as=["Scrooge"],
            summary_of_actions="Refuses to celebrate Christmas.",
        ),
        Character(
            name="Captain Ahab",
            description="A whaling captain obsessed with a white whale.",
            physical_description="Has a whalebone leg.",
            known_as=["Ahab"],
            summary_of_actions="Hunts Moby Dick.",
        ),
    ]
    annotator = JevAnnotator(characters)

    scores = annotator.annotate_scores("Scrooge sat alone in his counting-house, counting coins.")

    assert scores[0] > scores[1]
    assert annotator.annotate("Scrooge said 'Bah! Humbug!'") >= {0}
