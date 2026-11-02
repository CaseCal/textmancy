import asyncio
import json
from pathlib import Path

import pytest

from textmancy.components.consolidator import Consolidator
from textmancy.targets import Character, Theme


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


def test_consolidate_stops_when_pass_does_not_shrink(recording_model):
    rec = recording_model(_echo)
    consolidator = Consolidator(Theme, target_num=3, batch_size=10, max_iter=2, model=rec.model)

    result = consolidator.consolidate(_themes(25))

    assert len(rec.calls) == 3
    assert len(result) == 25


def test_empty_batch_keeps_inputs_and_warns(recording_model, caplog):
    rec = recording_model(lambda system, user, i: [] if i == 0 else _batch_of(user))
    consolidator = Consolidator(Theme, target_num=1, batch_size=2, model=rec.model)

    result = consolidator.consolidate(_themes(4))

    assert [item.name for item in result] == ["t0", "t1", "t2", "t3"]
    assert len(rec.calls) == 2
    assert "empty" in caplog.text.lower()


def test_warns_when_max_passes_exhausted_above_target(recording_model, caplog):
    rec = recording_model(lambda system, user, i: _batch_of(user)[:-1])
    consolidator = Consolidator(Theme, target_num=1, batch_size=10, max_iter=0, model=rec.model)

    result = consolidator.consolidate(_themes(4))

    assert len(result) == 3
    assert "max" in caplog.text.lower()


def test_shrinking_passes_continue_until_limit(recording_model, caplog):
    rec = recording_model(lambda system, user, i: _batch_of(user)[:-1])
    consolidator = Consolidator(Theme, target_num=1, batch_size=10, max_iter=1, model=rec.model)

    result = consolidator.consolidate(_themes(5))

    assert len(result) == 3
    assert len(rec.calls) == 2
    assert "max_iter" in caplog.text


def test_large_match_group_respects_batch_size(recording_model):
    rec = recording_model(_echo)
    consolidator = Consolidator(Theme, target_num=1, batch_size=2, max_iter=0, model=rec.model)
    items = [Theme(name="same", reasoning=str(i)) for i in range(5)]

    result = consolidator.consolidate(items)

    assert len(result) == 5
    assert [len(_batch_of(call["user"])) for call in rec.calls] == [2, 2, 1]


def test_aliases_in_different_batches_are_grouped(recording_model):
    def character(name, aliases):
        return Character(
            name=name,
            known_as=aliases,
            description="",
            physical_description="",
            summary_of_actions="",
        )

    rec = recording_model(_echo)
    consolidator = Consolidator(Character, target_num=1, batch_size=2, max_iter=0, model=rec.model)
    items = [
        character("Ebenezer Scrooge", ["Scrooge"]),
        character("Bob", []),
        character("Alice", []),
        character("Scrooge", []),
    ]

    consolidator.consolidate(items)

    batches = [[item["name"] for item in _batch_of(call["user"])] for call in rec.calls]
    assert ["Ebenezer Scrooge", "Scrooge"] in batches


def test_sample_character_aliases_group_without_dropping_targets(recording_model):
    data = json.loads(Path("benchmark/data/christmas_carol.json").read_text(encoding="utf-8"))
    characters = [Character.model_validate(value) for value in data["targets"][:4]]
    alias = characters[0].model_copy(update={"name": "Scrooge", "known_as": []})
    rec = recording_model(_echo)
    consolidator = Consolidator(Character, target_num=1, batch_size=2, max_iter=0, model=rec.model)

    result = consolidator.consolidate([*characters, alias])

    batches = [[item["name"] for item in _batch_of(call["user"])] for call in rec.calls]
    assert ["Ebenezer Scrooge", "Scrooge"] in batches
    assert sorted(item.name for item in result) == sorted(
        item.name for item in [*characters, alias]
    )


def test_empty_input_and_invalid_batch_size(recording_model):
    rec = recording_model(_echo)
    assert Consolidator(Theme, model=rec.model).consolidate([]) == []
    assert not rec.calls
    with pytest.raises(ValueError, match="batch_size"):
        Consolidator(Theme, batch_size=0, model=rec.model).consolidate(_themes(1))


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


def test_consolidate_model_string_reuses_one_connection_per_call(openai_stub):
    openai_stub.response = [{"name": "a", "reasoning": "r"}]
    consolidator = Consolidator(
        Theme, target_num=10, batch_size=1, model="openai-chat:gpt-4o", max_concurrency=1
    )

    assert len(consolidator.consolidate(_themes(3))) == 3
    assert len(openai_stub.ports) == 3
    assert len(set(openai_stub.ports)) == 1
