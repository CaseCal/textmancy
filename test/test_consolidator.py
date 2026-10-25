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
