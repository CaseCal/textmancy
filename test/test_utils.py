import asyncio
import gc
import time

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from textmancy.components import LLMAnnotator
from textmancy.targets import Theme
from textmancy.utils import chunk_input, gather_bounded, run_sync, text_generator


def test_text_generator():
    texts = ["Lorem", "ipsum", "dolor", "sit", "amet"]
    max_chunk_size = 5
    expected_output = ["Lorem", "ipsum", "dolor", "sit", "amet"]

    generator = text_generator(texts, max_chunk_size)
    output = list(generator)

    assert output == expected_output


def test_text_generator_with_large_text():
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


def test_text_generator_with_empty_input():
    texts = []
    max_chunk_size = 5
    expected_output = []

    generator = text_generator(texts, max_chunk_size)
    output = list(generator)

    assert output == expected_output


def test_chunk_input_slices_strings():
    assert chunk_input("aaaabbbbcc", 4) == ["aaaa", "bbbb", "cc"]


def test_chunk_input_empty_string():
    assert chunk_input("", 4) == []


def test_chunk_input_accumulates_iterables_and_flushes_remainder():
    assert chunk_input(iter(["ab", "cd", "ef", "g"]), 4) == ["abcd", "efg"]


async def _value(value):
    return value


def test_run_sync_returns_result():
    assert run_sync(_value(42)) == 42


def test_run_sync_cancels_work_left_by_an_interrupted_call():
    state = {"finished": 0}

    async def slow():
        await asyncio.sleep(0.05)
        state["finished"] += 1

    def interrupt():
        raise KeyboardInterrupt

    async def main():
        # Raised from a loop callback, the way Ctrl-C reaches run_until_complete.
        asyncio.get_running_loop().call_later(0.01, interrupt)
        await gather_bounded([slow(), slow()], limit=2)

    with pytest.raises(KeyboardInterrupt):
        run_sync(main())
    run_sync(asyncio.sleep(0.1))

    assert state["finished"] == 0


def test_run_sync_inside_running_loop_raises():
    async def outer():
        with pytest.raises(RuntimeError, match="_async"):
            run_sync(_value(1))

    asyncio.run(outer())


def test_gather_bounded_respects_limit_and_preserves_order():
    state = {"active": 0, "peak": 0}

    async def work(i):
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0.01 * (5 - i))
        state["active"] -= 1
        return i

    result = asyncio.run(gather_bounded([work(i) for i in range(5)], limit=2))

    assert result == [0, 1, 2, 3, 4]
    assert state["peak"] == 2


def test_gather_bounded_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(gather_bounded([], limit=0))


def test_gather_bounded_cancels_siblings_on_failure():
    state = {"finished": 0, "cancelled": 0}

    async def boom():
        await asyncio.sleep(0.01)
        raise ValueError("boom")

    async def slow():
        try:
            await asyncio.sleep(5)
            state["finished"] += 1
        except asyncio.CancelledError:
            state["cancelled"] += 1
            raise

    async def main():
        with pytest.raises(ValueError, match="boom"):
            await gather_bounded([boom(), slow(), slow()], limit=3)
        # Siblings must already be cancelled when gather_bounded returns, not at loop shutdown.
        assert state == {"finished": 0, "cancelled": 2}

    start = time.monotonic()
    asyncio.run(main())

    assert time.monotonic() - start < 1


def test_gather_bounded_closes_unstarted_coroutines_on_failure(recwarn):
    async def boom():
        await asyncio.sleep(0)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(gather_bounded([boom(), *(_value(i) for i in range(3))], limit=1))
    gc.collect()

    assert not [w for w in recwarn if "never awaited" in str(w.message)]


def test_repeated_sync_calls_reuse_model_instance(openai_stub):
    model = OpenAIChatModel(
        "gpt-4o", provider=OpenAIProvider(base_url=openai_stub.url, api_key="test")
    )
    annotator = LLMAnnotator([Theme(name="a", reasoning="r")], model=model)

    for _ in range(3):
        assert annotator.annotate("x" * 30, chunk_size=10) == {0}
