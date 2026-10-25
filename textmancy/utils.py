import asyncio
import contextlib
import inspect
import threading
from typing import Any, Awaitable, Coroutine, Generator, Iterable, TypeVar, Union

T = TypeVar("T")

_local = threading.local()


def text_generator(texts: Iterable[str], max_chunk_size: int) -> Generator[str, None, None]:
    """
    Given a n iterable of strings, generates a stream that concatenates the strings
    into chunks of a given size.

    """
    next_chunk = ""
    for text in texts:
        # Split large text
        if len(text) > max_chunk_size:
            for i in range(0, len(text), max_chunk_size):
                yield text[i : i + max_chunk_size]

        # Return when chunk is full
        elif len(next_chunk) + len(text) > max_chunk_size:
            if next_chunk:
                yield next_chunk
            next_chunk = text
        else:
            next_chunk += text
    if next_chunk:
        yield next_chunk


def chunk_input(input_data: Union[str, Iterable[str]], chunk_size: int) -> list[str]:
    """
    Splits input into chunks for LLM calls.

    A string is sliced into pieces of exactly chunk_size (the last may be shorter). Any other
    iterable of strings is accumulated until a chunk reaches at least chunk_size, and the
    remainder is flushed at the end.
    """
    if isinstance(input_data, str):
        return [input_data[i : i + chunk_size] for i in range(0, len(input_data), chunk_size)]

    chunks = []
    current = ""
    for piece in input_data:
        current += piece
        if len(current) >= chunk_size:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _thread_loop() -> asyncio.AbstractEventLoop:
    """Returns this thread's persistent event loop, creating it if missing or closed."""
    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _local.loop = loop
    asyncio.set_event_loop(loop)
    return loop


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Runs a coroutine to completion from synchronous code.

    The coroutine runs on a persistent per-thread event loop that is reused across calls, so
    objects bound to a loop (such as a user-supplied Pydantic AI Model's HTTP client) keep
    working on repeated calls. If the call is interrupted (e.g. by Ctrl-C), its task is cancelled
    and cleaned up before the exception propagates, so its work doesn't resume on the next call.
    This mirrors Pydantic AI's own Agent.run_sync.

    Raises:
        RuntimeError: If called while an event loop is already running (e.g. Jupyter). Await the
            matching *_async method instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        raise RuntimeError(
            "Cannot run synchronously inside a running event loop; "
            "await the *_async method instead."
        )

    loop = _thread_loop()
    task = loop.create_task(coro)
    try:
        return loop.run_until_complete(task)
    except BaseException:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                loop.run_until_complete(task)
        raise


async def gather_bounded(coros: Iterable[Awaitable[T]], limit: int) -> list[T]:
    """
    Awaits coroutines concurrently with at most `limit` in flight, returning results in input
    order. If any coroutine raises, the rest are cancelled and the exception is re-raised.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    semaphore = asyncio.Semaphore(limit)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    coros = list(coros)
    tasks = [asyncio.ensure_future(_bounded(c)) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Coroutines cancelled while waiting for a slot never started; close them so Python
        # doesn't warn that they were never awaited. Closing a finished coroutine is a no-op.
        for coro in coros:
            if inspect.iscoroutine(coro):
                coro.close()
        raise
