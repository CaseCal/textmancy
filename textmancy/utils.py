import asyncio
from typing import Any, Awaitable, Coroutine, Generator, Iterable, TypeVar, Union

T = TypeVar("T")


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


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Runs a coroutine to completion from synchronous code.

    Raises:
        RuntimeError: If called while an event loop is already running (e.g. Jupyter). Await the
            matching *_async method instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(
        "Cannot run synchronously inside a running event loop; await the *_async method instead."
    )


async def gather_bounded(coros: Iterable[Awaitable[T]], limit: int) -> list[T]:
    """
    Awaits coroutines concurrently with at most `limit` in flight, returning results in input
    order.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    semaphore = asyncio.Semaphore(limit)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return list(await asyncio.gather(*(_bounded(c) for c in coros)))
