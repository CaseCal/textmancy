from typing import Generator, Iterable


def text_generator(
    texts: Iterable[str], max_chunk_size: int
) -> Generator[str, None, None]:
    """
    Given a n iterable of strings, generates a stream that concatenates the strings into chunks of a given size.

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
