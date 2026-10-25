import logging
from abc import ABC, abstractmethod
from typing import Sequence

from pydantic import BaseModel

from ... import utils


class Annotator(ABC):
    """
    Base class for mapping a fixed list of targets back onto text.

    Subclasses implement _annotate_chunk_async. They may extend annotate/annotate_async with
    extra keyword arguments, but must keep the base signature compatible.

    Attributes:
        targets (list[BaseModel]): The targets to identify; results are indices into this list.
        max_concurrency (int): The maximum number of chunks annotated concurrently.
    """

    def __init__(self, targets: Sequence[BaseModel], max_concurrency: int = 10):
        if not targets:
            raise ValueError("targets must not be empty")

        self._logger = logging.getLogger(__name__)

        self.targets = list(targets)
        self.target_class = type(self.targets[0])
        self.target_name = self.target_class.__name__
        self.max_concurrency = max_concurrency

    @abstractmethod
    async def _annotate_chunk_async(self, chunk: str) -> set[int]:
        """Returns the indices of targets present in a single chunk."""

    async def annotate_async(self, text: str, chunk_size: int = 4000) -> set[int]:
        """
        Identifies which targets appear in the text.

        Args:
            text: The text to annotate.
            chunk_size: The chunk size in characters.

        Returns:
            The indices of targets found in any chunk.
        """
        chunks = utils.chunk_input(text, chunk_size)
        results = await utils.gather_bounded(
            (self._annotate_chunk_async(chunk) for chunk in chunks),
            self.max_concurrency,
        )
        self._logger.debug(f"Annotated {len(chunks)} chunks")
        return set().union(*results)

    def annotate(self, text: str, chunk_size: int = 4000) -> set[int]:
        """Synchronous wrapper for annotate_async."""
        return utils.run_sync(self.annotate_async(text, chunk_size))
