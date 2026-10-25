import logging
from typing import Optional, Union

from pydantic import BaseModel
from pydantic_ai.models import Model

from .. import utils
from .consolidator import Consolidator
from .extractor import Extractor


class Processor:
    """
    Extracts targets from a corpus of texts and consolidates them into a final list.

    Attributes:
        extractor (Extractor): Extracts raw targets; asked for 60% of target_num per chunk.
        consolidator (Consolidator): Merges raw targets down toward target_num.
    """

    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        model: Union[str, Model] = "openai-chat:gpt-4o",
        extractor_args: Optional[dict] = None,
        consolidator_args: Optional[dict] = None,
    ):
        self._logger = logging.getLogger(__name__)

        self.extractor = Extractor(
            target_class=target_class,
            model=model,
            target_num=max(1, int(target_num * 0.6)),  # Extractor extracts 60% of the targets
            **(extractor_args or {}),
        )
        self.consolidator = Consolidator(
            target_class=target_class,
            model=model,
            target_num=target_num,
            **(consolidator_args or {}),
        )

    async def process_async(self, texts: list[str]) -> list[BaseModel]:
        """
        Extracts features from the texts, then consolidates them.

        Args:
            texts: The text fragments (e.g. pages) to process.

        Returns:
            The consolidated targets.
        """
        self._logger.info("Creating text stream for extraction")
        text_stream = utils.text_generator(texts, max_chunk_size=10000)

        self._logger.info("Extracting features")
        results = await self.extractor.extract_async(text_stream)
        self._logger.debug(f"Extracted {len(results)} features")

        self._logger.info("Consolidating features")
        consolidated = await self.consolidator.consolidate_async(results)
        self._logger.debug(f"Consolidated into {len(consolidated)} features")

        return consolidated

    def process(self, texts: list[str]) -> list[BaseModel]:
        """Synchronous wrapper for process_async."""
        return utils.run_sync(self.process_async(texts))
