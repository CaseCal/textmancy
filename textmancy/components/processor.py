import logging

from langchain.pydantic_v1 import BaseModel

from .consolidator import Consolidator
from .extractor import Extractor
from .. import utils


class TextmancyResult(BaseModel):
    """
    A class that represents the result of processing a text corpus.
    """

    targets: list
    annotations: list[int]


class Processor:
    def __init__(
        self,
        target_class: type[BaseModel],
        model: str = "gpt-4-1106-preview",
        target_num_hint: int = 10,
    ):
        self.extractor = Extractor(
            target_class=target_class, model=model, target_num=target_num_hint
        )
        self.consolidator = Consolidator(target_class=target_class, model=model)
        self._logger = logging.getLogger(__name__)

    def process(self, texts: list[str]) -> TextmancyResult:
        """
        Extracts featres from text, consolidates and then annotates the given text fragments.
        """

        # Create text stream for extractor
        self._logger.info("Creating text stream for extraction")
        text_stream = utils.text_generator(texts, max_chunk_size=10000)

        # Extract and consolidate
        self._logger.info("Extracting features")
        results = self.extractor.extract(text_stream)
        self._logger.debug(f"Extracted {len(results)} features")

        self._logger.info("Consolidating features")
        consolidated = self.consolidator.consolidate(results)
        self._logger.debug(f"Consolidated into {len(consolidated)} features")

        return consolidated
