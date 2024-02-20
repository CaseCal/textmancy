from .annotator import Annotator
from .consolidator import Consolidator
from .extractor import Extractor
from . import utils

from langchain.pydantic_v1 import BaseModel


class TextMancyResult(BaseModel):
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

    def process(self, texts: list[str]) -> TextMancyResult:
        """
        Extracts featres from text, consolidates and then annotates the given text fragments.
        """

        # Create text stream for extractor
        text_stream = utils.text_generator(texts, max_chunk_size=10000)

        # Extract and consolidate
        results = self.extractor.extract(text_stream)
        consolidated = self.consolidator.consolidate(results)

        return consolidated
