import json
import logging
from typing import Iterable, Optional, Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .. import utils


class Extractor:
    """
    Extracts target objects from text using an LLM.

    Attributes:
        target_class (type[BaseModel]): The class of the target to extract from the text.
        target_num (int): The expected number of targets to extract per chunk.
        agent (Agent): The Pydantic AI agent that performs extraction.
    """

    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        target_examples: Optional[list[BaseModel]] = None,
        model: Union[str, Model] = "openai:gpt-4o",
        additional_instructions: str = "",
        max_concurrency: int = 10,
        retries: int = 1,
    ):
        self._logger = logging.getLogger(__name__)

        self.target_class = target_class
        self.target_num = target_num
        self.target_name = target_class.__name__
        self.max_concurrency = max_concurrency

        self.agent = Agent(
            model,
            output_type=list[target_class],
            system_prompt=self._create_system_prompt(
                target_class, additional_instructions, target_examples
            ),
            retries=retries,
            defer_model_check=True,
        )

    @staticmethod
    def _create_system_prompt(
        target_class: type[BaseModel],
        additional_instructions: str = "",
        examples: Optional[list[BaseModel]] = None,
    ) -> str:
        name = target_class.__name__
        prompt = (
            "You are an ethnographer. This is the first stage of classification. "
            f"Your job is to determine any {name}s present in a given text. "
            f"{name} is defined as: {target_class.__doc__}"
        )
        if additional_instructions:
            prompt += f"\n{additional_instructions}"
        if examples:
            prompt += f"\nExamples: {json.dumps([e.model_dump(mode='json') for e in examples])}"
        return prompt

    async def _extract_from_block_async(
        self, text: str, number: Optional[int] = None
    ) -> list[BaseModel]:
        n = number or self.target_num
        result = await self.agent.run(f"Text:\n{text}\n\nLook for around {n} {self.target_name}s.")
        return result.output or []

    async def extract_async(
        self,
        input_data: Union[str, Iterable[str]],
        chunk_size: int = 4000,
        number: Optional[int] = None,
    ) -> list[BaseModel]:
        """
        Extracts targets from a string or a stream of strings.

        Args:
            input_data: A string (sliced into chunk_size pieces) or an iterable of strings
                (accumulated into chunks of at least chunk_size).
            chunk_size: The chunk size in characters.
            number: Overrides target_num for this call.

        Returns:
            All extracted targets, in chunk order.
        """
        chunks = utils.chunk_input(input_data, chunk_size)
        results = await utils.gather_bounded(
            (self._extract_from_block_async(chunk, number) for chunk in chunks),
            self.max_concurrency,
        )
        self._logger.debug(f"Extracted from {len(chunks)} chunks")
        return [target for chunk_result in results for target in chunk_result]

    def extract(
        self,
        input_data: Union[str, Iterable[str]],
        chunk_size: int = 4000,
        number: Optional[int] = None,
    ) -> list[BaseModel]:
        """Synchronous wrapper for extract_async."""
        return utils.run_sync(self.extract_async(input_data, chunk_size, number))
