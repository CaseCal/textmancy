from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Generator, Iterable, List, Optional, Sequence, Union

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field, create_model


class Extractor:
    """
    A class for extracting information from a given text using a language chain.

    Attributes:
        target_class (type[BaseModel]): The class of the target to extract from the text.
        target_num (int): The expected number of targets to extract from the text.
        target_examples (list): A list of examples of the target to extract from the text.
    """

    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        target_examples: list = None,
        model: str = "gpt-4o",
        additional_instructions: str = "",
    ):
        # Logger
        self._logger = logging.getLogger(__name__)

        # Vars
        self.target_class = target_class
        self.target_num = target_num
        self.target_examples = target_examples or []
        self.target_name = target_class.__name__

        # Grouped class
        fields = {
            f"{self.target_name}s": (
                Sequence[target_class],
                Field(..., description=f"A list of {self.target_name}s"),
            ),
        }
        self.grouped_target_type = create_model(f"{self.target_name}s", **fields)
        self.grouped_target_type.__doc__ = f"A list of {self.target_name}s"

        # Create runnable
        prompt = self._create_extraction_prompt(
            target_class, additional_instructions, target_examples
        )
        self.extraction_runnable = prompt | ChatOpenAI(
            model=model
        ).with_structured_output(self.grouped_target_type)

    @classmethod
    def _create_extraction_prompt(
        cls,
        target_class: type[BaseModel],
        additional_instructions: Optional[str] = None,
        examples: Optional[List[BaseModel]] = None,
    ) -> ChatPromptTemplate:
        prompt = (
            "You are an ethnographer. This is the first stage of classification."
            f"Your job is to determine any {target_class.__name__}s "
            "present in a given text."
            f"{target_class.__name__} is defined as {target_class.__doc__}"
            "Here is the text: \n {text}"
            f"\n Please determine any {target_class.__name__}s present in the text. "
            f"Look for around {{target_num}} {target_class.__name__}s."
            f"\n {additional_instructions}"
        )
        if examples:
            prompt += f"\n ex: {[e.dict() for e in examples]}"

        return ChatPromptTemplate.from_template(prompt)

    def _extract_from_block(self, text: str, number: int = None) -> list:
        """
        Extracts information from the given text using the language chain.

        Args:
            text (str): The text to extract information from.

        Returns:
            list: A list of extracted information from the given text.
        """
        result = self.extraction_runnable.invoke(
            {"text": text, "target_num": number or self.target_num}
        )
        return getattr(result, self.target_name + "s")

    def extract(
        self, input_data: Union[str, Iterable, Generator], chunk_size=4000, **kwargs
    ) -> list:
        pool = ThreadPoolExecutor(max_workers=10)
        futures = []

        if isinstance(input_data, str):
            # Input is a string, process it in chunks
            for i in range(0, len(input_data), chunk_size):
                text_chunk = input_data[i : i + chunk_size]
                futures.append(
                    pool.submit(self._extract_from_block, text_chunk, **kwargs)
                )
        else:
            # Input is a stream or generator, process it iteratively
            text_chunk = ""
            for piece in input_data:
                text_chunk += piece
                if len(text_chunk) >= chunk_size:
                    # Submit the current chunk for processing
                    futures.append(
                        pool.submit(self._extract_from_block, text_chunk, **kwargs)
                    )
                    text_chunk = ""
            # Make sure to process the last chunk if it's not empty
            if text_chunk:
                futures.append(
                    pool.submit(self._extract_from_block, text_chunk, **kwargs)
                )

        # Retrieve results as they complete
        results = []
        completed = 0
        for future in as_completed(futures):
            # Get the result of the future
            result = future.result()
            # Since the result itself might be a list, we extend our result list with it
            results.extend(result)
            completed += 1
            self._logger.debug(f"Finished {completed} of {len(futures)}")

        return results
