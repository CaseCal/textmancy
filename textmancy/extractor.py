from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from typing import Generator, Iterable, Sequence, Union

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from langchain.pydantic_v1 import BaseModel, Field, create_model


class Extractor:
    """
    A class for extracting information from a given text using a language chain.

    Attributes:
        target_class (type[BaseModel]): The class of the target to extract from the text.
        target_num (int): The expected number of targets to extract from the text.
        target_examples (list): A list of examples of the target to extract from the text.

    Methods:
        extract(text): Extracts information from the given text using the language chain.
    """

    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        target_examples: list = None,
        model: str = "gpt-4",
    ):
        # Vars
        self.target_name = target_class.__name__
        self.target_desc = target_class.__doc__
        self.target_num = target_num
        self.target_examples = target_examples or []
        self.llm = ChatOpenAI(model=model)

        # Grouped class
        fields = {
            f"{self.target_name}s": (
                Sequence[target_class],
                Field(..., description=f"A list of {self.target_name}s"),
            ),
        }
        grouped_type = create_model(f"{self.target_name}s", **fields)
        grouped_type.__doc__ = f"A list of {self.target_name}s"
        self.grouped_target_type = grouped_type

    @cached_property
    def _extraction_prompt(self) -> ChatPromptTemplate:
        prompt = (
            f"You are a literary analyst. Your job is to determine any {self.target_name}s "
            "present in a given text."
            f"{self.target_name} is defined as {self.target_desc}"
            "Here is the text: \n {text}"
            f"\n Please determine any {self.target_name}s present in the text. "
            f"Look for around {{target_num}} {self.target_name}s."
        )
        if self.target_examples:
            prompt += f"\n ex: {self.target_examples}"

        return ChatPromptTemplate.from_template(prompt)

    def _create_extraction_chain(self):
        runnable = create_structured_output_runnable(
            output_schema=self.grouped_target_type,
            llm=self.llm,
            prompt=self._extraction_prompt,
        )
        return runnable

    def _extract_from_block(self, text: str, number: int = None) -> list:
        """
        Extracts information from the given text using the language chain.

        Args:
            text (str): The text to extract information from.

        Returns:
            list: A list of extracted information from the given text.
        """
        extraction_chain = self._create_extraction_chain()
        result = extraction_chain.invoke(
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
                print(f"Generator returned {piece}")
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
            print(f"Finished {completed} of {len(futures)}")

        return results
