from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel


class Annotator:
    """
    A class for annotating text samples to identify target indices.

    Attributes:
        targets (List[BaseModel]): The list of target objects to identify in the text.
        model (str): The model to use for annotation.
    """

    def __init__(
        self,
        targets: List[BaseModel],
        model: str = "gpt-4o",
    ):
        # Logger
        self._logger = logging.getLogger(__name__)

        # Vars
        self.targets = targets
        self.target_class = targets[0].__class__
        self.target_name = self.target_class.__name__
        self.target_desc = self.target_class.__doc__
        self.llm = ChatOpenAI(model=model)

        # Prepare condensed example
        _max_example_size: int = 500
        example_target = targets[0]
        example = {}

        for attr, value in example_target.dict().items():
            if len(str(example)) < _max_example_size:
                example[attr] = value
            else:
                break

        # Schema
        self.json_schema = {
            "title": f"{self.target_name}s",
            "description": (
                f"List of {self.target_name}s indices that are present in the text, 0-based. For example, "  # noqa: E501
                f"if [{example}] is present in the text, then include 0 in the response."
            ),
            "type": "object",
            "properties": {
                "indices": {
                    "title": f"{self.target_name}s",
                    "description": f"List of 0-based indices of {self.target_name}s that are present in the text.",  # noqa: E501
                    "type": "array",
                    "items": {
                        "type": "number",
                        "enum": [i for i in range(len(self.targets))],
                    },
                },
            },
        }
        print(self.json_schema)

        # Create runnable
        prompt = self._create_annotation_prompt()
        self.annotation_runnable = prompt | ChatOpenAI(
            model=model
        ).with_structured_output(self.json_schema)

    def _create_annotation_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            f"You are tasked with annotating a text sample to identify {self.target_name}s. "
            f"Here is an ordered list of valid targets: \n {self.targets}\n"
            "Here is the text sample: \n {text} \n"
            f"Please return all of the given {self.target_name} indices that are in the text."
        )

    def _annotate_chunk(self, text: str) -> list:
        """
        Annotates a chunk of text to identify target indices.

        Args:
            text (str): The text chunk to annotate.

        Returns:
            list: The list of target indices found in the text chunk.
        """
        result = self.annotation_runnable.invoke({"text": text})
        return result["indices"]

    def annotate(self, text: str, chunk_size=4000, **kwargs) -> set:
        """
        Annotates the given text to identify target indices.

        Args:
            text (str): The text to annotate.
            chunk_size (int, optional): The size of text chunks for annotation. Defaults to 4000.

        Returns:
            set: A set of unique target indices found in the text.
        """
        pool = ThreadPoolExecutor(max_workers=10)
        futures = []

        # Split text into chunks and extract asynchronously
        for i in range(0, len(text), chunk_size):
            text_chunk = text[i : i + chunk_size]
            futures.append(pool.submit(self._annotate_chunk, text_chunk, **kwargs))

        # Retrieve results as they complete
        results = []
        completed = 0
        for future in as_completed(futures):
            # Get the result of the future
            result = future.result()
            results.extend(result)
            completed += 1
            self._logger.debug(f"Finished {completed} of {len(futures)}")

        return set(results)
