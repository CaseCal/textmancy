from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import logging
from typing import List


from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from langchain.pydantic_v1 import BaseModel


class Annotator:
    def __init__(
        self,
        targets=List[BaseModel],
        model: str = "gpt-4",
    ):
        # logger
        self._logger = logging.getLogger(__name__)

        # Vars
        target_class = targets[0].__class__
        self.target_name = target_class.__name__
        self.target_desc = target_class.__doc__
        self.targets = targets
        self.llm = ChatOpenAI(model=model)

        # Schema
        self.json_schema = {
            "title": f"{self.target_name}s",
            "description": (
                f"List of {self.target_name}s indices that are present in the text, 0-based. For example, "  # noqa: E501
                f"if {self.targets[0]} is present in the text, then include 0 in the response."
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

    @cached_property
    def _annotation_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            f"You are tasked with annotating a text sample to identify {self.target_name}s. "
            f"Here is an ordered list of valid targets: \n {self.targets}\n"
            "Here is the text sample: \n {text} \n"
            f"Please return all of the given {self.target_name} indices that are in the text."
        )

    def _create_annotation_chain(self):
        runnable = create_structured_output_runnable(
            output_schema=self.json_schema,
            llm=self.llm,
            prompt=self._annotation_prompt,
        )
        return runnable

    def _annotate(self, text: str) -> list:
        extraction_chain = self._create_annotation_chain()
        result = extraction_chain.invoke({"text": text})
        return result["indices"]

    def annotate(self, text: str, chunk_size=4000, **kwargs) -> set:
        pool = ThreadPoolExecutor(max_workers=10)

        # Split text into chunks and extract asynchronously
        futures = []
        for i in range(0, len(text), chunk_size):
            text_chunk = text[i : i + chunk_size]
            futures.append(pool.submit(self._annotate, text_chunk, **kwargs))

        # Retrieve results in order
        results = []
        completed = 0
        for future in futures:
            # Get the result of the future
            result = future.result()
            results.extend(result)
            completed += 1
            self._logger.debug(f"Finished {completed} of {len(futures)}")

        return set(results)
