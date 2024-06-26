from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property
from typing import List, Sequence

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.openai_functions import (
    create_structured_output_runnable,
)
from langchain.pydantic_v1 import BaseModel, Field, create_model


class Consolidator:
    """
    A class that consolidates a list of target objects into a grouped list.

    Args:
        target_class (type[BaseModel]): The class of the target objects.
        target_num (int, optional): The desired number of target objects in the grouped list.
            Defaults to 3.
        model (str, optional): The model to use for consolidation. Defaults to "gpt-4".
        batch_size (int, optional): The batch size for processing the target objects.
            Defaults to 10.
    """

    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        model: str = "gpt-4",
        batch_size: int = 10,
        max_iter: int = 3,
        tolerance: float = 1.5,
        additional_instructions: str = "",
    ):
        # Vars
        self.target_name = target_class.__name__
        self.target_desc = target_class.__doc__
        self.target_num = target_num
        self.llm = ChatOpenAI(model=model)
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.additional_instructions = additional_instructions

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

        # objects
        self.consolidation_chain = self._create_consolidation_chain()

    @cached_property
    def _consolidation_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            "You are an ethnographer. In this stage, text has been freely classified."
            "You are now consoldiating various annotations to create a final list."
            f"Please consolidate the following {self.target_name}s into a list: \n"
            + "{targets}"
            + f"\n Consolidate into {self.target_num} {self.target_name}s."
            + f"Where possible, combine similar {self.target_name}s into one."
            + f"Look for cases where the same {self.target_name} is referred to by "
            + "different names. Avoid repition and redundancy."
            + f"\n {self.additional_instructions}"
        )

    def _create_consolidation_chain(self):
        runnable = create_structured_output_runnable(
            output_schema=self.grouped_target_type,
            llm=self.llm,
            prompt=self._consolidation_prompt,
        )
        return runnable

    def consolidate(self, items: List[BaseModel], current_iter: int = 0) -> list:
        """
        Consolidates a list of target objects into a grouped list.

        Args:
            items (List[BaseModel]): The list of target objects to be consolidated.

        Returns:
            list: The consolidated grouped list of target objects.
        """
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._consolidate, batch) for batch in batches]
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.extend(result)

        if len(results) >= self.target_num * self.tolerance and current_iter < self.max_iter:
            return self.consolidate(results, current_iter + 1)

        return results

    def _consolidate(self, results: List[BaseModel]) -> list:
        """
        Consolidates a batch of target objects using the consolidation chain.

        Args:
            results (List[BaseModel]): The batch of target objects to be consolidated.

        Returns:
            list: The consolidated list of target objects.
        """
        # consolidate with agent
        consolidated_results = self.consolidation_chain.invoke({"targets": results})
        result = getattr(consolidated_results, self.target_name + "s")
        return result
