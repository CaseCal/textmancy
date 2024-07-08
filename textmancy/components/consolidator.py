from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Optional, Sequence, Type

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field, create_model


class Consolidator:
    """
    A class that consolidates a list of target objects into a grouped list.

    Attributes:
        target_class (Type[BaseModel]): The class of the target objects.
        target_num (int): The desired number of target objects in the grouped list.
        batch_size (int): The batch size for processing the target objects.
        max_iter (int): The maximum number of iterations for consolidation.
        tolerance (float): The tolerance level for the number of consolidated targets.
    """

    def __init__(
        self,
        target_class: Type[BaseModel],
        target_num: int = 3,
        model: str = "gpt-4o",
        batch_size: int = 10,
        max_iter: int = 3,
        tolerance: float = 1.5,
        additional_instructions: str = "",
    ):
        # Logger
        self._logger = logging.getLogger(__name__)

        # Vars
        self.target_class = target_class
        self.target_num = target_num
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.target_name = target_class.__name__
        self.target_desc = target_class.__doc__
        self.additional_instructions = additional_instructions

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
        prompt = self._create_consolidation_prompt(
            target_class, additional_instructions
        )
        self.consolidation_runnable = prompt | ChatOpenAI(
            model=model
        ).with_structured_output(self.grouped_target_type)

    @classmethod
    def _create_consolidation_prompt(
        cls,
        target_class: Type[BaseModel],
        additional_instructions: Optional[str] = None,
    ) -> ChatPromptTemplate:
        prompt = (
            "You are an ethnographer. In this stage, text has been freely classified."
            "You are now consolidating various annotations to create a final list."
            f"Please consolidate the following {target_class.__name__}s into a list: \n"
            + "{targets}"
            + f"\n Consolidate into {target_class.__name__}s."
            + f"Where possible, combine similar {target_class.__name__}s into one."
            + f"Look for cases where the same {target_class.__name__} is referred to by "
            + "different names. Avoid repetition and redundancy."
            + f"\n {additional_instructions}"
        )
        return ChatPromptTemplate.from_template(prompt)

    def _consolidate_batch(self, targets: List[BaseModel]) -> list:
        """
        Consolidates a batch of target objects using the consolidation chain.

        Args:
            targets (List[BaseModel]): The batch of target objects to be consolidated.

        Returns:
            list: The consolidated list of target objects.
        """
        result = self.consolidation_runnable.invoke({"targets": targets})
        return getattr(result, self.target_name + "s")

    def consolidate(self, items: List[BaseModel], current_iter: int = 0) -> list:
        """
        Consolidates a list of target objects into a grouped list.

        Args:
            items (List[BaseModel]): The list of target objects to be consolidated.
            current_iter (int): The current iteration count for consolidation.

        Returns:
            list: The consolidated grouped list of target objects.
        """
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self._consolidate_batch, batch) for batch in batches
            ]
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.extend(result)

        if (
            len(results) >= self.target_num * self.tolerance
            and current_iter < self.max_iter
        ):
            return self.consolidate(results, current_iter + 1)

        return results
