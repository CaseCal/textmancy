import json
import logging
from typing import Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from thefuzz import fuzz

from .. import utils


class Consolidator:
    """
    Consolidates a list of target objects by merging duplicates with an LLM.

    Attributes:
        target_class (type[BaseModel]): The class of the target objects.
        target_num (int): The desired number of target objects in the final list.
        batch_size (int): The number of targets sent per LLM call.
        max_iter (int): The maximum number of re-consolidation passes.
        tolerance (float): Re-consolidate while len(results) >= target_num * tolerance.
        agent (Agent): The Pydantic AI agent that performs consolidation.
    """

    def __init__(
        self,
        target_class: type[BaseModel],
        target_num: int = 3,
        model: Union[str, Model] = "openai-chat:gpt-4o",
        batch_size: int = 10,
        max_iter: int = 3,
        tolerance: float = 1.5,
        additional_instructions: str = "",
        max_concurrency: int = 10,
        retries: int = 1,
    ):
        self._logger = logging.getLogger(__name__)

        self.target_class = target_class
        self.target_num = target_num
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.target_name = target_class.__name__
        self.max_concurrency = max_concurrency

        self.agent = Agent(
            model,
            output_type=list[target_class],
            system_prompt=self._create_system_prompt(target_class, additional_instructions),
            retries=retries,
            defer_model_check=True,
        )

    @staticmethod
    def _create_system_prompt(
        target_class: type[BaseModel], additional_instructions: str = ""
    ) -> str:
        name = target_class.__name__
        prompt = (
            "You are an ethnographer. In this stage, text has been freely classified. "
            "You are now consolidating various annotations to create a final list. "
            f"Consolidate the given {name}s into a list of {name}s. "
            f"Where possible, combine similar {name}s into one. "
            f"Look for cases where the same {name} is referred to by different names. "
            "Avoid repetition and redundancy."
        )
        if additional_instructions:
            prompt += f"\n{additional_instructions}"
        return prompt

    async def _consolidate_batch_async(self, targets: list[BaseModel]) -> list[BaseModel]:
        payload = json.dumps([t.model_dump(mode="json") for t in targets], indent=2)
        result = await self.agent.run(f"{self.target_name}s to consolidate:\n{payload}")
        return result.output or []

    def _batches(self, items: list[BaseModel]) -> list[list[BaseModel]]:
        """Keep likely matches together without merging or discarding targets."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        groups: list[list[BaseModel]] = []
        names: list[set[str]] = []
        for item in items:
            name = getattr(item, "name", None)
            aliases = getattr(item, "known_as", [])
            values = [name, *(aliases if isinstance(aliases, list) else [])]
            keys = {
                value.strip().casefold()
                for value in values
                if isinstance(value, str) and value.strip()
            }
            matches = [
                i
                for i, existing in enumerate(names)
                if any(fuzz.ratio(a, b) >= 90 for a in keys for b in existing)
            ]
            if not matches:
                groups.append([item])
                names.append(keys)
                continue
            first = matches[0]
            groups[first].append(item)
            names[first].update(keys)
            for index in reversed(matches[1:]):
                groups[first].extend(groups.pop(index))
                names[first].update(names.pop(index))

        batches: list[list[BaseModel]] = []
        for group in groups:
            while group:
                if not batches or len(batches[-1]) == self.batch_size:
                    batches.append([])
                if len(group) <= self.batch_size and len(group) > self.batch_size - len(
                    batches[-1]
                ):
                    batches.append([])
                count = self.batch_size - len(batches[-1])
                batches[-1].extend(group[:count])
                group = group[count:]
        return batches

    async def consolidate_async(
        self, items: list[BaseModel], current_iter: int = 0
    ) -> list[BaseModel]:
        """
        Consolidates targets in batches, re-running while the result is still too large.

        Args:
            items: The targets to consolidate.
            current_iter: The current pass; used internally for recursion.

        Returns:
            The consolidated targets.
        """
        batches = self._batches(items)
        # Entering the agent resolves a model string once per call and opens the provider's HTTP
        # client on the running loop, closing it on exit. Recursive passes re-enter as a no-op.
        async with self.agent:
            results = await utils.gather_bounded(
                (self._consolidate_batch_async(batch) for batch in batches),
                self.max_concurrency,
            )
            consolidated = []
            for batch, batch_result in zip(batches, results):
                if not batch_result:
                    self._logger.warning(
                        "Consolidation returned an empty batch; retaining its input"
                    )
                    batch_result = batch
                consolidated.extend(batch_result)

            if len(consolidated) >= self.target_num * self.tolerance:
                if len(consolidated) >= len(items):
                    self._logger.warning("Consolidation stopped without reducing the target count")
                elif current_iter >= self.max_iter:
                    self._logger.warning("Consolidation reached max_iter above the target")
                else:
                    return await self.consolidate_async(consolidated, current_iter + 1)

        return consolidated

    def consolidate(self, items: list[BaseModel], current_iter: int = 0) -> list[BaseModel]:
        """Synchronous wrapper for consolidate_async."""
        return utils.run_sync(self.consolidate_async(items, current_iter))
