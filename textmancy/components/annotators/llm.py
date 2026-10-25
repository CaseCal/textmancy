import json
from typing import Literal, Sequence, Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .base import Annotator


class LLMAnnotator(Annotator):
    """
    Annotates text by asking an LLM which target indices appear in each chunk.

    The output type only allows valid indices, so an out-of-range answer fails validation and
    is retried rather than silently dropped.

    Attributes:
        agent (Agent): The Pydantic AI agent that performs annotation.
    """

    def __init__(
        self,
        targets: Sequence[BaseModel],
        model: Union[str, Model] = "openai:gpt-4o",
        max_concurrency: int = 10,
        retries: int = 1,
    ):
        super().__init__(targets, max_concurrency)

        index_type = Literal[tuple(range(len(self.targets)))]
        self.agent = Agent(
            model,
            output_type=list[index_type],
            system_prompt=self._create_system_prompt(),
            retries=retries,
            defer_model_check=True,
        )

    def _create_system_prompt(self) -> str:
        listing = "\n".join(
            f"[{i}] {json.dumps(target.model_dump(mode='json'))}"
            for i, target in enumerate(self.targets)
        )
        return (
            f"You are annotating a text sample to identify {self.target_name}s. "
            f"Here is the indexed list of valid {self.target_name}s:\n{listing}\n"
            f"Return the index of every {self.target_name} that appears in, or is referred to "
            "by, the text. Return an empty list if none appear."
        )

    async def _annotate_chunk_async(self, chunk: str) -> set[int]:
        result = await self.agent.run(f"Text:\n{chunk}")
        return set(result.output or [])
