from typing import Any, Callable

import pytest
from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    SystemPromptPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

# Never let a test reach a real provider.
models.ALLOW_MODEL_REQUESTS = False

Responder = Callable[[str, str, int], Any]


class RecordingModel:
    """
    Wraps a FunctionModel that records each request's prompts and replies with canned
    structured output.

    Args:
        respond: Called as respond(system_prompt, user_prompt, call_index); its return value is
            sent as the output tool's "response" argument.
    """

    def __init__(self, respond: Responder):
        self.calls: list[dict[str, str]] = []
        self._respond = respond
        self.model = FunctionModel(self._handle)

    def _handle(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in getattr(message, "parts", [])]
        system = "".join(p.content for p in parts if isinstance(p, SystemPromptPart))
        user = next(p.content for p in parts if isinstance(p, UserPromptPart))
        self.calls.append({"system": system, "user": user})
        output = self._respond(system, user, len(self.calls) - 1)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": output})])


@pytest.fixture
def recording_model() -> type[RecordingModel]:
    return RecordingModel
