import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
        self._lock = threading.Lock()
        self.model = FunctionModel(self._handle)

    def _handle(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parts = [part for message in messages for part in getattr(message, "parts", [])]
        system = "".join(p.content for p in parts if isinstance(p, SystemPromptPart))
        user = next(p.content for p in parts if isinstance(p, UserPromptPart))
        with self._lock:
            self.calls.append({"system": system, "user": user})
            call_index = len(self.calls) - 1
        output = self._respond(system, user, call_index)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"response": output})])


@pytest.fixture
def recording_model() -> type[RecordingModel]:
    return RecordingModel


class OpenAIStub:
    """
    A local OpenAI Chat Completions server that answers every request by calling the output tool
    with `response`.

    Attributes:
        url (str): The base URL to point an OpenAI provider at.
        response (Any): Sent as the output tool's "response" argument.
        ports (list[int]): The client port of each request, one per connection used.
    """

    def __init__(self):
        self.response: Any = [0]
        self.ports: list[int] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.ports.append(self.client_address[1])
                tool = body["tools"][0]["function"]["name"]
                data = json.dumps(stub._reply(tool)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def _reply(self, tool: str) -> dict:
        call = {"name": tool, "arguments": json.dumps({"response": self.response})}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": call}],
        }
        return {
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@pytest.fixture
def openai_stub(monkeypatch) -> OpenAIStub:
    """Serves an OpenAIStub that both OpenAI providers and openai-chat: model strings reach."""
    stub = OpenAIStub()
    thread = threading.Thread(target=stub._server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", True)
    monkeypatch.setenv("OPENAI_BASE_URL", stub.url)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    yield stub
    stub._server.shutdown()
    stub._server.server_close()
