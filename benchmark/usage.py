"""Token accounting for benchmark runs."""

from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.usage import RequestUsage

# USD per million tokens: (input, output). Override with --price-in/--price-out.
PRICING: dict[str, tuple[float, float]] = {
    "openai:gpt-4o": (2.50, 10.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    # typesafe.ai lists $42 per billion input tokens; output tokens are currently free.
    "jev-latest": (0.042, 0.0),
}


@dataclass
class UsageTracker:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, usage: RequestUsage) -> None:
        """Records one request; accepts any usage object with input_tokens/output_tokens."""
        self.requests += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0

    def as_dict(self) -> dict:
        return asdict(self)


class UsageTrackingModel(WrapperModel):
    """Wraps a model and records every response's token usage."""

    def __init__(self, wrapped: Union[Model, str], tracker: UsageTracker):
        super().__init__(wrapped)
        self.tracker = tracker

    async def request(self, messages, model_settings, model_request_parameters):
        response = await super().request(messages, model_settings, model_request_parameters)
        self.tracker.add(response.usage)
        return response


class UsageTrackingClient:
    """Wraps a TypeSafe async client and records every system_one response's token usage."""

    def __init__(self, wrapped: Any, tracker: UsageTracker):
        self.wrapped = wrapped
        self.tracker = tracker

    async def system_one(self, *args, **kwargs):
        response = await self.wrapped.system_one(*args, **kwargs)
        self.tracker.add(response.usage)
        return response


def estimate_cost(
    model: str,
    tracker: UsageTracker,
    price_in: Optional[float] = None,
    price_out: Optional[float] = None,
) -> Optional[float]:
    table_in, table_out = PRICING.get(model, (None, None))
    price_in = table_in if price_in is None else price_in
    price_out = table_out if price_out is None else price_out
    if price_in is None or price_out is None:
        return None
    return (tracker.input_tokens * price_in + tracker.output_tokens * price_out) / 1_000_000
