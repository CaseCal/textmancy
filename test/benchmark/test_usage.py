from types import SimpleNamespace

import pytest
from pydantic_ai import Agent

from benchmark.usage import UsageTracker, UsageTrackingClient, UsageTrackingModel, estimate_cost
from textmancy.utils import run_sync


def test_tracking_model_counts_requests_and_tokens(recording_model):
    model = recording_model(lambda system, user, i: [0])
    tracker = UsageTracker()
    agent = Agent(UsageTrackingModel(model.model, tracker), output_type=list[int])
    agent.run_sync("hello")
    agent.run_sync("again")
    assert tracker.requests == 2
    assert tracker.input_tokens > 0 and tracker.output_tokens > 0


def test_estimate_cost_uses_table_and_overrides():
    tracker = UsageTracker(requests=1, input_tokens=1_000_000, output_tokens=500_000)
    assert estimate_cost("openai:gpt-4o", tracker) == pytest.approx(2.50 + 5.00)
    assert estimate_cost("openai:gpt-4o", tracker, price_in=1, price_out=2) == pytest.approx(2)
    assert estimate_cost("unknown:model", tracker) is None


class FakeTypeSafeClient:
    async def system_one(self, state, questions, *, model=None):
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=120, output_tokens=3))


def test_tracking_client_counts_requests_and_tokens():
    tracker = UsageTracker()
    client = UsageTrackingClient(FakeTypeSafeClient(), tracker)
    run_sync(client.system_one("a", {}))
    run_sync(client.system_one(state="b", questions={}, model="jev-latest"))
    assert (tracker.requests, tracker.input_tokens, tracker.output_tokens) == (2, 240, 6)
