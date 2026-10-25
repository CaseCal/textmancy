import pytest
from pydantic_ai import Agent

from benchmark.usage import UsageTracker, UsageTrackingModel, estimate_cost


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
