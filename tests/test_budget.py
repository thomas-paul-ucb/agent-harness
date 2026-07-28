"""Tests for TokenBudget's semantic scoring and trimming behavior.

Uses a fake Anthropic client (no real API key needed) since only
count_tokens is called by TokenBudget, and we can stub that cheaply.
The embedding model (sentence-transformers) runs locally — also no
API key required, just a one-time local model download.
"""

from unittest.mock import MagicMock

from agentharness.budget import TokenBudget


def make_fake_client(token_count: int):
    """A minimal stand-in for anthropic.Anthropic that only implements
    what TokenBudget actually calls."""
    client = MagicMock()
    client.messages.count_tokens.return_value = MagicMock(input_tokens=token_count)
    return client


def tool_result_message(content: str, is_error: bool = False) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "fake-id",
                "content": content,
                "is_error": is_error,
            }
        ],
    }


def test_no_trimming_when_under_budget():
    client = make_fake_client(token_count=100)
    budget = TokenBudget(client=client, model="claude-sonnet-4-6", max_tokens=6000)

    messages = [{"role": "user", "content": "hello"}]
    result_messages, count = budget.enforce(messages, system="", task="hello")

    assert result_messages == messages  # untouched
    assert count == 100


def test_trims_irrelevant_result_before_relevant_one():
    """This is the core 'smart' behavior: given two tool results, one
    clearly about the task and one clearly not, the irrelevant one
    should be trimmed first."""
    client = MagicMock()
    # First call: over budget. Every call after: under budget, so trimming stops
    # as soon as one result is removed.
    client.messages.count_tokens.side_effect = [
        MagicMock(input_tokens=10000),  # initial check: over budget
        MagicMock(input_tokens=500),  # after trimming one block: under budget
    ]

    budget = TokenBudget(client=client, model="claude-sonnet-4-6", max_tokens=6000, keep_recent=0)

    messages = [
        {"role": "user", "content": "Plan a birthday party for my dog"},
        tool_result_message("The stock market closed up 2% today on tech earnings."),
        tool_result_message("Dog-friendly bakeries in your area: Pawsome Treats, Bark Bakery."),
    ]

    result_messages, _ = budget.enforce(messages, system="", task="Plan a birthday party for my dog")

    stock_block = result_messages[1]["content"][0]["content"]
    dog_block = result_messages[2]["content"][0]["content"]

    assert stock_block.startswith("[trimmed")  # irrelevant result got cut
    assert not dog_block.startswith("[trimmed")  # relevant result survived


def test_recent_messages_protected_by_keep_recent():
    client = MagicMock()
    client.messages.count_tokens.return_value = MagicMock(input_tokens=10000)  # always over budget

    budget = TokenBudget(client=client, model="claude-sonnet-4-6", max_tokens=6000, keep_recent=2)

    messages = [
        {"role": "user", "content": "task"},
        tool_result_message("old irrelevant result"),
        tool_result_message("recent result"),  # protected: within keep_recent window
    ]

    result_messages, _ = budget.enforce(messages, system="", task="task")

    # keep_recent=2 means the last 2 messages (indices 1 and 2) are never touched
    recent_block = result_messages[2]["content"][0]["content"]
    assert recent_block == "recent result"  # untouched, protected by keep_recent