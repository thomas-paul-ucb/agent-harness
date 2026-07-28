"""Tests for tool execution and retry/backoff behavior.

Run with: pytest tests/
"""

from agentharness.tools import Tool, ToolRegistry
from agentharness.types import ToolCall


def test_successful_call_returns_output():
    registry = ToolRegistry()
    registry.register(
        Tool(name="echo", description="", parameters_schema={}, fn=lambda text: text)
    )
    result = registry.execute(ToolCall(name="echo", arguments={"text": "hi"}, call_id="1"))
    assert result.success is True
    assert result.output == "hi"
    assert result.attempts == 1


def test_unknown_tool_returns_error_not_exception():
    registry = ToolRegistry()
    result = registry.execute(ToolCall(name="nope", arguments={}, call_id="1"))
    assert result.success is False
    assert "Unknown tool" in result.error


def test_retries_then_succeeds():
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError("temporary failure")
        return "recovered"

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="flaky",
            description="",
            parameters_schema={},
            fn=flaky,
            max_retries=3,
            backoff_seconds=0,  # no real delay in tests
        )
    )
    result = registry.execute(ToolCall(name="flaky", arguments={}, call_id="1"))
    assert result.success is True
    assert result.output == "recovered"
    assert result.attempts == 3


def test_exhausts_retries_and_fails_gracefully():
    def always_fails():
        raise RuntimeError("permanent failure")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="broken",
            description="",
            parameters_schema={},
            fn=always_fails,
            max_retries=2,
            backoff_seconds=0,
        )
    )
    result = registry.execute(ToolCall(name="broken", arguments={}, call_id="1"))
    assert result.success is False
    assert result.attempts == 3  # 1 initial + 2 retries
    assert "permanent failure" in result.error