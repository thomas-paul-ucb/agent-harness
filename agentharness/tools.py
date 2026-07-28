"""Tool registration and safe execution.

This is the piece that stops a single flaky tool (a timed-out search API,
a rate-limited endpoint) from silently corrupting the agent's reasoning.
Every tool call goes through here — never call a tool function directly
from the loop.

If a ToolCache is provided, exact-match calls are served from cache
before the real tool function ever runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from agentharness.tool_cache import ToolCache
from agentharness.types import ToolCall, ToolResult


@dataclass
class Tool:
    name: str
    description: str
    parameters_schema: dict[str, Any]  # JSON-schema-like dict, passed to the model
    fn: Callable[..., Any]
    max_retries: int = 2
    backoff_seconds: float = 1.0
    cacheable: bool = True  # set False for tools whose output changes call to call (e.g. "current time")


class ToolRegistry:
    """Holds available tools and executes calls safely, with optional caching."""

    def __init__(self, cache: ToolCache | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._cache = cache

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas in the shape the Claude API expects."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, serving from cache on an exact-match hit,
        with retry + backoff on an actual (uncached) execution.

        Always returns a ToolResult — never raises. A failed tool call is
        data the agent can reason about, not an exception to catch.
        """
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=f"Unknown tool '{call.name}'",
            )

        if self._cache is not None and tool.cacheable:
            cached = self._cache.lookup(call.name, call.arguments)
            if cached is not None:
                return ToolResult(call_id=call.call_id, success=True, output=cached, attempts=0)

        last_error: str | None = None
        for attempt in range(1, tool.max_retries + 2):  # +1 initial try
            try:
                output = tool.fn(**call.arguments)
                if self._cache is not None and tool.cacheable:
                    self._cache.store(call.name, call.arguments, output)
                return ToolResult(
                    call_id=call.call_id,
                    success=True,
                    output=output,
                    attempts=attempt,
                )
            except Exception as exc:  # noqa: BLE001 - deliberately broad, tool code is untrusted
                last_error = str(exc)
                if attempt <= tool.max_retries:
                    time.sleep(tool.backoff_seconds * attempt)  # linear backoff
                continue

        return ToolResult(
            call_id=call.call_id,
            success=False,
            error=last_error,
            attempts=tool.max_retries + 1,
        )