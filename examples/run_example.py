"""Minimal working example: an agent with a calculator and a fake search tool.

Run with:
    export ANTHROPIC_API_KEY=your-key-here
    python examples/run_example.py
"""

import os

import anthropic

from agentharness.agent import Agent
from agentharness.tools import Tool, ToolRegistry


def calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression. Deliberately naive — swap
    for a real math library if you extend this."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("Expression contains disallowed characters")
    return eval(expression)  # noqa: S307 - restricted charset above


def fake_search(query: str) -> str:
    """Stand-in for a real search API. Replace with a real HTTP call —
    this is here so the example runs without external dependencies."""
    return f"Top result for '{query}': (placeholder search result)"


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression.",
            parameters_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            fn=calculator,
        )
    )
    registry.register(
        Tool(
            name="search",
            description="Search the web for a query and return a summary.",
            parameters_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            fn=fake_search,
        )
    )
    return registry


def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    agent = Agent(client=client, tools=build_registry(), max_steps=6)

    result = agent.run("What is 47 * 12, and then search for what that number could refer to?")

    print(f"\nFinal answer: {result.final_answer}")
    print(f"Stopped reason: {result.stopped_reason}")
    print(f"\nStep-by-step trace ({len(result.steps)} steps):")
    for step in result.steps:
        print(f"  [{step.step_number}] {step.status.value}")
        if step.tool_call:
            print(f"      tool: {step.tool_call.name}({step.tool_call.arguments})")
        if step.tool_result:
            outcome = "ok" if step.tool_result.success else f"FAILED: {step.tool_result.error}"
            print(f"      result: {outcome}")


if __name__ == "__main__":
    main()