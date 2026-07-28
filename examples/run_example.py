"""Minimal working example: an agent with a calculator and a fake search
tool, wired up with all four optimization layers — token budget, semantic
task cache, tool-call cache, and a metrics report at the end.

Run with:
    export ANTHROPIC_API_KEY=your-key-here
    python examples/run_example.py
"""

import os

import anthropic

from agentharness.agent import Agent
from agentharness.budget import TokenBudget
from agentharness.cache import SemanticCache
from agentharness.metrics import compute_metrics
from agentharness.tool_cache import ToolCache
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


def build_registry(tool_cache: ToolCache) -> ToolRegistry:
    registry = ToolRegistry(cache=tool_cache)
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
            cacheable=True,  # deterministic, safe to reuse forever
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
            cacheable=True,  # placeholder tool; a real search API might set this False
        )
    )
    return registry


def print_run(label: str, result) -> None:
    print(f"\n--- {label} ---")
    print(f"Final answer: {result.final_answer}")
    print(f"Cache hit: {result.cache_hit}")
    print(f"Tokens used: input={result.input_tokens}, output={result.output_tokens}")
    print(f"Steps taken: {len(result.steps)}")


def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    tool_cache = ToolCache()
    token_budget = TokenBudget(client=client, model="claude-sonnet-4-6", max_tokens=6000)
    semantic_cache = SemanticCache()

    agent = Agent(
        client=client,
        tools=build_registry(tool_cache),
        max_steps=6,
        budget=token_budget,
        cache=semantic_cache,
    )

    task = "What is 47 * 12, and then search for what that number could refer to?"

    result_1 = agent.run(task)
    print_run("Run 1 (first time seeing this task)", result_1)

    # Same task again — should hit the semantic cache and cost ~0 tokens.
    result_2 = agent.run(task)
    print_run("Run 2 (same task, should be a cache hit)", result_2)

    # A close paraphrase — should ALSO hit the cache if the threshold is tuned well.
    paraphrased_task = "Multiply 47 by 12, then look up what that result might mean."
    result_3 = agent.run(paraphrased_task)
    print_run("Run 3 (paraphrased task, testing semantic match)", result_3)

    report = compute_metrics([result_1, result_2, result_3])
    print("\n--- Metrics report ---")
    print(report.summary())


if __name__ == "__main__":
    main()