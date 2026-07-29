# agent-harness

A control layer for Claude tool-use agents. It manages the **decide → act → observe → repeat** loop that turns a raw LLM into a working agent - plus three layers of token/cost optimization that most agent demos skip entirely.

## What it does

**Core loop**
- **Bounded execution** - a hard `max_steps` ceiling so an agent can never run forever
- **Stuck-loop detection** - stops the agent if it calls the same tool with the same arguments too many times in a row
- **Retry with backoff** - tool failures are retried automatically, then surfaced to the model as data it can reason about, not swallowed as exceptions
- **Full step tracing** - every decision, tool call, and result is recorded for replay/debugging

**Token optimization (three layers, three different scopes)**
- **Semantic task cache** (`cache.py`) - before running a task at all, checks if a similar task was already solved (embedding similarity via chromadb). A hit skips the whole loop, ~0 tokens.
- **Token budget / context assembler** (`budget.py`) - within a run, scores every tool result by semantic relevance to the current task (not just recency) and trims the least-relevant ones first when context grows too large.
- **Tool-call cache** (`tool_cache.py`) - exact-match memoization per `(tool_name, arguments)`, so the same tool call is never re-executed, even across different tasks. Per-tool opt-out (`cacheable=False`) for tools whose output changes over time.

**Metrics** (`metrics.py`) — turns a batch of runs into a cache-hit rate and an estimated token-savings comparison against a naive baseline.

## Why this exists

Giving an LLM tools and letting it work autonomously is easy to demo and hard to make reliable *or* cheap. Left unmanaged, an agent can loop forever, silently fail when a tool times out, drift off-task, and burn tokens re-sending context that's no longer relevant. This harness is the layer between "the model decided to do something" and "that happens safely, without redoing work it's already done."

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python examples/run_example.py
```

The example runs one task, runs it again (semantic cache hit), then a paraphrased version (tests the similarity threshold), and prints a metrics report.

## Project structure

```
agentharness/
  agent.py       - core decide/act/observe loop, wires in budget + cache
  tools.py       - tool registration, safe execution, retry/backoff, tool cache
  budget.py      - context assembler: relevance-based trimming under a token ceiling
  cache.py       - semantic task-level cache (chromadb)
  tool_cache.py  - exact-match tool-call memoization (sqlite)
  metrics.py     - cache hit rate + estimated token savings
  types.py       - shared data types
examples/
  run_example.py - runnable demo wiring all layers together
tests/
  test_tools.py, test_budget.py, test_cache.py
```

## Design notes

- Every optimization layer is optional (`budget=None`, `cache=None` both work) — the core loop runs standalone without any of them.
- The task cache uses a similarity threshold (default 0.90) — tuned conservatively to avoid returning a wrong cached answer for a different question. Paraphrase matching isn't guaranteed at this threshold; that's a deliberate precision/recall tradeoff, not a bug.
- Metrics report an *estimated* token-savings number, derived from your own non-cached runs' average cost — not a measured before/after, since tasks aren't run twice just to compare.

## Possible next steps

Structured output validation for malformed tool arguments, TTL-based expiry for tool cache entries, and multi-agent orchestration (planner/executor split) would be natural extensions.

## License

MIT
