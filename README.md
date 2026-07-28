# agent-harness

A control layer for Claude tool-use agents. It manages the **decide → act → observe → repeat** loop that turns a raw LLM into a working agent, with the guardrails that loop needs to be reliable in practice:

- **Bounded execution** — a hard `max_steps` ceiling so an agent can never run forever
- **Stuck-loop detection** — stops the agent if it calls the same tool with the same arguments too many times in a row (a common failure mode where the model gets stuck retrying the same thing)
- **Retry with backoff** — tool failures (timeouts, flaky APIs) are retried automatically before being surfaced to the model as data it can reason about, not swallowed as exceptions
- **Full step tracing** — every decision, tool call, and result is recorded so a run can be replayed and debugged after the fact

## Why this exists

Giving an LLM tools and letting it work autonomously is easy to demo and hard to make reliable. Left unmanaged, an agent can loop forever, call a tool with bad arguments, silently fail when an API times out, or drift off-task with no way to tell why. This harness is the layer between "the model decided to do something" and "that something actually, safely happens."

## Status

Early / in progress. Current focus: the core loop (`agent.py`) and safe tool execution (`tools.py`). Planned next: a token-budget module that manages context growth in long-running loops (see Roadmap).

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python examples/run_example.py
```

## Project structure
agentharness/
agent.py - the core decide/act/observe loop and stop conditions
tools.py - tool registration + safe execution with retry/backoff
types.py - shared data types (ToolCall, ToolResult, RunResult, ...)
examples/
run_example.py - a runnable agent with a calculator and a search tool
tests/
test_tools.py - tests for retry/backoff and error handling

## Roadmap

- [x] Core agent loop with bounded execution and stuck-loop detection
- [x] Safe tool execution with retry + backoff
- [ ] Structured output validation (reject/retry malformed tool arguments)
- [ ] Token-budget module — track context size across loop iterations and
      summarize/drop low-priority history before it blows the context window
- [ ] Metrics: steps-to-completion, tool failure rate, tokens per run
- [ ] Multi-agent orchestration (planner/executor split)

## License

MIT