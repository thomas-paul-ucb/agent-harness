"""The agent loop itself: decide -> act -> observe -> repeat.

This is the heart of the harness. Everything else (tools.py, budget.py)
plugs into this loop. The loop's only job is to keep the model's
autonomous behavior bounded and debuggable:

  - it forces a stop after `max_steps` iterations (no infinite loops)
  - it detects when the model calls the same tool with the same
    arguments repeatedly (a common failure mode: the model gets "stuck")
  - it records every step so a failed run can be replayed and inspected
  - it enforces a token budget on the conversation before every model
    call, trimming the least-relevant tool results first (see budget.py)
"""

from __future__ import annotations

import json

import anthropic

from agentharness.budget import TokenBudget
from agentharness.tools import ToolRegistry
from agentharness.types import (
    RunResult,
    StepRecord,
    StepStatus,
    ToolCall,
)


class Agent:
    def __init__(
        self,
        client: anthropic.Anthropic,
        tools: ToolRegistry,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 10,
        max_repeat_calls: int = 2,
        system_prompt: str = "You are a careful, methodical agent. Use tools when needed.",
        budget: TokenBudget | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.max_repeat_calls = max_repeat_calls
        self.system_prompt = system_prompt
        self.budget = budget  # optional: if None, no trimming happens

    def run(self, task: str) -> RunResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        steps: list[StepRecord] = []
        recent_calls: list[tuple[str, str]] = []  # (tool_name, json-args) history

        for step_number in range(1, self.max_steps + 1):
            if self.budget is not None:
                messages, _ = self.budget.enforce(messages, self.system_prompt, task)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.tools.schemas(),
                messages=messages,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            # Case 1: model gave a final answer, no tool call
            if not tool_use_blocks:
                final_text = "".join(b.text for b in text_blocks)
                steps.append(
                    StepRecord(
                        step_number=step_number,
                        status=StepStatus.FINAL_ANSWER,
                        model_text=final_text,
                    )
                )
                return RunResult(final_answer=final_text, steps=steps)

            # Only handle the first tool call per step for simplicity;
            # extend here for parallel tool calls if needed.
            block = tool_use_blocks[0]
            call = ToolCall(name=block.name, arguments=block.input, call_id=block.id)

            # Stuck-loop detection: same tool + same args called too many times
            signature = (call.name, json.dumps(call.arguments, sort_keys=True))
            recent_calls.append(signature)
            repeat_count = recent_calls.count(signature)
            if repeat_count > self.max_repeat_calls:
                steps.append(StepRecord(step_number=step_number, status=StepStatus.STOPPED))
                return RunResult(
                    final_answer=None,
                    steps=steps,
                    stopped_reason=(
                        f"Detected repeated identical call to '{call.name}' "
                        f"({repeat_count} times) — stopping to avoid an infinite loop."
                    ),
                )

            result = self.tools.execute(call)
            status = StepStatus.TOOL_CALLED if result.success else StepStatus.TOOL_FAILED
            steps.append(
                StepRecord(step_number=step_number, status=status, tool_call=call, tool_result=result)
            )

            # Feed the assistant's turn (including the tool_use block) back in
            messages.append({"role": "assistant", "content": response.content})

            # Feed the tool result back in the shape the API expects.
            # Failures are surfaced as content, not swallowed — the model
            # needs to know a tool call failed so it can adapt.
            result_content = (
                json.dumps(result.output)
                if result.success
                else f"Tool call failed after {result.attempts} attempt(s): {result.error}"
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call.call_id,
                            "content": result_content,
                            "is_error": not result.success,
                        }
                    ],
                }
            )

        # Ran out of steps without a final answer
        return RunResult(
            final_answer=None,
            steps=steps,
            stopped_reason=f"Reached max_steps ({self.max_steps}) without a final answer.",
        )